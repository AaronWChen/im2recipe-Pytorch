#!/usr/bin/env python
"""Merge RecipeNLG with Recipe1M and write language-model training text.

    python scripts/merge_recipenlg.py --recipenlg data/recipenlg/full_dataset.csv \
        --layer1 data/recipe1M/layer1.json --out-dir data/recipenlg_merged

RecipeNLG already contains Recipe1M's recipes (its `source` column says "Recipes1M"), but it dropped Recipe1M's ids
and train/val/test partitions. This script restores them by joining on the recipe URL, then:

  * recipe_links.csv       one row per RecipeNLG row that is also in Recipe1M: recipenlg_row, link, recipe1m_id,
                           partition, source, title. This is the key for joining anything built on Recipe1M
                           (images, trijoint embeddings, your skip-instruction vectors) to RecipeNLG's text and NER.
  * lm_train.txt           tagged recipe text (one recipe per line, RecipeNLG's format) for fine-tuning a language model,
                           WITHOUT the recipes in Recipe1M's val/test partitions and without exact copies of them
  * lm_val.txt             a small random slice of the rest (--val-fraction), for the fine-tuning validation loss
  * lm_holdout.txt         the Recipe1M val/test recipes themselves, never trained on: perplexity here shows how
                           the model does on unseen recipes from your evaluation set
  * merge_stats.json       every count, to check against the published numbers

RecipeNLG's own quality filter is applied by default (--filter recipenlg). As written in their notebook it measures the
length of the JSON *strings* in the CSV, so it effectively keeps: title >= 4 characters, ingredients and directions text
long enough (>= 2 and >= 30 characters), and no "step" or "mix all" anywhere in the directions. On the published
full_dataset.csv (2,231,142 rows) it leaves 2,101,075.
The split of a recipe into train / val is a hash of its link, so it is the same on every run.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from textenc.lmdata import recipe_to_text  # noqa: E402

COLUMNS = ["title", "ingredients", "directions", "link", "source", "NER"]
_SCHEME = re.compile(r"^https?://", re.IGNORECASE)
_BAD = re.compile(r"(step|mix all)", re.IGNORECASE)


def norm_link(url):
    """RecipeNLG stored links without the scheme; do the same to both sides, and ignore a trailing slash."""
    return _SCHEME.sub("", (url or "").strip()).rstrip("/")


def passes_recipenlg_filter(title, ingredients_str, directions_str):
    return (len(title) >= 4 and len(ingredients_str) >= 2 and len(directions_str) >= 30
            and _BAD.search(directions_str) is None)


def content_hash(title, directions):
    text = " ".join(title.lower().split()) + "\n" + " ".join(" ".join(directions).lower().split())
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()


def read_rows(csv_path, chunksize=200_000, limit=None):
    """Yield (row_number, dict) for every CSV row, streaming (the file is ~2.3 GB)."""
    import pandas as pd

    n = 0
    for chunk in pd.read_csv(csv_path, usecols=lambda c: c in COLUMNS, dtype=str, keep_default_na=False,
                             chunksize=chunksize):
        for rec in chunk.to_dict("records"):
            if limit is not None and n >= limit:
                return
            yield n, rec
            n += 1


def load_recipe1m_index(layer1_path):
    """norm(url) -> (recipe id, partition)"""
    with open(layer1_path, encoding="utf-8") as f:
        layer1 = json.load(f)
    index = {}
    for e in layer1:
        index[norm_link(e.get("url", ""))] = (e["id"], e["partition"])
    return index, layer1


def merge(recipenlg_csv, layer1_path, out_dir, holdout_partitions=("val", "test"), val_fraction=0.005,
          filter_name="recipenlg", dedup_holdout=True, chunksize=200_000, limit=None, log=print):
    os.makedirs(out_dir, exist_ok=True)
    index, layer1 = load_recipe1m_index(layer1_path)
    r1m_total = {}
    for e in layer1:
        r1m_total[e["partition"]] = r1m_total.get(e["partition"], 0) + 1
    del layer1
    log(f"Recipe1M: {len(index)} distinct urls, partitions {r1m_total}")

    def keep(rec):
        return filter_name == "none" or passes_recipenlg_filter(rec["title"], rec["ingredients"], rec["directions"])

    # pass 1: fingerprints of the held-out recipes, so exact copies of them can be kept out of training
    holdout_hashes = set()
    if dedup_holdout:
        log("pass 1/2: fingerprinting held-out recipes")
        for _, rec in read_rows(recipenlg_csv, chunksize, limit):
            hit = index.get(norm_link(rec["link"]))
            if hit and hit[1] in holdout_partitions:
                try:
                    holdout_hashes.add(content_hash(rec["title"], json.loads(rec["directions"])))
                except ValueError:
                    pass
    stats = {"rows": 0, "filtered_out": 0, "bad_json": 0, "no_ner": 0, "matched_recipe1m": 0,
             "matched_by_source": {}, "matched_by_partition": {}, "source_counts": {}, "recipes1m_source_rows": 0,
             "recipes1m_source_matched": 0, "dup_of_holdout": 0, "written": {"train": 0, "val": 0, "holdout": 0},
             "recipe1m_partition_totals": r1m_total, "holdout_partitions": list(holdout_partitions),
             "filter": filter_name, "val_fraction": val_fraction}
    seen_ids = set()
    out = {k: open(os.path.join(out_dir, f"lm_{k}.txt"), "w", encoding="utf-8", newline="\n")
           for k in ("train", "val", "holdout")}
    links = open(os.path.join(out_dir, "recipe_links.csv"), "w", encoding="utf-8", newline="")
    link_writer = csv.writer(links)
    link_writer.writerow(["recipenlg_row", "link", "recipe1m_id", "partition", "source", "title"])
    log("pass 2/2: matching, filtering and writing" if dedup_holdout else "matching, filtering and writing")
    try:
        for n, rec in read_rows(recipenlg_csv, chunksize, limit):
            stats["rows"] += 1
            src = rec["source"]
            stats["source_counts"][src] = stats["source_counts"].get(src, 0) + 1
            link = norm_link(rec["link"])
            hit = index.get(link)
            if src == "Recipes1M":
                stats["recipes1m_source_rows"] += 1
                stats["recipes1m_source_matched"] += hit is not None
            if hit:
                rid, part = hit
                stats["matched_recipe1m"] += 1
                stats["matched_by_source"][src] = stats["matched_by_source"].get(src, 0) + 1
                stats["matched_by_partition"][part] = stats["matched_by_partition"].get(part, 0) + 1
                seen_ids.add(rid)
                link_writer.writerow([n, link, rid, part, src, rec["title"]])
            if not keep(rec):
                stats["filtered_out"] += 1
                continue
            if rec["NER"] == "":
                stats["no_ner"] += 1
                continue
            try:
                ingredients, directions, ner = (json.loads(rec[k]) for k in ("ingredients", "directions", "NER"))
            except ValueError:
                stats["bad_json"] += 1
                continue
            if hit and hit[1] in holdout_partitions:
                bucket = "holdout"
            elif dedup_holdout and content_hash(rec["title"], directions) in holdout_hashes:
                stats["dup_of_holdout"] += 1
                continue
            else:
                bucket = "val" if zlib.crc32(link.encode()) % 100_000 < val_fraction * 100_000 else "train"
            out[bucket].write(recipe_to_text(rec["title"], ingredients, directions, ner) + "\n")
            stats["written"][bucket] += 1
    finally:
        for f in list(out.values()) + [links]:
            f.close()
    matched_by_part = stats["matched_by_partition"]
    stats["recipe1m_not_in_recipenlg"] = {p: total - matched_by_part.get(p, 0) for p, total in r1m_total.items()}
    with open(os.path.join(out_dir, "merge_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    return stats


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recipenlg", required=True, help="RecipeNLG full_dataset.csv")
    p.add_argument("--layer1", required=True, help="Recipe1M layer1.json")
    p.add_argument("--out-dir", default="data/recipenlg_merged")
    p.add_argument("--holdout-partitions", nargs="+", default=["val", "test"],
                   help="Recipe1M partitions kept out of training (default val test)")
    p.add_argument("--val-fraction", type=float, default=0.005)
    p.add_argument("--filter", choices=["recipenlg", "none"], default="recipenlg", dest="filter_name")
    p.add_argument("--no-dedup-holdout", action="store_true",
                   help="skip the extra pass that keeps exact copies of held-out recipes out of training")
    p.add_argument("--chunksize", type=int, default=200_000)
    p.add_argument("--limit-rows", type=int, default=None, help="only read this many CSV rows (for a quick trial)")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    s = merge(a.recipenlg, a.layer1, a.out_dir, tuple(a.holdout_partitions), a.val_fraction, a.filter_name,
              not a.no_dedup_holdout, a.chunksize, a.limit_rows)
    print(f"rows read {s['rows']}; removed by the quality filter {s['filtered_out']}; sources {s['source_counts']}")
    print(f"matched to Recipe1M {s['matched_recipe1m']} (by partition {s['matched_by_partition']}); "
          f"of the rows with source=Recipes1M, {s['recipes1m_source_matched']} of {s['recipes1m_source_rows']} matched")
    print(f"Recipe1M recipes not found in RecipeNLG: {s['recipe1m_not_in_recipenlg']}")
    print(f"written: {s['written']}; exact copies of held-out recipes dropped: {s['dup_of_holdout']}")


if __name__ == "__main__":
    main()
