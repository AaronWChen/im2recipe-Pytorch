"""Tokenize Recipe1M instructions, one instruction per line.

    python -m skipinstr.tokenize_instructions --dataset data/recipe1M --out-dir data/skipinstr

For each partition (train / val / test) this writes
    instructions_<part>.txt         one tokenized instruction per line, recipes contiguous
    instructions_<part>.index.tsv   recipe_id <TAB> number of instruction lines
and, with --w2v-corpus, tokenized_instructions_<part>.txt in the format the
original scripts/tokenize_instructions.py produced for word2vec training
(one recipe per line, instructions separated by a tab).

Tokenization matches the original script: detected ingredient names found in
the instruction text are joined with underscores ("olive oil" -> "olive_oil"),
and punctuation is split into separate tokens. New: whitespace inside an
instruction (newlines, tabs, runs of spaces) is collapsed so every
instruction is exactly one line, and instructions are never merged or dropped.
"""
import argparse
import json
import os

import numpy as np

from .data import write_index

PUNCT = [",", ".", ";", "(", ")", "?", "!", "&", "%", ":", "*", '"']


def tok(text, punct=PUNCT):
    for p in punct:
        text = text.replace(p, f" {p} ")
    return text


def tokenize_recipe(instruction_texts, valid_ingredients, lower=False):
    """List of raw instruction strings -> list of tokenized strings (same length)."""
    if not instruction_texts:
        return []
    joined = "\t".join(" ".join(t.split()) for t in instruction_texts)
    for name in valid_ingredients:
        if " " in name:  # single words are unchanged by underscoring
            joined = joined.replace(name, name.replace(" ", "_"))
    if lower:
        joined = joined.lower()
    joined = tok(joined)
    return [" ".join(part.split()) for part in joined.split("\t")]


def _progress(iterable, **kw):
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kw)
    except ImportError:
        return iterable


def process(layer1_path, det_path, out_dir, partitions=("train", "val", "test"), lower=False, w2v_corpus=False):
    with open(det_path, "rt", encoding="utf-8") as f:
        det = {d["id"]: d for d in json.load(f)}
    with open(layer1_path, "rt", encoding="utf-8") as f:
        layer1 = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    sent = {p: open(os.path.join(out_dir, f"instructions_{p}.txt"), "wt", encoding="utf-8", newline="\n") for p in partitions}
    w2v = {p: open(os.path.join(out_dir, f"tokenized_instructions_{p}.txt"), "wt", encoding="utf-8", newline="\n")
           for p in partitions} if w2v_corpus else {}
    ids = {p: [] for p in partitions}
    rlens = {p: [] for p in partitions}
    word_counts = []  # words per instruction, train partition (for choosing --maxlen)
    missing_det = 0

    try:
        for entry in _progress(layer1, desc="recipes"):
            part = entry["partition"]
            if part not in sent:
                continue
            d = det.get(entry["id"])
            if d is None:
                missing_det += 1
                valid = []
            else:
                valid = list(dict.fromkeys(
                    i["text"] for i, ok in zip(d["ingredients"], d["valid"]) if ok))
            lines = tokenize_recipe([i["text"] for i in entry["instructions"]], valid, lower)
            for line in lines:
                sent[part].write(line + "\n")
            if part in w2v:
                w2v[part].write("\t".join(lines) + "\n")
            ids[part].append(entry["id"])
            rlens[part].append(len(lines))
            if part == "train":
                word_counts.extend(len(line.split()) for line in lines)
    finally:
        for f in list(sent.values()) + list(w2v.values()):
            f.close()

    for p in partitions:
        write_index(os.path.join(out_dir, f"instructions_{p}.index.tsv"), ids[p], rlens[p])
        print(f"{p}: {len(ids[p])} recipes, {sum(rlens[p])} instructions")
    if missing_det:
        print(f"warning: {missing_det} recipes had no det_ingrs entry (no ingredient underscoring)")
    if word_counts:
        wc = np.asarray(word_counts)
        pct = {q: int(np.percentile(wc, q)) for q in (50, 90, 95, 99)}
        print(f"train words per instruction: {pct}, max {wc.max()}; "
              f"{100 * np.mean(wc > 29):.2f}% exceed 29 words (the default --maxlen 30 truncates these)")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="data/", help="folder with layer1.json and det_ingrs.json")
    p.add_argument("--out-dir", default="data/skipinstructions")
    p.add_argument("--partitions", nargs="+", default=["train", "val", "test"])
    p.add_argument("--lower", action="store_true", help="lowercase after ingredient joining (off by default, as in the original)")
    p.add_argument("--w2v-corpus", action="store_true", help="also write the recipe-per-line file for word2vec")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    process(os.path.join(a.dataset, "layer1.json"), os.path.join(a.dataset, "det_ingrs.json"),
            a.out_dir, tuple(a.partitions), a.lower, a.w2v_corpus)


if __name__ == "__main__":
    main()
