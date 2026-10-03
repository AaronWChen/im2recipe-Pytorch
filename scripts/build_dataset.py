#!/usr/bin/env python
"""Build the training data for the PyTorch trijoint model (replaces mk_dataset.py).

Reads the Recipe1M json layers, the ingredient vocabulary, the class dictionary and the
skip-instruction vectors written by `python -m skipinstr.encode`, and writes one flat-array
store per partition (see recipe_store.py):

    <out-dir>/train_store/   <out-dir>/val_store/   <out-dir>/test_store/

Example, run from the scripts/ folder:

    python build_dataset.py --dataset ../data/recipe1M --vocab ../data/text/vocab.txt \
        --skip-dir ../data/skipinstr --out-dir ../data

Recipes are skipped exactly as in mk_dataset.py: too many instructions or ingredients
(>= --maxlen), no images, or listed in the removal file. Differences from mk_dataset.py:
  * the removal file is a set lookup (the old dict lookup never removed its first entry)
  * recipes missing from the skip-instruction index, or whose instruction count disagrees with
    it, are skipped and counted instead of crashing / silently mis-slicing
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))  # recipe_store.py lives next to train.py

import utils  # noqa: E402
from proc import detect_ingrs  # noqa: E402
from recipe_store import write_store  # noqa: E402

PARTITIONS = ("train", "val", "test")


def _progress(iterable, **kw):
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kw)
    except ImportError:
        return iterable


def load_skip(skip_dir, part):
    """(memmapped vectors, {recipe_id: (start, length)}) written by skipinstr.encode."""
    with open(os.path.join(skip_dir, part + ".index.json")) as f:
        index = json.load(f)
    encs = np.load(os.path.join(skip_dir, part + ".encs.npy"), mmap_mode="r")
    if encs.shape[1] != index["dim"]:
        raise ValueError(f"{part}: vectors have dim {encs.shape[1]} but the index says {index['dim']}")
    lookup = {rid: (s, n) for rid, s, n in zip(index["ids"], index["starts"], index["rlens"])}
    return encs, lookup


def build(dataset_dir, vocab_path, classes_path, skip_dir, out_dir, remove_path=None, maxlen=20,
          max_images=5, partitions=PARTITIONS, dtype=None):
    remove_ids = set()
    if remove_path and os.path.exists(remove_path):
        with open(remove_path) as f:
            remove_ids = {line.rstrip() for line in f}

    print("Loading skip-instruction vectors...")
    skip = {p: load_skip(skip_dir, p) for p in partitions}

    print("Loading dataset.")
    dataset = utils.Layer.merge([utils.Layer.L1, utils.Layer.L2, utils.Layer.INGRS], dataset_dir)
    print("Loading ingr vocab.")
    with open(vocab_path) as f:
        ingr_vocab = {w.rstrip(): i + 2 for i, w in enumerate(f)}  # +1 for lua, +1 for the end token
    ingr_vocab["</i>"] = 1

    with open(classes_path, "rb") as f:  # written by scripts/bigrams.py (a local, trusted file)
        class_dict = pickle.load(f)
        pickle.load(f)  # id2class, unused here

    rows = {p: {"ids": [], "ingrs": [], "classes": [], "images": [], "spans": []} for p in partitions}
    skipped = {"filtered": 0, "no_vectors": 0, "count_mismatch": 0}

    print("Assembling dataset.")
    for entry in _progress(dataset):
        part = entry["partition"]
        if part not in rows:
            continue
        ninstrs = len(entry["instructions"])
        ingr_detections = detect_ingrs(entry, ingr_vocab)
        ningrs = len(ingr_detections)
        imgs = entry.get("images")
        if ninstrs >= maxlen or ningrs >= maxlen or ningrs == 0 or not imgs or entry["id"] in remove_ids:
            skipped["filtered"] += 1
            continue

        span = skip[part][1].get(entry["id"])
        if span is None:
            skipped["no_vectors"] += 1
            continue
        if span[1] != ninstrs:
            skipped["count_mismatch"] += 1
            continue

        ingr_vec = np.zeros(maxlen, dtype="uint16")
        ingr_vec[:ningrs] = ingr_detections

        r = rows[part]
        r["ids"].append(entry["id"])
        r["ingrs"].append(ingr_vec)
        r["classes"].append(class_dict[entry["id"]])
        r["images"].append([im["id"] for im in imgs[:max_images]])
        r["spans"].append(span)

    for part in partitions:
        r = rows[part]
        write_store(os.path.join(out_dir, part + "_store"), r["ids"],
                    np.stack(r["ingrs"]) if r["ingrs"] else np.zeros((0, maxlen), dtype="uint16"),
                    r["classes"], r["images"], skip[part][0], r["spans"], dtype=dtype,
                    extra_meta={"partition": part, "max_images": max_images})
    print("Training samples: %d - Validation samples: %d - Testing samples: %d" % tuple(
        len(rows[p]["ids"]) if p in rows else 0 for p in PARTITIONS))
    print("Skipped: " + ", ".join(f"{k}={v}" for k, v in skipped.items()))
    if skipped["no_vectors"] or skipped["count_mismatch"]:
        print("warning: some recipes passed the filters but had no usable skip-instruction vectors; "
              "check that tokenize_instructions and encode were run on the same dataset.")
    return {p: len(rows[p]["ids"]) for p in partitions}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="../data/recipe1M/", help="folder with layer1.json, layer2.json, det_ingrs.json")
    p.add_argument("--vocab", default="../data/text/vocab.txt", help="ingredient vocabulary (one word per line)")
    p.add_argument("--classes", default="classes1M.pkl")
    p.add_argument("--remove", default=os.path.join(HERE, "remove1M.txt"))
    p.add_argument("--skip-dir", default="../data/skipinstr", help="folder with <partition>.encs.npy / .index.json")
    p.add_argument("--out-dir", default="../data/")
    p.add_argument("--maxlen", type=int, default=20)
    p.add_argument("--max-images", type=int, default=5)
    p.add_argument("--partitions", nargs="+", default=list(PARTITIONS), choices=PARTITIONS)
    p.add_argument("--dtype", choices=["float32", "float16"], default=None,
                   help="storage type of the vectors (default: same as the skipinstr output)")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    build(a.dataset, a.vocab, a.classes, a.skip_dir, a.out_dir, a.remove, a.maxlen, a.max_images,
          tuple(a.partitions), a.dtype)


if __name__ == "__main__":
    main()
