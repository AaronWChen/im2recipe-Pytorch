#!/usr/bin/env python
"""Write a small random sample of RecipeNLG's full_dataset.csv, e.g. to share or to try the scripts quickly.

    python scripts/sample_recipenlg.py full_dataset.csv --rows 5000 --out recipenlg_sample.csv

The file is ordered (the scraped recipes first, the Recipe1M ones last), so the first rows are not representative;
this samples uniformly across the whole file, streaming it in chunks, and keeps the original column names.
The sample is a few MB. It keeps the `Unnamed: 0` index column so rows can be traced back to the full file.
"""
import argparse

import pandas as pd


def sample_csv(path, out, rows=5000, seed=0, chunksize=200_000):
    total = sum(len(c) for c in pd.read_csv(path, usecols=[0], dtype=str, chunksize=chunksize))
    rows = min(rows, total)
    frac = rows / total
    parts = []
    for i, chunk in enumerate(pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=chunksize)):
        parts.append(chunk.sample(frac=frac, random_state=seed + i))
    df = pd.concat(parts).sort_index()
    df.to_csv(out, index=False)
    return total, len(df)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--rows", type=int, default=5000)
    p.add_argument("--out", default="recipenlg_sample.csv")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    total, n = sample_csv(a.csv, a.out, a.rows, a.seed)
    print(f"{n} of {total} rows -> {a.out}")


if __name__ == "__main__":
    main()
