#!/usr/bin/env python
"""Check that ingredient ids line up with the rows of the word2vec matrix.

build_dataset.py gives line i of vocab.txt the id i + 2 (0 = padding, 1 = end-of-ingredients token), and the
model reads that id as row i + 2 of its embedding. The embedding is the vocab.bin matrix with --ingr_extra_rows
zero rows put in front, so vocab line i must be row i + 2 - extra_rows of vocab.bin:

    extra rows 2 (default): vocab.txt lists the words of vocab.bin in file order   (what get_vocab.py writes)
    extra rows 0          : vocab.bin must already start with two reserved rows that vocab.txt skips

    python check_ingredient_vocab.py --w2v ../data/text/vocab.bin --vocab ../data/text/vocab.txt

Exit status 0 if the layout matches, 1 otherwise.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from word2vec_io import ingredient_row_offset, load_word2vec_bin  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--w2v", default="../data/text/vocab.bin")
    p.add_argument("--vocab", default="../data/text/vocab.txt")
    p.add_argument("--extra-rows", type=int, default=2, help="the trijoint --ingr_extra_rows you will train with")
    a = p.parse_args(argv)
    expected = 2 - a.extra_rows

    words, vectors = load_word2vec_bin(a.w2v)
    with open(a.vocab, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    print(f"{a.w2v}: {len(words)} rows of dimension {vectors.shape[1]}; {a.vocab}: {len(lines)} words")

    offset = ingredient_row_offset(words, lines)
    if offset == expected:
        print(f"OK: vocab line i is row i + {expected} of the file; with {a.extra_rows} reserved rows in front, "
              "its id i + 2 reads its own vector.")
        return 0
    if offset is None:
        print("MISMATCH: vocab.txt is not a contiguous run of the words in vocab.bin, so ingredient ids "
              "would point at unrelated vectors.")
    else:
        print(f"MISMATCH: vocab line i is row i + {offset} of the file but with {a.extra_rows} reserved rows it must be "
              f"row i + {expected}. Ingredient ids would read vectors {expected - offset:+d} rows away from their own.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
