"""Write vocab.txt (the words of a word2vec .bin, one per line, in file order) next to the .bin.

Usage: python get_vocab.py /path/to/vocab.bin
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from word2vec_io import write_vocab_txt  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print("Wrote %s" % write_vocab_txt(sys.argv[1]))
