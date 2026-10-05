"""Vocabulary helpers.

The vocabulary is a plain text file with one token per line; the line number
is the token id. Ids 0 and 1 are reserved for <eos> and <unk>.
"""
from collections import Counter

EOS_TOKEN, UNK_TOKEN = "<eos>", "<unk>"
EOS, UNK = 0, 1


def open_text(path):
    # newline="\n": only "\n" ends a line, and nothing is translated. Line
    # numbers therefore always agree with the recipe index files.
    return open(path, "rt", encoding="utf-8", newline="\n")


def build_vocab(path, vocab_size=20000, min_count=1):
    """Most frequent words in a one-sentence-per-line file (already tokenized)."""
    counts = Counter()
    with open_text(path) as f:
        for line in f:
            counts.update(line.split())
    words = [w for w, c in counts.most_common() if c >= min_count and w not in (EOS_TOKEN, UNK_TOKEN)]
    return [EOS_TOKEN, UNK_TOKEN] + words[: max(vocab_size - 2, 0)]


def save_vocab(itos, path):
    with open(path, "wt", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(itos) + "\n")


def load_vocab(path):
    with open_text(path) as f:
        itos = [line.rstrip("\n") for line in f if line.rstrip("\n")]
    if itos[:2] != [EOS_TOKEN, UNK_TOKEN]:
        raise ValueError(f"{path}: first two entries must be {EOS_TOKEN} and {UNK_TOKEN}")
    return itos
