"""Load word2vec .bin files (replaces torchwordemb).

The ingredient ids written by scripts/build_dataset.py are row numbers into this matrix, so
every row must stay at its file position. The reader is plain numpy: the format is a text
header "<n> <dim>\n" followed by n records of  word, one space, dim little-endian float32,
and (usually) a newline.

gensim is deliberately not used: for a word that appears twice, gensim keeps only the first
vector and leaves the duplicate's row as zeros, which would silently corrupt the matrix.
"""
import numpy as np


def load_word2vec_bin(path):
    """Return (words, vectors): list[str] and float32 (n, dim) array, both in file order."""
    with open(path, "rb") as f:
        n, dim = (int(x) for x in f.readline().split())
        vectors = np.empty((n, dim), dtype="<f4")
        words = []
        for i in range(n):
            chars = bytearray()
            while True:
                c = f.read(1)
                if not c:
                    raise EOFError(f"{path}: ended while reading word {i} of {n}")
                if c == b" ":
                    break
                if c != b"\n" or chars:  # skip the newline that follows the previous vector
                    chars += c
            words.append(chars.decode("utf-8", errors="replace"))
            buf = f.read(4 * dim)
            if len(buf) != 4 * dim:
                raise EOFError(f"{path}: ended inside the vector of word {i} ({words[-1]!r}) of {n}")
            vectors[i] = np.frombuffer(buf, dtype="<f4")
    return words, vectors.astype(np.float32, copy=False)


def ingredient_row_offset(words, vocab_lines):
    """How far vocab.txt is shifted relative to the rows of the word2vec matrix.

    Returns d such that words[i + d] == vocab_lines[i] for every line i, or None if the
    two lists are not the same sequence at any constant shift. build_dataset.py gives
    vocab line i the id i + 2 (0 is padding, 1 is the end token) and the model looks that id
    up as a row of the matrix, so the matching offset is 2.
    """
    if not vocab_lines or vocab_lines[0] not in words:
        return None
    d = words.index(vocab_lines[0])
    if d + len(vocab_lines) > len(words):
        return None
    return d if all(words[i + d] == w for i, w in enumerate(vocab_lines)) else None


def write_vocab_txt(w2v_path, out_path=None):
    """Write the words of a word2vec .bin, one per line in file order, next to it as vocab.txt."""
    import os
    words, _ = load_word2vec_bin(w2v_path)
    out_path = out_path or os.path.join(os.path.dirname(os.path.abspath(w2v_path)), "vocab.txt")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(words) + "\n")
    return out_path
