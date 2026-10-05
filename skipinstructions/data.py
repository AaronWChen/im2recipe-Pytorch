"""Sentence corpus + recipe index helpers.

Corpus format: one tokenized instruction per line. Instructions of the same
recipe are on consecutive lines. An optional index file (TSV: recipe_id,
number_of_instructions) says where each recipe starts and lets training mask
out (previous, next) pairs that would straddle two different recipes.
"""
import json

import numpy as np
import torch

from .vocab import EOS, UNK, open_text


def read_index(path):
    ids, rlens = [], []
    with open_text(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                rid, n = line.split("\t")
                ids.append(rid)
                rlens.append(int(n))
    return ids, np.asarray(rlens, dtype=np.int64)


def write_index(path, ids, rlens):
    with open(path, "wt", encoding="utf-8", newline="\n") as f:
        for rid, n in zip(ids, rlens):
            f.write(f"{rid}\t{n}\n")


def write_encoding_index(path, ids, rlens, dim):
    """JSON index for an encoded partition; `starts` is 0-based."""
    rlens = np.asarray(rlens, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(rlens)[:-1]]) if len(rlens) else rlens
    with open(path, "wt", encoding="utf-8") as f:
        json.dump({"dim": int(dim), "ids": list(ids), "starts": starts.tolist(), "rlens": rlens.tolist()}, f)


class SentenceCorpus:
    """A corpus held in memory as int32 token ids.

    tokens[i]  : word ids, truncated to maxlen-1 words, then <eos>, then <eos> padding
    lengths[i] : number of real steps including the <eos> (always >= 1)
    """

    def __init__(self, path, itos, maxlen=30, index_path=None):
        stoi = {w: i for i, w in enumerate(itos)}
        with open_text(path) as f:
            n = sum(1 for _ in f)
        self.maxlen = maxlen
        self.tokens = np.full((n, maxlen), EOS, dtype=np.int32)
        self.lengths = np.ones(n, dtype=np.int32)
        with open_text(path) as f:
            for i, line in enumerate(f):
                ids = [stoi.get(w, UNK) for w in line.split()[: maxlen - 1]]
                self.tokens[i, : len(ids)] = ids
                self.lengths[i] = len(ids) + 1

        self.index_ids = self.rlens = self.line_recipe = None
        if index_path:
            self.index_ids, self.rlens = read_index(index_path)
            if int(self.rlens.sum()) != n:
                raise ValueError(f"{index_path} covers {int(self.rlens.sum())} instructions but {path} has {n} lines")
            self.line_recipe = np.repeat(np.arange(len(self.rlens)), self.rlens)

    def __len__(self):
        return len(self.lengths)

    def batch(self, start, size):
        """Contiguous window -> (tokens, lengths, pair_valid).

        pair_valid[i] says whether lines i and i+1 belong to the same recipe.
        """
        sl = slice(start, start + size)
        tokens = torch.from_numpy(self.tokens[sl].astype(np.int64))
        lengths = torch.from_numpy(self.lengths[sl].astype(np.int64))
        if self.line_recipe is not None:
            r = self.line_recipe[sl]
            pair_valid = torch.from_numpy(r[:-1] == r[1:])
        else:
            pair_valid = torch.ones(len(tokens) - 1, dtype=torch.bool)
        return tokens, lengths, pair_valid

    def random_batch(self, size, rng):
        start = int(rng.integers(0, len(self) - size + 1))
        return self.batch(start, size)
