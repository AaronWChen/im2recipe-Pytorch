"""Encode every instruction of a partition into a skip-instruction vector.

    python -m skipinstructions.encode --checkpoint runs/skip1/skipinstructions-best.pt \
        --sentences instructions_train.txt --index instructions_train.index.tsv \
        --out-prefix data/train

Writes <prefix>.encs.npy  array of shape (num_instructions, thought_size), float32 (or float16 with --dtype)
       <prefix>.index.json  {"dim", "ids", "starts" (0-based), "rlens"}

Recipe k's vectors are encs[starts[k] : starts[k] + rlens[k]].
"""
import argparse
import os

import numpy as np
import torch

from .data import SentenceCorpus, write_encoding_index
from .model import UniSkip
from .vocab import load_vocab


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    model = UniSkip(cfg["vocab_size"], cfg["word_size"], cfg["thought_size"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def encode_corpus(model, corpus, batch_size, device, out_path, dtype=np.float32):
    n, dim = len(corpus), model.encoder.thought_size
    if n == 0:
        np.save(out_path, np.zeros((0, dim), dtype=dtype))
        return
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=dtype, shape=(n, dim))
    for s in range(0, n, batch_size):
        tokens, lengths, _ = corpus.batch(s, min(batch_size, n - s))
        thoughts = model.encode(tokens.to(device), lengths.to(device))
        out[s:s + len(thoughts)] = thoughts.cpu().numpy().astype(dtype)
    out.flush()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sentences", required=True)
    p.add_argument("--index", required=True, help="recipe index written by tokenize_instructions")
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--vocab", default=None, help="default: vocab.txt next to the checkpoint")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                   help="storage type; float16 halves the file (the vectors are bounded, so precision loss is small)")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = load_model(args.checkpoint, device)
    itos = load_vocab(args.vocab or os.path.join(os.path.dirname(args.checkpoint), "vocab.txt"))
    if len(itos) != cfg["vocab_size"]:
        raise ValueError("vocabulary size does not match the checkpoint")
    corpus = SentenceCorpus(args.sentences, itos, cfg["maxlen"], args.index)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    encode_corpus(model, corpus, args.batch_size, device, args.out_prefix + ".encs.npy", np.dtype(args.dtype))
    write_encoding_index(args.out_prefix + ".index.json", corpus.index_ids, corpus.rlens, cfg["thought_size"])
    print(f"encoded {len(corpus)} instructions from {len(corpus.index_ids)} recipes -> {args.out_prefix}.encs.npy")


if __name__ == "__main__":
    main()
