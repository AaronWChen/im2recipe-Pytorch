"""Encode every instruction of a partition with a Hugging Face model.

    python -m textenc.encode --model gpt2 \
        --sentences data/skipinstructions/raw_instructions_train.txt --index data/skipinstructions/instructions_train.index.tsv \
        --out-prefix data/gpt2/train

--model is a Hugging Face id (gpt2, mbien/recipenlg, ...) or a local folder, e.g. a model you fine-tuned and saved
with save_pretrained. Give it the RAW instruction text (tokenize_instructions --raw), not the word2vec-style
tokenized text: the model has its own tokenizer.

Writes <prefix>.encs.npy       (num_instructions, hidden_size) float32 (or float16)
       <prefix>.index.json     {"dim", "ids", "starts" (0-based), "rlens"}   (same as skipinstructions.encode)
       <prefix>.encoder.json   how the vectors were made (model, pooling, layer, ...), for reproducibility

Then build the dataset with scripts/build_dataset.py --skip-dir <folder> and train with --stDim <hidden_size>.
"""

import argparse
import json
import os

import numpy as np
import torch

from skipinstructions.data import read_index, write_encoding_index
from skipinstructions.vocab import open_text

from .models import encode_texts, load_encoder
from .pooling import POOLINGS


def count_lines(path):
    with open_text(path) as f:
        return sum(1 for _ in f)


def encode_file(
    model,
    tokenizer,
    sentences_path,
    out_path,
    device="cpu",
    dtype=np.float32,
    window=20000,
    **kw,
):
    """Stream the file in windows of `window` lines into a memory-mapped .npy. Returns (n, dim)."""
    n = count_lines(sentences_path)
    dim = (
        model.config.hidden_size
        if hasattr(model.config, "hidden_size")
        else getattr(model.config, "n_embd", None)
    )
    if n == 0:
        np.save(out_path, np.zeros((0, dim or 0), dtype=dtype))
        return 0, dim
    out, pos = None, 0
    with open_text(sentences_path) as f:
        while True:
            lines = [line.rstrip("\n") for _, line in zip(range(window), f)]
            if not lines:
                break
            vecs = encode_texts(model, tokenizer, lines, device, **kw)
            if out is None:
                dim = vecs.shape[1]
                out = np.lib.format.open_memmap(
                    out_path, mode="w+", dtype=dtype, shape=(n, dim)
                )
            out[pos : pos + len(vecs)] = vecs.astype(dtype)
            pos += len(vecs)
    out.flush()
    return n, dim


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="Hugging Face id or local folder")
    p.add_argument("--sentences", required=True, help="raw instructions, one per line")
    p.add_argument(
        "--index", required=True, help="recipe index (instructions_<part>.index.tsv)"
    )
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--pooling", choices=POOLINGS, default="mean")
    p.add_argument(
        "--layer", type=int, default=-1, help="hidden layer to pool; -1 = the last one"
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="tokens per instruction, longer ones are truncated (default 128, or the model's limit)",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--window", type=int, default=20000, help="lines held in memory at a time"
    )
    p.add_argument(
        "--dtype",
        choices=["float32", "float16"],
        default="float32",
        help="storage type of the output",
    )
    p.add_argument(
        "--fp16", action="store_true", help="run the model in half precision (GPU only)"
    )
    p.add_argument("--l2-normalize", action="store_true")
    p.add_argument("--prefix", default="", help="text put before every instruction (some embedding models expect one)")
    p.add_argument("--suffix", default="", help="text put after every instruction")
    p.add_argument("--allow-pickle", action="store_true",
        help="accept models without safetensors weights (trusted only)",
    )
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ids, rlens = read_index(a.index)
    n_lines = count_lines(a.sentences)
    if int(rlens.sum()) != n_lines:
        raise ValueError(
            f"{a.index} covers {int(rlens.sum())} instructions but {a.sentences} has {n_lines} lines"
        )
    model, tokenizer = load_encoder(a.model, device, a.allow_pickle, half=a.fp16)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)), exist_ok=True)
    n, dim = encode_file(
        model,
        tokenizer,
        a.sentences,
        a.out_prefix + ".encs.npy",
        device,
        np.dtype(a.dtype),
        a.window,
        pooling=a.pooling,
        max_length=a.max_length,
        batch_size=a.batch_size,
        layer=a.layer,
        l2_normalize=a.l2_normalize,
        prefix=a.prefix, 
        suffix=a.suffix,
    )
    write_encoding_index(a.out_prefix + ".index.json", ids, rlens, dim)
    with open(a.out_prefix + ".encoder.json", "w") as f:
        json.dump(
            {
                "model": a.model,
                "pooling": a.pooling,
                "layer": a.layer,
                "max_length": a.max_length,
                "dim": dim,
                "l2_normalize": a.l2_normalize,
                "dtype": a.dtype,
                "fp16": a.fp16,
                "prefix": a.prefix, 
                "suffix": a.suffix, 
            },
            f,
            indent=1,
        )
    print(
        f"encoded {n} instructions from {len(ids)} recipes -> {a.out_prefix}.encs.npy  (dim {dim}; use --stDim {dim})"
    )


if __name__ == "__main__":
    main()
