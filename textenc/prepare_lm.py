"""Tokenize tagged recipe text into packed fixed-size blocks for language-model fine-tuning.

    python -m textenc.prepare_lm --model gpt2 --train data/recipenlg_merged/lm_train.txt \
        --val data/recipenlg_merged/lm_val.txt --holdout data/recipenlg_merged/lm_holdout.txt --out-dir data/lm

Like RecipeNLG: recipes are packed greedily into blocks of --block-size tokens without splitting a recipe across blocks;
a recipe longer than a block is skipped (counted); the rest of a block is padding, which the loss ignores.
The tokenizer gets the 13 recipe tags as special tokens and is saved to <out-dir>/tokenizer, so training and the saved model use
the same one. Output per split: <split>.bin (token ids, uint16 or uint32) + <split>.lengths.npy (real tokens per block),
and meta.json.
"""
import argparse
import json
import os

import numpy as np

from .lmdata import END_TOKEN, SPECIAL_TOKENS


def load_tokenizer(name_or_path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name_or_path)
    missing = [t for t in SPECIAL_TOKENS if t not in tok.get_vocab()]
    if missing:
        tok.add_special_tokens({"additional_special_tokens": missing})
    return tok


def pack_file(tokenizer, text_path, out_prefix, block_size, dtype, batch_lines=5000):
    """Returns (n_blocks, n_recipes, n_skipped_too_long, n_tokens)."""
    end_id = tokenizer.convert_tokens_to_ids(END_TOKEN)
    blocks, lengths = [], []
    cur, n_recipes, skipped, n_tokens = [], 0, 0, 0
    with open(out_prefix + ".bin", "wb") as binf, open(text_path, encoding="utf-8", newline="\n") as f:
        def flush():
            nonlocal blocks
            if blocks:
                np.asarray(blocks, dtype=dtype).tofile(binf)
                blocks = []

        def close_block():
            nonlocal cur
            lengths.append(len(cur))
            blocks.append(cur + [end_id] * (block_size - len(cur)))
            cur = []
            if len(blocks) >= 2000:
                flush()

        while True:
            lines = [line.rstrip("\n") for _, line in zip(range(batch_lines), f)]
            if not lines:
                break
            for ids in tokenizer(lines, add_special_tokens=False)["input_ids"]:
                if len(ids) > block_size or not ids:
                    skipped += 1
                    continue
                if len(cur) + len(ids) > block_size:
                    close_block()
                cur += ids
                n_recipes += 1
                n_tokens += len(ids)
        if cur:
            close_block()
        flush()
    np.save(out_prefix + ".lengths.npy", np.asarray(lengths, dtype=np.int32))
    return len(lengths), n_recipes, skipped, n_tokens


def prepare(model, out_dir, splits, block_size=1024):
    os.makedirs(out_dir, exist_ok=True)
    tok = load_tokenizer(model)
    dtype = np.uint16 if len(tok) < 2 ** 16 else np.uint32
    tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    meta = {"tokenizer": model, "block_size": block_size, "dtype": np.dtype(dtype).name, "vocab_size": len(tok),
            "end_id": tok.convert_tokens_to_ids(END_TOKEN), "splits": {}}
    for name, path in splits.items():
        if path is None:
            continue
        blocks, recipes, skipped, tokens = pack_file(tok, path, os.path.join(out_dir, name), block_size, dtype)
        meta["splits"][name] = {"blocks": blocks, "recipes": recipes, "skipped_too_long": skipped, "tokens": tokens,
                                "source": path}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt2", help="tokenizer to use: a Hugging Face id or folder (the model you will fine-tune)")
    p.add_argument("--train", required=True)
    p.add_argument("--val", default=None)
    p.add_argument("--holdout", default=None)
    p.add_argument("--out-dir", default="data/lm")
    p.add_argument("--block-size", type=int, default=1024)
    a = p.parse_args(argv)
    meta = prepare(a.model, a.out_dir, {"train": a.train, "val": a.val, "holdout": a.holdout}, a.block_size)
    for name, s in meta["splits"].items():
        print(f"{name}: {s['recipes']} recipes in {s['blocks']} blocks ({s['tokens']} tokens); "
              f"{s['skipped_too_long']} skipped as longer than {a.block_size} tokens")


if __name__ == "__main__":
    main()
