"""Train the skip-thought model on one-instruction-per-line text.

    python -m skipinstr.train --train-file instructions_train.txt \
        --train-index instructions_train.index.tsv --out-dir runs/skip1
"""
import argparse
import os
import time
from collections import deque

import numpy as np
import torch

from .data import SentenceCorpus
from .model import UniSkip
from .vocab import build_vocab, load_vocab, save_vocab


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-file", required=True)
    p.add_argument("--train-index", default=None, help="recipe index; masks pairs that cross recipes")
    p.add_argument("--val-file", default=None)
    p.add_argument("--val-index", default=None)
    p.add_argument("--out-dir", default="skipinstr_out")
    p.add_argument("--vocab", default=None, help="default: <out-dir>/vocab.txt, built from the train file if missing")
    p.add_argument("--vocab-size", type=int, default=20000)
    p.add_argument("--min-count", type=int, default=1)
    p.add_argument("--maxlen", type=int, default=30, help="max steps per sentence, including <eos>")
    p.add_argument("--word-size", type=int, default=620)
    p.add_argument("--thought-size", type=int, default=1024, help="must equal --stDim of the trijoint model")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--iters", type=int, default=1_000_000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


@torch.no_grad()
def evaluate(model, corpus, batch_size, n_batches, device):
    model.eval()
    starts = np.linspace(0, len(corpus) - batch_size, n_batches).astype(int)
    total = 0.0
    for s in starts:
        tokens, lengths, valid = corpus.batch(int(s), batch_size)
        total += model(tokens.to(device), lengths.to(device), valid.to(device)).item()
    model.train()
    return total / len(starts)


def save_checkpoint(path, model, optimizer, it, config, best):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "iter": it, "config": config, "best": best}, path)


def run(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    ckpt = torch.load(args.resume, map_location=device, weights_only=True) if args.resume else None

    vocab_path = args.vocab or os.path.join(args.out_dir, "vocab.txt")
    if os.path.exists(vocab_path):
        itos = load_vocab(vocab_path)
    elif ckpt is not None:
        raise FileNotFoundError(f"resuming needs the original vocabulary at {vocab_path}")
    else:
        itos = build_vocab(args.train_file, args.vocab_size, args.min_count)
        save_vocab(itos, vocab_path)
        print(f"built vocabulary of {len(itos)} tokens -> {vocab_path}")

    config = ckpt["config"] if ckpt else {
        "vocab_size": len(itos), "word_size": args.word_size,
        "thought_size": args.thought_size, "maxlen": args.maxlen}
    if config["vocab_size"] != len(itos):
        raise ValueError("vocabulary size does not match the checkpoint")

    train = SentenceCorpus(args.train_file, itos, config["maxlen"], args.train_index)
    val = SentenceCorpus(args.val_file, itos, config["maxlen"], args.val_index) if args.val_file else None
    for name, corpus in (("train", train), ("val", val)):
        if corpus is not None and len(corpus) < args.batch_size:
            raise ValueError(f"{name} corpus has fewer than --batch-size ({args.batch_size}) sentences")
    print(f"{len(train)} training sentences" + (f", {len(val)} validation sentences" if val else ""))

    model = UniSkip(config["vocab_size"], config["word_size"], config["thought_size"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start, best = 0, float("inf")
    if ckpt:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start, best = ckpt["iter"], ckpt["best"]
        print(f"resumed from iteration {start}")

    trail = deque(maxlen=100)
    history = []
    t0 = time.time()
    model.train()
    for it in range(start + 1, args.iters + 1):
        tokens, lengths, valid = train.random_batch(args.batch_size, rng)
        loss = model(tokens.to(device), lengths.to(device), valid.to(device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        trail.append(loss.item())
        history.append(trail[-1])
        if it % args.log_every == 0:
            print(f"iter {it}  loss {np.mean(trail):.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if it % args.eval_every == 0:
            metric = evaluate(model, val, args.batch_size, args.eval_batches, device) if val else float(np.mean(trail))
            print(f"iter {it}  {'val' if val else 'train(trailing)'} loss {metric:.4f}  best {min(best, metric):.4f}")
            if metric < best:
                best = metric
                save_checkpoint(os.path.join(args.out_dir, "skipinstr-best.pt"), model, optimizer, it, config, best)
        if it % args.save_every == 0:
            save_checkpoint(os.path.join(args.out_dir, "skipinstr-last.pt"), model, optimizer, it, config, best)

    save_checkpoint(os.path.join(args.out_dir, "skipinstr-last.pt"), model, optimizer, args.iters, config, best)
    return history


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
