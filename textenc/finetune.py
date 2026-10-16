"""Fine-tune a causal language model (GPT-2, a RecipeNLG checkpoint, or any AutoModelForCausalLM) on recipe text.

    python -m textenc.prepare_lm --model gpt2 --train ... --val ... --holdout ... --out-dir data/lm      # once
    python -m textenc.finetune --model gpt2 --data data/lm --out models/gpt2-recipes \
        --batch-size 8 --grad-accum 8 --epochs 1 --sample-ingredients "chicken, garlic, lemon"

Plain PyTorch loop: AdamW, linear warmup then cosine decay, gradient accumulation, bf16/fp16 autocast, gradient clipping.
Output folders (each loads with from_pretrained and holds model.safetensors + the tokenizer with the recipe tags):
    <out>/best   lowest validation loss so far        <out>/last   latest save; --resume continues from it
    <out>/log.jsonl   every logged step and evaluation        <out>/sample.txt   a generated recipe (--sample-ingredients)
<out>/last also holds trainer_state.pt (optimizer state, a torch pickle, used only by --resume).
Validation loss/perplexity use <data>/val; if <data>/holdout exists it is evaluated too (recipes from Recipe1M's val/test
partitions that were kept out of training), but it is never used to pick the best model.
Fine-tuned folders can be given to textenc.encode as --model.
"""
import argparse
import json
import math
import os
import shutil
import time

import numpy as np
import torch

from .lmdata import prompt_for


class Blocks:
    """Packed blocks on disk (see prepare_lm): token ids + how many of each block's positions are real."""

    def __init__(self, data_dir, split, meta):
        info = meta["splits"].get(split)
        self.n = info["blocks"] if info else 0
        self.size = meta["block_size"]
        if self.n:
            self.ids = np.memmap(os.path.join(data_dir, split + ".bin"), dtype=np.dtype(meta["dtype"]), mode="r",
                                 shape=(self.n, self.size))
            self.lengths = np.load(os.path.join(data_dir, split + ".lengths.npy"))

    def __len__(self):
        return self.n

    def get(self, idx, device):
        idx = np.asarray(idx)
        ids = torch.from_numpy(self.ids[idx].astype(np.int64))
        lens = torch.from_numpy(self.lengths[idx].astype(np.int64))
        mask = torch.arange(self.size)[None, :] < lens[:, None]
        return ids.to(device), mask.to(device), ids.masked_fill(~mask, -100).to(device)


def lr_at(step, total, base_lr, warmup, min_ratio=0.1):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def pick_precision(name, device):
    if name != "auto":
        return name
    return "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"


def autocast(precision, device):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None)


@torch.no_grad()
def evaluate(model, blocks, batch_size, max_blocks, device, precision):
    """Mean cross-entropy per real token over evenly spaced blocks -> (loss, perplexity)."""
    if len(blocks) == 0:
        return None
    pick = np.unique(np.linspace(0, len(blocks) - 1, min(max_blocks, len(blocks))).astype(int))
    model.eval()
    total, count = 0.0, 0
    for s in range(0, len(pick), batch_size):
        ids, mask, labels = blocks.get(pick[s:s + batch_size], device)
        with autocast(precision, device):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        n = int((labels[:, 1:] != -100).sum())
        total += float(loss) * n
        count += n
    model.train()
    mean = total / max(1, count)
    return mean, math.exp(min(mean, 50))


def save_dir(model, tokenizer, path):
    os.makedirs(path, exist_ok=True)
    use_cache = model.config.use_cache
    model.config.use_cache = True  # the saved model should generate normally; training switches the cache off
    model.save_pretrained(path)
    model.config.use_cache = use_cache
    tokenizer.save_pretrained(path)


def generate_sample(model, tokenizer, ingredients, device, max_new_tokens=400, seed=0):
    from .lmdata import END_TOKEN

    model.eval()
    # no special tokens: training blocks were packed without them, and modern tokenizers would add a BOS the model never saw
    prompt = tokenizer(prompt_for(ingredients), add_special_tokens=False, return_tensors="pt").to(device)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(**prompt, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.9, temperature=0.8,
                             eos_token_id=tokenizer.convert_tokens_to_ids(END_TOKEN),
                             pad_token_id=tokenizer.convert_tokens_to_ids(END_TOKEN))
    model.train()
    return tokenizer.decode(out[0], skip_special_tokens=False)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Hugging Face id or local folder to start from")
    p.add_argument("--data", required=True, help="folder written by textenc.prepare_lm")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=8, help="blocks per step on the device")
    p.add_argument("--grad-accum", type=int, default=8, help="steps accumulated per optimizer update")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=None, help="optimizer updates; overrides --epochs")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=1000, help="optimizer updates between evaluations")
    p.add_argument("--eval-blocks", type=int, default=200)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--grad-checkpointing", action="store_true", help="trade speed for memory")
    p.add_argument("--resume", action="store_true", help="continue from <out>/last")
    p.add_argument("--allow-pickle", action="store_true", help="start from a model that has no safetensors weights (trusted only)")
    p.add_argument("--sample-ingredients", default=None, help='e.g. "chicken, garlic, lemon": write a sample recipe at the end')
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def run(a, log=print):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    precision = pick_precision(a.precision, device)
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 needs a CUDA device; use bf16 or fp32")
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    with open(os.path.join(a.data, "meta.json")) as f:
        meta = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(a.data, "tokenizer"))
    train, val, holdout = (Blocks(a.data, s, meta) for s in ("train", "val", "holdout"))
    if len(train) == 0:
        raise ValueError(f"{a.data} has no training blocks")

    last = os.path.join(a.out, "last")
    state = None
    if a.resume:
        if not os.path.exists(os.path.join(last, "trainer_state.pt")):
            raise FileNotFoundError(f"--resume needs {last}/trainer_state.pt")
        state = torch.load(os.path.join(last, "trainer_state.pt"), map_location="cpu", weights_only=True)
        source, use_st = last, True
    else:
        source, use_st = a.model, (None if a.allow_pickle else True)
    model = AutoModelForCausalLM.from_pretrained(source, use_safetensors=use_st)
    if model.get_input_embeddings().num_embeddings < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))  # room for the recipe tags
    model.config.use_cache = False
    if a.grad_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device).train()

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=precision == "fp16")

    per_update = a.batch_size * a.grad_accum
    steps_per_epoch = max(1, len(train) // per_update)
    total = a.max_steps or max(1, int(a.epochs * steps_per_epoch))
    step, best = 0, float("inf")
    if state:
        step, best = state["step"], state["best"]
        opt.load_state_dict(state["optimizer"])
        if "scaler" in state and scaler.is_enabled():
            scaler.load_state_dict(state["scaler"])
        log(f"resumed at update {step} of {total}")
    n_params = sum(p.numel() for p in model.parameters())
    log(f"{n_params / 1e6:.1f}M parameters, {len(train)} training blocks of {train.size} tokens, "
        f"{steps_per_epoch} updates per epoch, {total} updates in total, precision {precision}, device {device}")

    def save(tag, with_state=False):
        tmp = os.path.join(a.out, tag + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        save_dir(model, tokenizer, tmp)
        if with_state:
            torch.save({"step": step, "best": best, "optimizer": opt.state_dict(), "scaler": scaler.state_dict()},
                       os.path.join(tmp, "trainer_state.pt"))
        final = os.path.join(a.out, tag)
        shutil.rmtree(final, ignore_errors=True)
        os.rename(tmp, final)

    def record(entry):
        with open(os.path.join(a.out, "log.jsonl"), "a") as f:
            f.write(json.dumps(entry) + "\n")

    def do_eval():
        nonlocal best
        entry = {"step": step}
        v = evaluate(model, val, a.batch_size, a.eval_blocks, device, precision)
        h = evaluate(model, holdout, a.batch_size, a.eval_blocks, device, precision)
        if v:
            entry.update(val_loss=v[0], val_ppl=v[1])
        if h:
            entry.update(holdout_loss=h[0], holdout_ppl=h[1])
        log("eval " + json.dumps({k: round(x, 4) if isinstance(x, float) else x for k, x in entry.items()}))
        record(entry)
        metric = v[0] if v else None
        if metric is not None and metric < best:
            best = metric
            save("best")
        return metric

    t0, tokens_seen, running = time.time(), 0, []
    while step < total:
        if len(train) < per_update:  # fewer blocks than one update needs (a tiny dataset): sample with replacement
            idx = np.random.default_rng(a.seed + step).integers(0, len(train), per_update)
        else:
            epoch, offset = divmod(step, steps_per_epoch)
            idx = np.random.default_rng(a.seed + epoch).permutation(len(train))[offset * per_update:(offset + 1) * per_update]
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total, a.lr, a.warmup_steps, a.min_lr_ratio)
        loss_sum = 0.0
        for m in range(a.grad_accum):
            ids, mask, labels = train.get(idx[m * a.batch_size:(m + 1) * a.batch_size], device)
            with autocast(precision, device):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss / a.grad_accum
            scaler.scale(loss).backward()
            loss_sum += float(loss.detach())
            tokens_seen += int(mask.sum())
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        running.append(loss_sum)
        if step % a.log_every == 0 or step == total:
            entry = {"step": step, "loss": float(np.mean(running)), "lr": opt.param_groups[0]["lr"],
                     "tokens_per_s": tokens_seen / (time.time() - t0)}
            running = []
            log("step {step}/{total} loss {loss:.4f} lr {lr:.2e} {tokens_per_s:.0f} tok/s".format(total=total, **entry))
            record(entry)
        if step % a.eval_every == 0 and step < total:
            do_eval()
        if step % a.save_every == 0 and step < total:
            save("last", with_state=True)
    do_eval()
    save("last", with_state=True)
    if not os.path.exists(os.path.join(a.out, "best")):  # no validation split: the final model is the best we have
        save("best")
    if a.sample_ingredients:
        text = generate_sample(model, tokenizer, [x.strip() for x in a.sample_ingredients.split(",")], device)
        with open(os.path.join(a.out, "sample.txt"), "w") as f:
            f.write(text + "\n")
        log("sample:\n" + text)
    return {"steps": step, "best_val_loss": best if best < float("inf") else None}


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
