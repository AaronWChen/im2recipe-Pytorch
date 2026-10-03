# skipinstr

Skip-thought encoder for recipe instructions: a modernized PyTorch port of
`sanyam5/skip-thoughts` (UniSkip) plus the data tooling the im2recipe trijoint
model needs. It replaces the Lua/Torch `th-skip` pipeline that produced
`encs_{train,val,test}_1024.t7`.

## Pipeline

```bash
# 1. Tokenize: one instruction per line (+ a recipe index). Prints an instruction
#    length summary so you can sanity-check --maxlen.
python -m skipinstr.tokenize_instructions --dataset data/recipe1M --out-dir data/skipinstr --w2v-corpus

# 2. Train on the train partition (vocabulary is built from it on first run).
python -m skipinstr.train \
    --train-file data/skipinstr/instructions_train.txt --train-index data/skipinstr/instructions_train.index.tsv \
    --val-file   data/skipinstr/instructions_val.txt   --val-index   data/skipinstr/instructions_val.index.tsv \
    --out-dir runs/skip1

# 3. Encode every partition.
for p in train val test; do
  python -m skipinstr.encode --checkpoint runs/skip1/skipinstr-best.pt \
      --sentences data/skipinstr/instructions_$p.txt --index data/skipinstr/instructions_$p.index.tsv \
      --out-prefix data/skipinstr/$p
done
```

Step 3 writes `<prefix>.encs.npy` (float32, `[num_instructions, 1024]`, written
incrementally so it never has to fit in RAM) and `<prefix>.index.json` with
`ids`, `starts` (**0-based**) and `rlens`. Recipe `k` is
`encs[starts[k] : starts[k] + rlens[k]]`.

Use `--w2v-corpus` output (`tokenized_instructions_train.txt`) for the word2vec
step exactly as in the original README.

## What changed from sanyam5/skip-thoughts

- Runs on current PyTorch and any device (no `Variable`, no hard-coded CUDA id).
- Encoder uses packed sequences: a sentence's vector doesn't depend on padding or batch mates.
- Loss mask is applied per token (the original masked logits).
- Optional recipe index masks (previous, next) pairs that straddle two recipes.
- Word ids 0/1 are `<eos>`/`<unk>`; vocabulary is a plain text file.
- Thought size defaults to **1024** (the original: 1200) so the trijoint `--stDim` stays 1024.
- Scripts, checkpoints with config (resumable), validation loss, gradient clipping.

## Things to know

- `--maxlen 30` counts the `<eos>`; longer instructions are truncated. Check the tokenizer's summary.
- Training loss is a sum of the previous- and next-sentence losses, as in the original.
- Word embeddings are random-initialized (as in sanyam5's code), not word2vec.
- Tested on toy data only (see `tests/`); not run on Recipe1M.

## Tests

```bash
python -m pytest skipinstr/tests -q
```
