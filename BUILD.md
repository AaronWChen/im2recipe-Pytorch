# Building the tri-joint model from scratch

Every command below is run from the repository root unless it says `cd scripts`. The whole sequence,
with these exact command lines, runs end to end on a tiny synthetic dataset:
`bash tests/e2e_dry_run.sh` (needs network once for the nltk data used by `bigrams.py`).
It has not been run on real Recipe1M data.

## The parts, and what each one needs

| Part | What it is | Needs first | Feeds |
|---|---|---|---|
| **Tokenized text** | one instruction per line, ingredient names joined with `_` | Recipe1M json | everything below |
| **Ingredient word2vec** (`vocab.bin`, `vocab.txt`) | embeddings of ingredient words; ingredient ids are `vocab.txt` line + 2 | tokenized text | semantic classes, dataset, `ingRNN` |
| **Semantic classes** (`classes1M.pkl`) | class label per recipe, from title bigrams + Food-101 | `vocab.txt` | dataset, semantic loss |
| **Skip-instruction encoder** (`runs/skip1/`) | sentence encoder; its vectors are the input of `stRNN` | tokenized text | dataset |
| **ResNet-50** | the image branch; ImageNet weights are downloaded by torchvision | nothing | trijoint |
| **Dataset** (`data/*_store/`) | per recipe: instruction vectors, ingredient ids, class, image names | w2v vocab + classes + skip vectors | trijoint |
| **Tri-joint model** | `ingRNN` + `stRNN` + recipe embedding, ResNet + image embedding, semantic classifier | dataset, `vocab.bin`, images | test / rank |

The skip-instruction encoder is trained **separately and frozen**: the tri-joint model only reads its output vectors.
So the tri-joint model has to be trained after, and on, the vectors from *your* encoder.

## Order of operations

```
          Recipe1M json + images
                    |
            [1] tokenize
             /               \
  [2] ingredient word2vec     [4] skip-instruction train -> [4b] encode (train, val, test)
          |                                                  |
  [3] semantic classes                                       |
             \                                              /
              +------------> [5] build dataset <------------+
                                     |
                     [6] train tri-joint (ResNet-50 downloads here)
                                     |
                           [7] test -> rank
```

After step 1, the chain 2 -> 3 and the chain 4 -> 4b do not depend on each other and can run in parallel.
Step 5 waits for both. Steps 5 to 7 are strictly sequential.

### 0. Set up and get the data

```bash
pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ('punkt', 'punkt_tab', 'stopwords')]"   # for bigrams.py
python -m pytest tests skipinstructions/tests -q      # checks the install; takes about a minute
```

All tests should pass; tests that need an optional package (`transformers`, `tokenizers`, `gensim`, `pandas`) are skipped, with the reason shown by `-rs`, if it is not installed.
Each release zip is the complete tree: replace your copy with it rather than patching. If you update file by file, or the tests show errors that look like old and
new code mixed together, run `python scripts/verify_checkout.py` (add `--package <name>` if you renamed the `skipinstructions` package); it lists every file that is missing or differs
from the release, ignoring line endings. To see what differs, unzip the release elsewhere and run
`python <release>/scripts/verify_checkout.py --root . --reference <release> --diff`. The tests run on the CPU even on a GPU machine; set `IM2RECIPE_TEST_GPU=1` to leave the GPU visible.
Keep the package folder names `skipinstructions` and `textenc`: imports, checkpoint file names (`skipinstructions-best.pt`) and the tests all use them. If the tests stop at
"The skipinstructions package that Python imports is older than these tests", an old copy of that folder is being imported; replace it with the one from the zip.

Get Recipe1M through the form linked in `README.md` and arrange it like this:

```
data/recipe1M/layer1.json  layer2.json  det_ingrs.json
data/images/               (four-level folders: 0/f/a/8/0fa8....jpg)
data/food101_classes_renamed.txt
```

### 1. Tokenize

```bash
python -m skipinstructions.tokenize_instructions --dataset data/recipe1M --out-dir data/skipinstructions --w2v-corpus
```
Writes `instructions_<part>.txt` (+ `.index.tsv`) for skip-instructions and `tokenized_instructions_<part>.txt` for word2vec.
Look at the printed instruction-length summary: instructions longer than `--maxlen` (default 30 steps) are truncated in step 4.

### 2. Ingredient word2vec

```bash
cd scripts
python train_w2v.py          # defaults: corpus ../data/skipinstructions/tokenized_instructions_train.txt, out ../data/text/vocab.bin
python check_ingredient_vocab.py
cd ..
```
Writes `data/text/vocab.bin` and `data/text/vocab.txt`. The check must print `OK`. It confirms that the id the dataset
gives an ingredient (`vocab.txt` line + 2) reads that ingredient's own vector in the model. Defaults are the flags the
README used with the C tool (skip-gram, hierarchical softmax, window 10, 10 epochs, 300 dimensions, min count 10).
`--size` must equal `--ingrW2VDim` in step 6.

### 3. Semantic classes (needs `vocab.txt` from step 2)

```bash
cd scripts
python bigrams.py --crtbgrs        # bigrams of training recipe titles -> ../data/bigrams1M.pkl
python bigrams.py --nocrtbgrs      # class labels from those bigrams + Food-101 -> ../data/classes1M.pkl
cd ..
```
The second command prints the number of classes (background included) on its last line.
If it is not 1048, pass that number as `--numClasses` in step 6. The second command loops over every recipe for each
candidate bigram in plain Python, so expect it to be slow. I have not timed it on the full dataset.

### 4. Skip-instruction encoder (independent of steps 2 and 3)

```bash
python -m skipinstructions.train \
    --train-file data/skipinstructions/instructions_train.txt --train-index data/skipinstructions/instructions_train.index.tsv \
    --val-file   data/skipinstructions/instructions_val.txt   --val-index   data/skipinstructions/instructions_val.index.tsv \
    --out-dir runs/skip1

for p in train val test; do
  python -m skipinstructions.encode --checkpoint runs/skip1/skipinstructions-best.pt \
      --sentences data/skipinstructions/instructions_$p.txt --index data/skipinstructions/instructions_$p.index.tsv \
      --out-prefix data/skipinstructions/$p
done
```
Training prints the validation loss every `--eval-every` iterations and keeps the best checkpoint as `skipinstructions-best.pt`.
The default `--iters` is 1,000,000; stop earlier once the validation loss stops improving. A resumed run uses
`--resume runs/skip1/skipinstructions-last.pt`. Keep `--thought-size 1024` equal to `--stDim` in step 6. Add `--dtype float16` to
`encode` to halve the files.

### 4b. Alternative to step 4: encode instructions with a Hugging Face model (GPT-2, RecipeNLG, your own fine-tune)

This replaces the skip-instruction encoder with any transformer, and produces files in the same format, so step 5 does not change.
It needs the instructions as plain text, because the model has its own tokenizer:

```bash
# once: also write raw_instructions_<part>.txt (rerunning step 1 rewrites the other files with identical content)
python -m skipinstructions.tokenize_instructions --dataset data/recipe1M --out-dir data/skipinstructions --raw

for p in train val test; do
  python -m textenc.encode --model gpt2 \
      --sentences data/skipinstructions/raw_instructions_$p.txt --index data/skipinstructions/instructions_$p.index.tsv \
      --out-prefix data/gpt2/$p
done
cd scripts && python build_dataset.py --skip-dir ../data/gpt2 --out-dir ../data_gpt2 && cd ..
python train.py --data_path data_gpt2/ --stDim 768 --img_path data/images256/ --ingrW2V data/text/vocab.bin --snapshots snapshots_gpt2 \
    --batch_size 64 --amp bf16 --workers 8
```

- `--model` is a Hugging Face id or a local folder, so a model you fine-tune later and save with `save_pretrained` works the same way.
- The vector size is the model's hidden size (768 for GPT-2 small). The encode command prints it, and it must be `--stDim` in steps 6 and 7.
- `--pooling` is `mean` (default), `last` or `first` (the `[CLS]` position of BERT-style models). `--layer` picks an earlier hidden layer. `--l2-normalize` rescales the vectors.
  GPT-2 states are not bounded like skip-instruction vectors, so if training is unstable, try `--l2-normalize`.
- Each `<prefix>.encoder.json` records how the vectors were made.
- Instructions are truncated to `--max-length` tokens (default 128, or the model's limit if lower).
- Weights must be safetensors. A model that only has a pickled `.bin` is refused with a message; convert it once (this needs `--allow-pickle`, so only for sources you trust):
  `python -m textenc.to_safetensors mbien/recipenlg --out models/recipenlg --allow-pickle`. The converter keeps the model's original class,
  so it can still generate text and be fine-tuned, and it checks that every tensor is identical after the round trip.
- RecipeNLG's text includes Recipe1M recipes, and its own split ignores Recipe1M's partitions. A RecipeNLG-tuned model has therefore very likely seen your val and test text (not images).
  Compare it with plain `gpt2` as a control.

### 4c. Merge RecipeNLG and fine-tune a language model on the combined text (optional)

RecipeNLG already contains Recipe1M's recipes (its `source` column), but without Recipe1M's ids and partitions. The merge restores them
by joining on the recipe URL, and writes training text that never includes your Recipe1M val and test recipes:

```bash
# RecipeNLG's full_dataset.csv is about 2.3 GB; this makes a ~5 MB random sample of it, handy for trying things
python scripts/sample_recipenlg.py data/recipenlg/full_dataset.csv --rows 5000 --out recipenlg_sample.csv

python scripts/merge_recipenlg.py --recipenlg data/recipenlg/full_dataset.csv --layer1 data/recipe1M/layer1.json \
    --out-dir data/recipenlg_merged
```
Outputs in `data/recipenlg_merged/`: `recipe_links.csv` (RecipeNLG row <-> Recipe1M id and partition, the key for joining everything built on Recipe1M),
`lm_train.txt`, `lm_val.txt` (a 0.5% slice for the validation loss), `lm_holdout.txt` (the Recipe1M val/test recipes, never trained on) and `merge_stats.json`.
Check the printed numbers: on the published file there should be 2,231,142 rows, and 2,101,075 after RecipeNLG's own quality filter (those two figures come from
their notebooks; I have not run the script on the real file). Of the rows with `source=Recipes1M`, essentially all should match a Recipe1M url; if they do not, the
link normalization needs a look. Rows that are exact copies of held-out recipes (same title and directions, different site) are dropped from training as well.

```bash
python -m textenc.prepare_lm --model gpt2 --train data/recipenlg_merged/lm_train.txt --val data/recipenlg_merged/lm_val.txt \
    --holdout data/recipenlg_merged/lm_holdout.txt --out-dir data/lm
python -m textenc.finetune --model gpt2 --data data/lm --out models/gpt2-recipes \
    --batch-size 8 --grad-accum 8 --epochs 1 --sample-ingredients "chicken, garlic, lemon"
```
- The text format is RecipeNLG's: ingredient names first, then ingredient lines, steps, and the title last, with 13 tag tokens that `prepare_lm` adds to the tokenizer.
  A model trained on it can be prompted with ingredient names to write a recipe (`--sample-ingredients` shows one at the end).
- `prepare_lm` prints the number of recipes, blocks and tokens. Packing follows RecipeNLG: whole recipes in blocks of 1024 tokens, longer ones skipped.
- Try `--max-steps 200` first to check speed and memory. If you run out of GPU memory, lower `--batch-size` and raise `--grad-accum`, or add `--grad-checkpointing`.
- The learning rate and schedule defaults are mine, not RecipeNLG's. Watch `val_ppl` and `holdout_ppl` in `models/gpt2-recipes/log.jsonl`;
  `holdout_ppl` is the honest number, because those recipes were kept out of training.
- `--model` can be any causal language model, including RecipeNLG's checkpoint after `textenc.to_safetensors`. Use `--resume` to continue a stopped run.
- `models/gpt2-recipes/best` and `last` are ordinary Hugging Face folders with `model.safetensors`. Give `best` to `textenc.encode --model` in step 4b
  to get instruction vectors from your fine-tuned model.

### 4d. Using a more modern model than GPT-2

Nothing in `textenc` is specific to GPT-2. `prepare_lm`, `finetune` and `encode` take any Hugging Face model through `--model`; the tests run the whole path with a
small random Llama-style model (the architecture of Llama, Qwen and SmolLM: rotary positions, grouped-query attention, tied embeddings, a tokenizer that adds a
start token and has no pad token). They have not been run on real downloaded checkpoints.

- **Generator you fine-tune** (ingredients in, recipe out): any causal language model. Models such as Qwen3 (0.6B to 8B), SmolLM3 (3B), Gemma 3, Llama 3.2 and Phi-4 mini are options;
  check each one's license. Run `prepare_lm --model <that model>` so the tokenizer matches; the 13 recipe tags are added as new tokens and the embedding matrix grows to fit them.
- **Memory:** `finetune.py` trains all weights with AdamW and fp32 master weights, which needs about 16 bytes per parameter before activations:
  roughly 10 GB for a 0.6B model, 27 GB for 1.7B. A 24 GB card therefore handles up to about 0.6B-parameter models; larger ones need LoRA or an 8-bit optimizer, which are not implemented yet.
- **Instruction vectors for the tri-joint model:** a dedicated embedding model (for example Qwen3-Embedding, EmbeddingGemma or bge-m3) is usually a better choice than a generator's hidden states.
  Use the pooling its model card gives (`--pooling last` for Qwen3-Embedding, `mean` or `first` for BERT-style models), `--prefix`/`--suffix` if it expects one,
  and `--stDim` equal to its output size (the encode command prints it).
  This reads the model with plain `transformers` plus pooling. Models packaged with sentence-transformers can add a projection or normalization layer on top, so their vectors may differ
  from the official ones; `--l2-normalize` covers normalization only. Models that need `trust_remote_code` are refused.

### 5. Build the dataset (needs 2, 3 and 4)

```bash
cd scripts
python build_dataset.py
cd ..
```

If some images that the json lists are not in your image folders (see step 5b), add `--image-root /full/path/to/extracted/images [more folders]`: the build then keeps only images that exist
(in their original order) and drops recipes left with none, so nothing trains on white placeholders. Use full paths, since the script runs from `scripts/`. The recipe counts shrink and no longer match the paper,
so numbers are not directly comparable. Run step 5b afterwards, so the resized images match the new stores.

Writes `data/{train,val,test}_store/`. Read the last two lines it prints. `filtered=` is normal (recipes without images, too many
instructions or ingredients, or in `remove1M.txt`). `no_vectors=` and `count_mismatch=` should be 0; if not, steps 1 and 4
were run on different data.

### 5b. Shrink the images (optional; saves most of the disk space)

Training only reads the first five images of each training recipe and the first image of each val/test recipe, resized to a shorter side
of 256 (then cropped to 224). `scripts/resize_images.py` keeps exactly those images, with the shorter side at 256 (never enlarged), as JPEG,
in the same four-level folder layout, so the result is a drop-in `--img_path`. It needs the stores from step 5.

```bash
# from the tar files, without extracting them (reads only the wanted files; one process per tar file)
python scripts/resize_images.py tars --tars /downloads/Recipe1M+_*.tar --data data --out data/images256

# or from images you have already extracted; --src takes several folders (for example one per tar file you extracted separately)
python scripts/resize_images.py extracted --src /path/to/recipe1M+_images --data data --out data/images256
python scripts/resize_images.py extracted --src /extract/a /extract/b /extract/0 --data data --out data/images256
```
Each wanted image is looked for as `<src>/0/f/a/8/0fa8309c13.jpg` in every `--src` folder. If your folders have some other layout (an extra wrapper folder, a flat folder), add `--scan`, which finds
the wanted file names anywhere below the `--src` folders; it walks every file once, so expect it to take a while on a large tree.
- It can be stopped and restarted: existing outputs are skipped (`--overwrite` redoes them). `<out>/missing.txt` and `failed.txt` list wanted images that were not found or could not be read; the loader substitutes a white image for those, so check the counts it prints.
- `extracted ... --in-place` replaces the originals of the wanted images instead of writing a new tree, to free disk space. Try it on a copy first, since the originals cannot be recovered. After checking the result you can delete the rest of the extracted tree.
- **Many images missing?** The script prints the missing share for each first character of the file name. A few characters near 100% mean whole tar files were not extracted (or sit in another folder: add them to `--src` or use `--scan`).
  The same share for every character means the images are not in the folders at all, and then the cause is the image list rather than the extraction: `layer2.json` is Recipe1M's list, and the Recipe1M+ image tars
  correspond to `layer2+.json` (the README says to replace one with the other for the extended dataset). Back up `layer2.json`, try `layer2+.json` under that name, and rerun `build_dataset.py`; it counts recipes without images as filtered.
- `--all-images` also keeps the non-first images of val/test recipes; `--size` and `--quality` change the target.
- Then train with `--img_path data/images256/`. Resizing in advance also removes most of the JPEG decoding cost, which makes training less dependent on the CPU count.

### 6. Train the tri-joint model

```bash
python train.py --img_path data/images/ --data_path data/ --ingrW2V data/text/vocab.bin --snapshots snapshots/
```
The ResNet-50 ImageNet weights download on first start (use `--no_pretrained` only for smoke tests). Defaults assume
`--stDim 1024 --ingrW2VDim 300 --numClasses 1048`; change them to match steps 2 to 4 if you changed those.
`--workers` defaults to 30 and `--batch_size` to 160, which is far more than a 24 GB card holds in fp32 (the ResNet-50 alone needs most of it).
If you get "CUDA out of memory", change these, in this order:

1. `--batch_size 64` (then 48 or 32 if needed). The learning rate was tuned for 160; I have not re-tuned it for smaller batches.
2. `--amp bf16`: bfloat16 mixed precision for the forward pass (weights and optimizer stay fp32), roughly halving activation memory. It needs an Ampere or newer GPU such as the 3090;
   the tests exercise it on CPU only, so watch the first epochs' loss on your GPU.
3. `--workers 8` (or your CPU count): worker processes use host RAM, not GPU memory.
4. If the traceback is in the DataLoader's `pin_memory` thread and mentions `cuMemHostAlloc`, the host could not lock memory for the batches; this is not a full GPU
   (training had not started). Add `--no_pin_memory` (copies to the GPU get slightly slower) and use fewer `--workers`. It is common under WSL, where the VM's memory limit can be raised in `.wslconfig`.

On Windows with WSL the desktop also takes some GPU memory; `nvidia-smi` shows what is free. Validation and testing no longer build gradient graphs (they used to), which lowers their peak memory as well.

Inside this step the schedule alternates between two phases:

1. **Phase 1, start:** ResNet-50 is frozen (learning rate 0); the recipe branch, image embedding and classifier train.
2. Every `--valfreq` epochs (default 10, skipping epoch 0) validation runs and reports median rank (MedR, lower is better) and recall.
3. If validation does not improve for `--patience` validations (default 1), the two groups **swap**: ResNet-50 trains, everything
   else is frozen. It swaps back the next time validation stalls, and so on.
4. A checkpoint `snapshots/model_eNNN_v-<val>.pth.tar` is written only when validation improves. Stop when it stops improving
   (the default is 720 epochs). Continue with `--resume <checkpoint>`.

The README says the original authors' default configuration converged in under 3 days on their hardware. I have no timing for this port.

### 7. Evaluate

```bash
python test.py --model_path snapshots/model_eNNN_v-X.XXX.pth.tar    # writes results/*.pkl
python scripts/rank.py --path_results results/                       # MedR and recall
```
Use the checkpoint with the best validation score. `rank.py` ranks random subsets of `--medr` samples (default 1000), so the
test partition must have at least that many recipes.

## What must agree across steps

| Setting | Where it is set | Where it must match |
|---|---|---|
| `--thought-size` (step 4) or the encoder's hidden size (step 4b) | skipinstructions training / `textenc.encode` output | `--stDim` in steps 6 and 7 |
| `--size` (step 2) | word2vec training | `--ingrW2VDim` in steps 6 and 7 |
| class count (step 3) | printed by `bigrams.py` | `--numClasses` in steps 6 and 7 |
| `--ingr_extra_rows` (default 2) | steps 6 and 7 | 2 if `vocab.txt` was written by `get_vocab.py`/`train_w2v.py`; 0 only for a `vocab.bin` that already starts with two reserved rows |
| `--maxlen` (default 20 in step 5) | `build_dataset.py` | the loader's 20-instruction limit in `data_loader.py` |

Steps 6 and 7 must be run with the same values, because the checkpoint's shapes depend on them.

## Saved models

| Model | Written by | Weights file (safe to share) | Also written (resume) |
|---|---|---|---|
| Skip-instruction encoder | `skipinstructions.train` | `runs/skip1/skipinstructions-best.safetensors` (config in its metadata) | `skipinstructions-best.pt`, `skipinstructions-last.pt` |
| Tri-joint model | `train.py` | `snapshots/model_eNNN_v-X.safetensors` (epoch, best val and training options in its metadata) | `.pth.tar` with the optimizer state |
| Hugging Face model | `save_pretrained`, `textenc.to_safetensors` | `model.safetensors` + `config.json` + tokenizer files | none needed |

`.safetensors` files hold tensors only, so loading them cannot run code. They are accepted by `test.py --model_path`, `image2embedding.py`,
`skipinstructions.encode --checkpoint` and `train.py --resume`. Resuming from a `.safetensors` file restarts the optimizer (Adam's running averages are not stored)
but restores the epoch, best validation value and which phase of the schedule training was in; resume from the `.pth.tar` to continue exactly.
`train.py --checkpoint_format safetensors` writes only the safetensors file. Convert a `.pt` skipinstructions checkpoint from before this existed with `python -m skipinstructions.export <file>.pt`.

## Using this data with MeaLeon

These are design notes: I have not seen MeaLeon's code, only its architecture (Flask/FastAPI services, PostgreSQL, Vespa, MLflow).

- **Join key.** `recipe_links.csv` maps the normalized recipe URL to the Recipe1M id, partition and RecipeNLG row. If MeaLeon's recipes keep a source URL,
  normalize it with `merge_recipenlg.norm_link` and join on it. Without URLs you would have to match on title and ingredients.
- **What each part could give MeaLeon.** RecipeNLG's `NER` ingredient names for the sparse `ingredients_bow` side of hybrid search; dense vectors from the
  tri-joint model (image and recipe share one space, so a photo can query recipes) or from the `textenc` encoder for the vector side; and the fine-tuned model
  for "write a recipe from what I have".
- **Serving.** `textenc.models.load_encoder` / `encode_texts` take a list of strings and return a batch of vectors, so they can sit behind the embedding service,
  or be wrapped as an MLflow pyfunc with the safetensors folder as its artifact, as with the TF-IDF model.
- **Licensing.** RecipeNLG's terms allow non-commercial research and educational use. Recipe1M has its own terms. Keep data and models derived from them
  out of any public deployment until you have checked both.
