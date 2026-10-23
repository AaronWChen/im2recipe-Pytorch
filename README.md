# im2recipe: Learning Cross-modal Embeddings for Cooking Recipes and Food Images

This repository contains the code to train and evaluate models from the paper:  
_Learning Cross-modal Embeddings for Cooking Recipes and Food Images_

Important note: In this repository the Skip-instructions has not been reimplemented in Pytorch, instead needed features are provided to train, validate and test the tri_joint model.

Clone it using:

```shell
git clone --recursive https://github.com/torralba-lab/im2recipe-Pytorch.git
```

If you find this code useful, please consider citing:

```plaintext
@article{marin2019learning,
  title = {Recipe1M+: A Dataset for Learning Cross-Modal Embeddings for Cooking Recipes and Food Images},
  author = {Marin, Javier and Biswas, Aritro and Ofli, Ferda and Hynes, Nicholas and 
  Salvador, Amaia and Aytar, Yusuf and Weber, Ingmar and Torralba, Antonio},
  journal = {{IEEE} Trans. Pattern Anal. Mach. Intell.},
  year = {2019}
}

@inproceedings{salvador2017learning,
  title={Learning Cross-modal Embeddings for Cooking Recipes and Food Images},
  author={Salvador, Amaia and Hynes, Nicholas and Aytar, Yusuf and Marin, Javier and 
          Ofli, Ferda and Weber, Ingmar and Torralba, Antonio},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}
```

## Contents

1. [Installation](#installation)
2. [Recipe1M Dataset](#recipe1m-and-recipe1m-datasets)
3. [Vision models](#vision-models)
4. [Build from Scratch](#build-from-scratch)
5. [Out-of-the-box training](#out-of-the-box-training)
6. [Prepare training data](#prepare-training-data)
7. [Training](#training)
8. [Testing](#testing)
9. [Pretrained model](#pretrained-model)
10. [Recipes with nutritional info](#recipes-with-nutritional-info)
11. [Contact](#contact)

## Installation

```bash
docker build -t im2recipe .
docker run -it im2recipe
```

You may use a volume to give snapshots, data, etc to the docker container.

If you are not using Docker, we do recommend to create a new environment with Python 3.7. Right after it, run ```pip install --upgrade cython``` and then install the dependencies with ```pip install -r requirements.txt```. Notice that this will install the latest PyTorch version available. Once you finish, you will need to install [torchwordemb](https://github.com/iamalbert/pytorch-wordemb). In order to do that (or at least the way we found it worked for us), we downloaded and installed it via ```python setup.py install```. In case you get an error related to  ```return {vocab, dest};```, you just need to change the original code to ```return VocabAndTensor(vocab, dest);```, and run ```python setup.py install``` again.

## Recipe1M and Recipe1M+ Datasets

In order to get access to the dataset, please fill the following form [here](https://forms.gle/EzYSu8j3D1LJzVbR8).

## Vision models

This current version of the code uses a pre-trained ResNet-50.

## Building Tri-joint Model from Scratch

Every command below is run from the repository root unless it says `cd scripts`. The whole sequence,
with these exact command lines, runs end to end on a tiny synthetic dataset:
`bash tests/e2e_dry_run.sh` (needs network once for the nltk data used by `bigrams.py`).
It has not been run on real Recipe1M data.

### The parts, and what each one needs

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

### Order of operations

```plaintext
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

#### 0. Set up and get the data

```bash
pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ('punkt', 'punkt_tab', 'stopwords')]"   # for bigrams.py
python -m pytest tests skipinstructions/tests -q      # checks the install; takes about a minute
```

Get Recipe1M through the form linked in `README.md` and arrange it like this:

`food101_classes_renamed.txt` can be found in the original `im2recipe` repo, not on the form listedin `README.md`

```plaintext
data/recipe1M/layer1.json  layer2.json  det_ingrs.json
data/images/               (four-level folders: 0/f/a/8/0fa8....jpg)
data/food101_classes_renamed.txt
```

#### 1. Tokenize

```bash
python -m skipinstructions.tokenize_instructions --dataset data/recipe1M --out-dir data/skipinstructions --w2v-corpus
```

Writes `instructions_<part>.txt` (+ `.index.tsv`) for skip-instructions and `tokenized_instructions_<part>.txt` for word2vec.
Look at the printed instruction-length summary: instructions longer than `--maxlen` (default 30 steps) are truncated in step 4.

#### 2. Ingredient word2vec

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

#### 3. Semantic classes (needs `vocab.txt` from step 2)

```bash
cd scripts
python bigrams.py --crtbgrs        # bigrams of training recipe titles -> ../data/bigrams1M.pkl
python bigrams.py --nocrtbgrs      # class labels from those bigrams + Food-101 -> ../data/classes1M.pkl
cd ..
```

The second command prints the number of classes (background included) on its last line.
If it is not 1048, pass that number as `--numClasses` in step 6. The second command loops over every recipe for each
candidate bigram in plain Python, so expect it to be slow. I have not timed it on the full dataset.

#### 4. Skip-instruction encoder (independent of steps 2 and 3)

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

#### 4b. Alternative to step 4: encode instructions with a Hugging Face model (GPT-2, RecipeNLG, your own fine-tune)

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
python train.py --data_path data_gpt2/ --stDim 768 --img_path data/images/ --ingrW2V data/text/vocab.bin --snapshots snapshots_gpt2
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

#### 4c. Merge RecipeNLG and fine-tune a language model on the combined text (optional)

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

#### 4d. Using a more modern model than GPT-2

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

#### 5. Build the dataset (needs 2, 3 and 4)

```bash
cd scripts
python build_dataset.py
cd ..
```

Writes `data/{train,val,test}_store/`. Read the last two lines it prints. `filtered=` is normal (recipes without images, too many
instructions or ingredients, or in `remove1M.txt`). `no_vectors=` and `count_mismatch=` should be 0; if not, steps 1 and 4
were run on different data.

#### 5b. Shrink the images (optional; saves most of the disk space)

Training only reads the first five images of each training recipe and the first image of each val/test recipe, resized to a shorter side
of 256 (then cropped to 224). `scripts/resize_images.py` keeps exactly those images, with the shorter side at 256 (never enlarged), as JPEG,
in the same four-level folder layout, so the result is a drop-in `--img_path`. It needs the stores from step 5.

```bash
# from the tar files, without extracting them (reads only the wanted files; one process per tar file)
python scripts/resize_images.py tars --tars /downloads/Recipe1M+_*.tar --data data --out data/images256

# or from images you have already extracted
python scripts/resize_images.py extracted --src /path/to/recipe1M+_images --data data --out data/images256
```
- It can be stopped and restarted: existing outputs are skipped (`--overwrite` redoes them). `<out>/missing.txt` and `failed.txt` list wanted images that were not found or could not be read; the loader substitutes a white image for those, so check the counts it prints.
- `extracted ... --in-place` replaces the originals of the wanted images instead of writing a new tree, to free disk space. Try it on a copy first, since the originals cannot be recovered. After checking the result you can delete the rest of the extracted tree.
- `--all-images` also keeps the non-first images of val/test recipes; `--size` and `--quality` change the target.
- Then train with `--img_path data/images256/`. Resizing in advance also removes most of the JPEG decoding cost, which makes training less dependent on the CPU count.

#### 6. Train the tri-joint model

```bash
python train.py --img_path data/images/ --data_path data/ --ingrW2V data/text/vocab.bin --snapshots snapshots/
```

The ResNet-50 ImageNet weights download on first start (use `--no_pretrained` only for smoke tests). Defaults assume
`--stDim 1024 --ingrW2VDim 300 --numClasses 1048`; change them to match steps 2 to 4 if you changed those.
`--workers` defaults to 30 and `--batch_size` to 160; lower them to fit your machine.

Inside this step the schedule alternates between two phases:

1. **Phase 1, start:** ResNet-50 is frozen (learning rate 0); the recipe branch, image embedding and classifier train.
2. Every `--valfreq` epochs (default 10, skipping epoch 0) validation runs and reports median rank (MedR, lower is better) and recall.
3. If validation does not improve for `--patience` validations (default 1), the two groups **swap**: ResNet-50 trains, everything
   else is frozen. It swaps back the next time validation stalls, and so on.
4. A checkpoint `snapshots/model_eNNN_v-<val>.pth.tar` is written only when validation improves. Stop when it stops improving
   (the default is 720 epochs). Continue with `--resume <checkpoint>`.

The README says the original authors' default configuration converged in under 3 days on their hardware. I have no timing for this port.

#### 7. Evaluate

```bash
python test.py --model_path snapshots/model_eNNN_v-X.XXX.pth.tar    # writes results/*.pkl
python scripts/rank.py --path_results results/                       # MedR and recall
```

Use the checkpoint with the best validation score. `rank.py` ranks random subsets of `--medr` samples (default 1000), so the
test partition must have at least that many recipes.

### What must agree across steps

| Setting | Where it is set | Where it must match |
| --- | --- | --- |
| `--thought-size` (step 4) or the encoder's hidden size (step 4b) | skipinstr training / `textenc.encode` output | `--stDim` in steps 6 and 7 |
| `--size` (step 2) | word2vec training | `--ingrW2VDim` in steps 6 and 7 |
| class count (step 3) | printed by `bigrams.py` | `--numClasses` in steps 6 and 7 |
| `--ingr_extra_rows` (default 2) | steps 6 and 7 | 2 if `vocab.txt` was written by `get_vocab.py`/`train_w2v.py`; 0 only for a `vocab.bin` that already starts with two reserved rows |
| `--maxlen` (default 20 in step 5) | `build_dataset.py` | the loader's 20-instruction limit in `data_loader.py` |

Steps 6 and 7 must be run with the same values, because the checkpoint's shapes depend on them.

### Saved models

| Model | Written by | Weights file (safe to share) | Also written (resume) |
|---|---|---|---|
| Skip-instruction encoder | `skipinstr.train` | `runs/skip1/skipinstr-best.safetensors` (config in its metadata) | `skipinstr-best.pt`, `skipinstr-last.pt` |
| Tri-joint model | `train.py` | `snapshots/model_eNNN_v-X.safetensors` (epoch, best val and training options in its metadata) | `.pth.tar` with the optimizer state |
| Hugging Face model | `save_pretrained`, `textenc.to_safetensors` | `model.safetensors` + `config.json` + tokenizer files | none needed |

`.safetensors` files hold tensors only, so loading them cannot run code. They are accepted by `test.py --model_path`, `image2embedding.py`,
`skipinstructions.encode --checkpoint` and `train.py --resume`. Resuming from a `.safetensors` file restarts the optimizer (Adam's running averages are not stored)
but restores the epoch, best validation value and which phase of the schedule training was in; resume from the `.pth.tar` to continue exactly.
`train.py --checkpoint_format safetensors` writes only the safetensors file. Convert a `.pt` skipinstr checkpoint from before this existed with `python -m skipinstructions.export <file>.pt`.

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

## Out-of-the-box training

To train the model, you will need to create following files:
* `data/{train,val,test}_store/`: one folder of flat `.npy` arrays per partition (skip-instructions vectors with recipe offsets, ingredient ids, categories, image names), written by `scripts/build_dataset.py` (see below).
* `data/text/vocab.txt`: file containing all the vocabulary found within the recipes.

And download the following ones: 
* `data/text/vocab.bin`: ingredient Word2Vec vocabulary. Used during training to select word2vec vectors given ingredient ids.
* `data/food101_classes_renamed.txt`: Food101 classes used to create the bigrams.
* `data/recipe1M/layer2+.json`: Recipe1M+ layer2.
* `data/images/Recipe1M+_{a..f}.tar`: 6 Tar files containing part of the images available in Recipe1M+ (~210Gb each).
* `data/images/Recipe1M+_{0..9}.tar`: 10 Tar files containing part of the images available in Recipe1M+ (~210Gb each).

To download these files, you first need to complete the following [form](https://forms.gle/EzYSu8j3D1LJzVbR8). After submission, we will share the necessary download links via email. Access to the dataset is granted only for research purposes to universities and research institutions. Original Recipe1M LMDBs and pickle files can be found in train.tar, val.tar and test.tar.

It is worth mentioning that the code is expecting images to be located in a four-level folder structure, e.g. image named `0fa8309c13.jpg` can be found in `./data/images/0/f/a/8/0fa8309c13.jpg`. Each one of the Tar files contains the first folder level, 16 in total. If you do not have enough space after downloading the Tar files, you can try to mount them locally and access them. We did use [ratarmount](https://github.com/mxmlnkn/ratarmount) in our latest test experiments. In order to properly access the images with ratarmount, we temporarily changed our code. We basically tried up to three times to load an image within our `default_loader`.

## Prepare training data

We also provide the steps to format and prepare Recipe1M/Recipe1M+ data for training the trijoint model. We hope these instructions will allow others to train similar models with other data sources as well.

### Word2Vec

Training word2vec with recipe data:

* Run

```bash
python -m skipinstructions.tokenize_instructions --dataset data/recipe1M --out-dir data/skipinstr --w2v-corpus
```

from the repository root. This writes `tokenized_instructions_<partition>.txt` (one recipe per line, used here for word2vec) and the one-instruction-per-line files used for skip-instructions below.
<!-- - Run the same ```python tokenize_instructions.py``` to generate the same file with data for all partitions (needed for skip-thoughts later). -->

See `skipinstructions/README.md`. From the repository root, in bash/terminal:

```bash
python -m skipinstructions.train --train-file data/skipinstructions/instructions_train.txt --train-index data/skipinstructions/instructions_train.index.tsv \
    --val-file data/skipinstructions/instructions_val.txt --val-index data/skipinstructions/instructions_val.index.tsv --out-dir runs/skip1
for p in train val test; do
  python -m skipinstructions.encode --checkpoint runs/skip1/skipinstr-best.pt \
      --sentences data/skipinstructions/instructions_$p.txt --index data/skipinstructions/instructions_$p.index.tsv \
      --out-prefix data/skipinstructions/$p
done
```

This writes `data/skipinstructions/<partition>.encs.npy` (+ `.index.json`). Add `--dtype float16` to halve their size.

* Original arguments for the original Word2Vec model were:

```bash
./word2vec -hs 1 -negative 0 -window 10 -cbow 0 -iter 10 -size 300 -binary 1 -min-count 10 -threads 20 -train tokenized_instructions_train.txt -output vocab.bin
```

### Choosing semantic categories

We provide the script we used to extract semantic categories from bigrams in recipe titles:

* Run ```python bigrams --crtbgrs```. This will save to disk all bigrams in the corpus of all recipe titles in the training set, sorted by frequency. Note that you will need to create first ```vocab.txt``` running ```python get_vocab.py ../data/vocab.bin``` within ```./scripts/```.
* Running the same script again with ```--nocrtbgrs``` will create class labels from those bigrams adding food101 categories.

These steps will create a file called ```classes1M.pkl``` in ```./data/``` that will be used later to create the LMDB file including categories.

### Skip-instructions (PyTorch)

Skipthoughts-pytorch implementation has been added as a subtree and will be used to retrain the skipthoughts model

* Prepare the dataset by running from the `scripts` directory:
`python skip-instructions_mk_dataset.py --dataset /path/to/recipe1M/ --vocab /path/to/w2v/word2vec_vocab.txt --toks /path/to/tokenized_instructions.txt`

The `skip-instructions_mk_dataset.py` file has default values already.

`tokenized_instructions.txt` contains text instructions for the entire dataset. Different instructions for the same recipe are separated by '\t' and different recipe instructions are delimited with '\n'. 

`word2vec_vocab.txt` is a file containing the entries of the previously trained word2vec model.

### Creating the dataset

Run the following from ```./scripts``` (it replaces `mk_dataset.py`):

```bash
python build_dataset.py \
  --vocab /path/to/w2v/vocab.txt \
  --skip-dir ../data/skipinstructions \
  --out-dir ../data
```

This writes `data/{train,val,test}_store/`. Recipes without images, with too many ingredients/instructions, or listed in `remove1M.txt` are skipped. Notice, that layer2 within ```./data/recipe1M/layer2.json``` will need to be replaced by layer2+.json in order to create our extended Recipe1M+ dataset.

## Training

* Train the model with:

```bash
python train.py \
  --img_path /path/to/images/ \
  --data_path /path/to/lmdbs/ \
  --ingrW2V /path/to/w2v/vocab.bin \
  --snapshots snapshots/ \
  --valfreq 10
```

* Note: Again, this can be run without arguments with default parameters if files are in the default location.*

* You can set ```-batchSize``` to ~160. This is the default config, which will make the model converge in less than 3 days. Pytorch version requires less memory. You should be able to train the model using two TITAN X 12gb with same batch size. In this version we are using LMDBs to load the instructions and ingredients instead of a single HDF5 file.

## Testing

* Extract features from test set

```bash
python test.py --model_path=snapshots/model*.tar
```

* They will be saved in `results`.
* After feature extraction, compute MedR and recall scores with ```python scripts/rank.py --path_results=results```.

## Pretrained model

Our best model trained with Recipe1M+ (journal extension) can be downloaded [here](http://data.csail.mit.edu/im2recipe/model_e500_v-8.950.pth.tar).

You can test it with:

```bash
python test.py --model_path=snapshots/model_e500_v-8.950.pth.tar
```

Our best model trained with Recipe1M (CVPR paper) can be downloaded [here](http://data.csail.mit.edu/im2recipe/model_e220_v-4.700.pth.tar).

## Recipes with nutritional info

We also provide a subset of recipes with nutritional information. Below you can see an example:

```plaintext
{'fsa_lights_per100g': {'fat': 'green',
  'salt': 'green',
  'saturates': 'green',
  'sugars': 'orange'},
 'id': '000095fc1d',
 'ingredients': [{'text': 'yogurt, greek, plain, nonfat'},
  {'text': 'strawberries, raw'},
  {'text': 'cereals ready-to-eat, granola, homemade'}],
 'instructions': [{'text': 'Layer all ingredients in a serving dish.'}],
 'nutr_per_ingredient': [{'fat': 0.8845044000000001,
   'nrg': 133.80964,
   'pro': 23.110512399999998,
   'sat': 0.26535132,
   'sod': 81.64656,
   'sug': 7.348190400000001},
  {'fat': 0.46,
   'nrg': 49.0,
   'pro': 1.02,
   'sat': 0.023,
   'sod': 2.0,
   'sug': 7.43},
  {'fat': 7.415,
   'nrg': 149.25,
   'pro': 4.17,
   'sat': 1.207,
   'sod': 8.0,
   'sug': 6.04}],
 'nutr_values_per100g': {'energy': 81.12946131894766,
  'fat': 2.140139263515891,
  'protein': 6.914436593565536,
  'salt': 0.05597816738985967,
  'saturates': 0.36534716195613937,
  'sugars': 5.08634103436144},
 'partition': 'train',
 'quantity': [{'text': '8'}, {'text': '1'}, {'text': '1/4'}],
 'title': 'Yogurt Parfaits',
 'unit': [{'text': 'ounce'}, {'text': 'cup'}, {'text': 'cup'}],
 'url': 'http://tastykitchen.com/recipes/breakfastbrunch/yogurt-parfaits/',
 'weight_per_ingr': [226.796, 152.0, 30.5]}
```

Note that these recipes include the matched ingredients from USDA instead of the original ones. There are 35,867 recipes for training, 7,687 for validation and 7,681 for testing. In order to obtain the grams of salt, we multiplied the sodium by 2.5 and divided it by 1000. Total weight per ingredient, fat, proteins/pro, salt, saturates/sat and sugars/sug are expressed in grams. Sodium/sod is expressed in mg and energy/nrg in kcal. FSA traffic lights are also included per 100g.

## Contact

For any questions or suggestions you can use the issues section or reach us at jmarin@csail.mit.edu.
