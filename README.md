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

## Build from Scratch

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

#### 5. Build the dataset (needs 2, 3 and 4)

```bash
cd scripts
python build_dataset.py
cd ..
```

Writes `data/{train,val,test}_store/`. Read the last two lines it prints. `filtered=` is normal (recipes without images, too many
instructions or ingredients, or in `remove1M.txt`). `no_vectors=` and `count_mismatch=` should be 0; if not, steps 1 and 4
were run on different data.

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
| `--thought-size` (step 4) | skipinstructions training | `--stDim` in steps 6 and 7 |
| `--size` (step 2) | word2vec training | `--ingrW2VDim` in steps 6 and 7 |
| class count (step 3) | printed by `bigrams.py` | `--numClasses` in steps 6 and 7 |
| `--ingr_extra_rows` (default 2) | steps 6 and 7 | 2 if `vocab.txt` was written by `get_vocab.py`/`train_w2v.py`; 0 only for a `vocab.bin` that already starts with two reserved rows |
| `--maxlen` (default 20 in step 5) | `build_dataset.py` | the loader's 20-instruction limit in `data_loader.py` |

Steps 6 and 7 must be run with the same values, because the checkpoint's shapes depend on them.

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
