#!/usr/bin/env bash
# Runs every stage of BUILD.md, in order, on a tiny synthetic Recipe1M, with the real command lines.
# Needs network once, for the nltk data used by scripts/bigrams.py. Usage: bash tests/e2e_dry_run.sh [WORKDIR]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
W="${1:-$(mktemp -d)}"
rm -rf "$W/repo" && mkdir -p "$W/repo" && cp -r "$REPO"/. "$W/repo" && cd "$W/repo"
PY="${PYTHON:-python3}"
mkdir -p data/recipe1M data/text
$PY tests/synthetic.py data/recipe1M
printf 'apple pie\n' > data/food101_classes_renamed.txt
$PY -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt', 'punkt_tab', 'stopwords')]"

echo "== 1 tokenize";            $PY -m skipinstructions.tokenize_instructions --dataset data/recipe1M --out-dir data/skipinstructions --w2v-corpus
echo "== 2a ingredient w2v";     (cd scripts && $PY train_w2v.py --size 6 --min-count 1 --epochs 2 --threads 1)
echo "== 2b check alignment";    (cd scripts && $PY check_ingredient_vocab.py)
echo "== 3 classes";             (cd scripts && $PY bigrams.py --crtbgrs && $PY bigrams.py --nocrtbgrs)
echo "== 4a skip-instr train";   $PY -m skipinstructions.train --train-file data/skipinstructions/instructions_train.txt --train-index data/skipinstructions/instructions_train.index.tsv \
        --val-file data/skipinstructions/instructions_val.txt --val-index data/skipinstructions/instructions_val.index.tsv --out-dir runs/skip1 \
        --maxlen 10 --word-size 16 --thought-size 24 --batch-size 16 --iters 40 --eval-every 20 --save-every 20 --log-every 20 --device cpu
echo "== 4b skip-instr encode"
for p in train val test; do
  $PY -m skipinstructions.encode --checkpoint runs/skip1/skipinstructions-best.pt --sentences data/skipinstructions/instructions_$p.txt \
      --index data/skipinstructions/instructions_$p.index.tsv --out-prefix data/skipinstructions/$p --device cpu
done
echo "== 5 build dataset";       (cd scripts && $PY build_dataset.py --remove remove1M.txt)
echo "== 6 images + trijoint";   $PY -c "
import sys; sys.path.insert(0, 'tests'); sys.path.insert(0, '.')
from synthetic import make_images
from recipe_store import RecipeStore
make_images('data/images', [RecipeStore('data/%s_store' % p) for p in ('train', 'val', 'test')])"
COMMON="--ingrW2VDim 6 --stDim 24 --srnnDim 16 --irnnDim 8 --embDim 12 --workers 0 --batch_size 8 --no_pretrained"
mkdir -p snapshots results
$PY train.py $COMMON --epochs 2 --valfreq 1 --medr 10 --snapshots snapshots/
echo "== 7 test + rank";         CK=$(ls snapshots/*.pth.tar | tail -1)
$PY test.py $COMMON --model_path "$CK" --path_results results/
$PY scripts/rank.py --path_results results/ --medr 5
echo "DRY RUN OK ($W/repo)"
