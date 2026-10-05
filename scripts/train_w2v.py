#!/usr/bin/env python
"""Train the ingredient / word embeddings with gensim (replaces the C word2vec tool).

    python train_w2v.py --corpus ../data/skipinstr/tokenized_instructions_train.txt --out ../data/text/vocab.bin

The defaults are the flags the README used with the C tool:
    -hs 1 -negative 0 -window 10 -cbow 0 -iter 10 -size 300 -min-count 10
Writes <out> (word2vec binary) and vocab.txt next to it. Row order of the .bin is gensim's
(most frequent word first); vocab.txt lists the words in that same order, which is what the
model and build_dataset.py expect (see scripts/check_ingredient_vocab.py).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from word2vec_io import write_vocab_txt  # noqa: E402


def train(corpus, out, size=300, window=10, epochs=10, min_count=10, workers=None, seed=1):
    from gensim.models import Word2Vec
    from gensim.models.word2vec import LineSentence

    model = Word2Vec(LineSentence(corpus), vector_size=size, window=window, sg=1, hs=1, negative=0,
                     min_count=min_count, epochs=epochs, workers=workers or os.cpu_count() or 1, seed=seed)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    model.wv.save_word2vec_format(out, binary=True)
    return len(model.wv), write_vocab_txt(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default="../data/skipinstr/tokenized_instructions_train.txt",
                   help="one recipe per line, tokens separated by whitespace (tokenize_instructions --w2v-corpus)")
    p.add_argument("--out", default="../data/text/vocab.bin")
    p.add_argument("--size", type=int, default=300, help="must equal --ingrW2VDim of the trijoint model")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--min-count", type=int, default=10)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args(argv)
    n, vocab_txt = train(a.corpus, a.out, a.size, a.window, a.epochs, a.min_count, a.threads, a.seed)
    print("%d words -> %s and %s" % (n, a.out, vocab_txt))


if __name__ == "__main__":
    main()
