"""scripts/train_w2v.py and scripts/get_vocab.py"""
import os
import subprocess
import sys

import numpy as np

from word2vec_io import load_word2vec_bin, write_vocab_txt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_corpus(path):
    lines = []
    for i in range(60):
        lines.append("heat the olive_oil . add garlic and onion , stir .\tcook with salt and olive_oil .")
        lines.append("chop the onion .\tseason with salt , then serve warm .")
    open(path, "w").write("\n".join(lines) + "\n")


def test_train_w2v_writes_bin_and_matching_vocab(tmp_path):
    import train_w2v
    corpus, out = tmp_path / "corpus.txt", tmp_path / "text" / "vocab.bin"
    write_corpus(str(corpus))
    n, vocab_txt = train_w2v.train(str(corpus), str(out), size=8, window=3, epochs=2, min_count=1, workers=1, seed=1)
    words, vecs = load_word2vec_bin(str(out))
    assert len(words) == n == len(set(words)) and vecs.shape == (n, 8) and np.isfinite(vecs).all()
    assert {"olive_oil", "garlic", "salt", "onion", "."} <= set(words)  # underscored ingredients stay one token
    assert open(vocab_txt).read().split("\n")[:-1] == words  # vocab.txt is the file order, which the model relies on
    assert os.path.dirname(vocab_txt) == str(out.parent)


def test_train_w2v_applies_min_count(tmp_path):
    import train_w2v
    corpus = tmp_path / "corpus.txt"
    write_corpus(str(corpus))
    open(corpus, "a").write("rare_word once\n")
    train_w2v.train(str(corpus), str(tmp_path / "a.bin"), size=8, window=3, epochs=1, min_count=2, workers=1)
    words, _ = load_word2vec_bin(str(tmp_path / "a.bin"))
    assert "rare_word" not in words and "salt" in words


def test_get_vocab_script(tmp_path):
    import train_w2v
    corpus = tmp_path / "corpus.txt"
    write_corpus(str(corpus))
    train_w2v.train(str(corpus), str(tmp_path / "v.bin"), size=8, window=3, epochs=1, min_count=1, workers=1)
    (tmp_path / "vocab.txt").unlink()
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "get_vocab.py"), str(tmp_path / "v.bin")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "vocab.txt").read_text().split("\n")[:-1] == load_word2vec_bin(str(tmp_path / "v.bin"))[0]
    assert write_vocab_txt(str(tmp_path / "v.bin"), str(tmp_path / "other.txt")) == str(tmp_path / "other.txt")
