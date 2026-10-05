"""Run from the folder that contains the skipinstr package:  python -m pytest skipinstr/tests -q"""
import json
import os
import random

import numpy as np
import torch

from skipinstr import encode, train
from skipinstr.data import SentenceCorpus, read_index, write_index
from skipinstr.model import UniSkip
from skipinstr.tokenize_instructions import process, tok, tokenize_recipe
from skipinstr.vocab import EOS, UNK, build_vocab, load_vocab, save_vocab


# ---------------------------------------------------------------- tokenizer
def test_tok_splits_punctuation():
    assert tok("Mix, then bake (20 min).").split() == ["Mix", ",", "then", "bake", "(", "20", "min", ")", "."]


def test_tokenize_recipe_joins_ingredients_and_keeps_line_count():
    out = tokenize_recipe(["Add the olive oil, stir.", "", "Line\nbreak\tand   spaces"], ["olive oil", "salt"])
    assert len(out) == 3
    assert "olive_oil" in out[0].split() and out[0].endswith(".")
    assert out[1] == ""  # empty instruction is kept, not dropped
    assert out[2] == "Line break and spaces"
    assert tokenize_recipe([], ["x y"]) == []


def _write_recipe_json(tmp_path):
    layer1 = [
        {"id": "r1", "partition": "train", "instructions": [{"text": "Heat the olive oil."}, {"text": "Add garlic, stir."}]},
        {"id": "r2", "partition": "val", "instructions": [{"text": "Bake for 20 minutes."}]},
        {"id": "r3", "partition": "train", "instructions": []},
        {"id": "r4", "partition": "train", "instructions": [{"text": "Serve."}, {"text": "  "}, {"text": "Enjoy!"}]},
    ]
    det = [
        {"id": "r1", "ingredients": [{"text": "olive oil"}, {"text": "garlic"}], "valid": [True, True]},
        {"id": "r2", "ingredients": [], "valid": []},
        {"id": "r3", "ingredients": [], "valid": []},
    ]  # r4 has no det entry on purpose
    (tmp_path / "layer1.json").write_text(json.dumps(layer1))
    (tmp_path / "det_ingrs.json").write_text(json.dumps(det))


def test_process_writes_aligned_files(tmp_path):
    _write_recipe_json(tmp_path)
    out = tmp_path / "out"
    process(str(tmp_path / "layer1.json"), str(tmp_path / "det_ingrs.json"), str(out), w2v_corpus=True)
    train_lines = (out / "instructions_train.txt").read_text().split("\n")[:-1]
    ids, rlens = read_index(str(out / "instructions_train.index.tsv"))
    assert ids == ["r1", "r3", "r4"] and rlens.tolist() == [2, 0, 3]
    assert len(train_lines) == rlens.sum() == 5
    assert train_lines[0] == "Heat the olive_oil ."
    assert train_lines[3] == ""  # whitespace-only instruction becomes an empty line
    assert (out / "tokenized_instructions_train.txt").read_text().split("\n")[0].count("\t") == 1
    assert len((out / "instructions_test.txt").read_text()) == 0


# -------------------------------------------------------------------- vocab
def test_vocab_reserved_ids_and_roundtrip(tmp_path):
    f = tmp_path / "s.txt"
    f.write_text("a b a\nb c a\n")
    itos = build_vocab(str(f), vocab_size=4)
    assert itos == ["<eos>", "<unk>", "a", "b"]  # capped at 4 entries, most frequent first
    save_vocab(itos, str(tmp_path / "v.txt"))
    assert load_vocab(str(tmp_path / "v.txt")) == itos


# ---------------------------------------------------------- toy corpus setup
def make_corpus(tmp_path, n_recipes=60, seed=0):
    """Recipes whose instructions follow a fixed order, so the model has something to learn."""
    rnd = random.Random(seed)
    steps = ["chop the onion .", "heat the pan .", "add oil and stir .", "cook for ten minutes .", "season with salt .", "serve warm ."]
    lines, ids, rlens = [], [], []
    for i in range(n_recipes):
        k = rnd.randint(3, len(steps))
        lines += steps[:k]
        ids.append(f"rec{i}")
        rlens.append(k)
    sent, index = tmp_path / "sent.txt", tmp_path / "sent.index.tsv"
    sent.write_text("\n".join(lines) + "\n")
    write_index(str(index), ids, rlens)
    return str(sent), str(index)


def test_corpus_lengths_and_pair_mask(tmp_path):
    sent, index = make_corpus(tmp_path)
    itos = build_vocab(sent)
    corpus = SentenceCorpus(sent, itos, maxlen=8, index_path=index)
    tokens, lengths, valid = corpus.batch(0, 20)
    assert tokens.shape == (20, 8) and int(lengths.min()) >= 1
    assert lengths[0] == 4 + 1  # "chop the onion ." + <eos>
    assert tokens[0, 4] == EOS
    starts = np.concatenate([[0], np.cumsum(corpus.rlens)])
    n_boundaries = sum(1 for s in starts[1:-1] if s < 20)
    assert int((~valid).sum()) == n_boundaries
    # truncation: maxlen 3 keeps 2 words + <eos>
    short = SentenceCorpus(sent, itos, maxlen=3)
    assert short.lengths[0] == 3 and short.tokens[0, 2] == EOS


def test_unknown_words_map_to_unk(tmp_path):
    f = tmp_path / "s.txt"
    f.write_text("a b\nzzz a\n")
    corpus = SentenceCorpus(str(f), ["<eos>", "<unk>", "a"], maxlen=4)
    assert corpus.tokens[0].tolist()[:3] == [2, UNK, EOS]
    assert corpus.tokens[1].tolist()[:3] == [UNK, 2, EOS]


# ------------------------------------------------------------------- model
def test_loss_is_finite_and_mask_matters(tmp_path):
    sent, index = make_corpus(tmp_path)
    itos = build_vocab(sent)
    corpus = SentenceCorpus(sent, itos, maxlen=8, index_path=index)
    torch.manual_seed(0)
    model = UniSkip(len(itos), 16, 24)
    tokens, lengths, valid = corpus.batch(0, 20)
    masked = model(tokens, lengths, valid)
    unmasked = model(tokens, lengths)
    assert torch.isfinite(masked) and torch.isfinite(unmasked)
    assert not torch.isclose(masked, unmasked)
    masked.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_encoding_is_independent_of_batching(tmp_path):
    sent, index = make_corpus(tmp_path)
    itos = build_vocab(sent)
    corpus = SentenceCorpus(sent, itos, maxlen=8, index_path=index)
    torch.manual_seed(0)
    model = UniSkip(len(itos), 16, 24).eval()
    full = model.encode(*corpus.batch(0, 40)[:2])
    for i in (0, 7, 39):
        single = model.encode(*corpus.batch(i, 1)[:2])
        assert torch.allclose(full[i:i + 1], single, atol=1e-5)
    part = model.encode(*corpus.batch(5, 9)[:2])
    assert torch.allclose(full[5:14], part, atol=1e-5)


# ------------------------------------------------------ train + encode (CLI)
def test_train_reduces_loss_then_encode(tmp_path):
    sent, index = make_corpus(tmp_path)
    out = tmp_path / "run"
    history = train.run(train.parse_args([
        "--train-file", sent, "--train-index", index, "--val-file", sent, "--val-index", index,
        "--out-dir", str(out), "--maxlen", "8", "--word-size", "16", "--thought-size", "24",
        "--batch-size", "16", "--iters", "150", "--log-every", "1000", "--eval-every", "50",
        "--eval-batches", "3", "--save-every", "150", "--lr", "3e-3", "--device", "cpu"]))
    assert np.mean(history[-20:]) < 0.6 * np.mean(history[:20])
    assert (out / "vocab.txt").exists() and (out / "skipinstr-best.pt").exists() and (out / "skipinstr-last.pt").exists()

    prefix = tmp_path / "enc" / "train"
    encode.main(["--checkpoint", str(out / "skipinstr-best.pt"), "--sentences", sent, "--index", index,
                 "--out-prefix", str(prefix), "--batch-size", "7", "--device", "cpu"])
    encs = np.load(str(prefix) + ".encs.npy")
    meta = json.load(open(str(prefix) + ".index.json"))
    assert encs.dtype == np.float32 and encs.shape == (sum(meta["rlens"]), 24) and meta["dim"] == 24
    assert meta["starts"][1] == meta["rlens"][0] and len(meta["ids"]) == len(meta["starts"])
    # different batch size gives the same vectors
    encode.main(["--checkpoint", str(out / "skipinstr-best.pt"), "--sentences", sent, "--index", index,
                 "--out-prefix", str(prefix) + "_b", "--batch-size", "64", "--device", "cpu"])
    assert np.allclose(encs, np.load(str(prefix) + "_b.encs.npy"), atol=1e-5)


def test_resume_continues_from_saved_iteration(tmp_path):
    sent, index = make_corpus(tmp_path)
    out = tmp_path / "run"
    common = ["--train-file", sent, "--train-index", index, "--out-dir", str(out), "--maxlen", "8",
              "--word-size", "16", "--thought-size", "24", "--batch-size", "16", "--log-every", "1000",
              "--eval-every", "1000", "--save-every", "10", "--device", "cpu"]
    train.run(train.parse_args(common + ["--iters", "10"]))
    history = train.run(train.parse_args(common + ["--iters", "15", "--resume", str(out / "skipinstr-last.pt")]))
    assert len(history) == 5
    assert torch.load(out / "skipinstr-last.pt", weights_only=True)["iter"] == 15
