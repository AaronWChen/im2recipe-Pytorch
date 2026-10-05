import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from args import get_parser
from synthetic import DIM, INGREDIENTS, write_w2v
from trijoint import im2recipe
from word2vec_io import ingredient_row_offset, load_word2vec_bin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W2V_DIM = 6


def small_opts(pipeline, *extra):
    """The real parser, with everything shrunk so a CPU can run it."""
    return get_parser().parse_args([
        "--img_path", str(pipeline / "images"), "--data_path", str(pipeline / "data"),
        "--ingrW2V", str(pipeline / "vocab.bin"), "--ingrW2VDim", str(W2V_DIM), "--stDim", str(DIM),
        "--srnnDim", "16", "--irnnDim", "8", "--embDim", "12", "--numClasses", "5",
        "--no_pretrained", "--workers", "0", "--batch_size", "8", "--medr", "10", *extra])


# ---------------------------------------------------------------- no import-time side effects
def test_modules_import_without_parsing_argv():
    code = "import trijoint, train, test, image2embedding, word2vec_io"
    r = subprocess.run([sys.executable, "-c", code, "--definitely-not-an-option"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# --------------------------------------------------------------------------- word2vec reader
def test_word2vec_reader_keeps_file_order_and_duplicates(tmp_path):
    words = ["<pad>", "</i>", "salt", "olive_oil", "salt"]  # a duplicate must not shift or zero any row
    vecs = write_w2v(str(tmp_path / "w.bin"), words, 4)
    got_words, got_vecs = load_word2vec_bin(str(tmp_path / "w.bin"))
    assert got_words == words and got_vecs.dtype == np.float32 and np.array_equal(got_vecs, vecs)


def test_word2vec_reader_rejects_truncated_file(tmp_path):
    write_w2v(str(tmp_path / "w.bin"), ["a", "b", "c"], 4)
    with open(tmp_path / "w.bin", "r+b") as f:
        f.truncate(os.path.getsize(tmp_path / "w.bin") - 10)
    with pytest.raises(EOFError):
        load_word2vec_bin(str(tmp_path / "w.bin"))


def test_ingredient_alignment_check():
    words = ["<pad>", "</i>", "onion", "olive_oil", "garlic"]
    assert ingredient_row_offset(words, ["onion", "olive_oil", "garlic"]) == 2  # what build_dataset assumes
    assert ingredient_row_offset(["</s>", "onion", "olive_oil"], ["</s>", "onion", "olive_oil"]) == 0  # would be off by two
    assert ingredient_row_offset(words, ["onion", "garlic"]) is None
    assert ingredient_row_offset(words, []) is None


# ------------------------------------------------------------------------------------- model
def test_model_layout_and_forward(pipeline):
    opts = small_opts(pipeline)
    model = im2recipe(opts, pretrained=False)
    # parameter names are the contract with existing checkpoints
    prefixes = {k.split(".")[0] for k in model.state_dict()}
    assert prefixes == {"visionMLP", "visual_embedding", "recipe_embedding", "stRNN_", "ingRNN_", "semantic_branch"}
    _, expected = load_word2vec_bin(str(pipeline / "vocab.bin"))
    w = model.ingRNN_.embs.weight.data
    assert w.shape == (len(expected) + 2, W2V_DIM)
    assert torch.count_nonzero(w[:2]) == 0  # id 0 = padding, id 1 = end-of-ingredients token
    assert torch.equal(w[2:], torch.from_numpy(expected))  # vocab.txt line i has id i + 2 and its own vector
    # ... which is what the data side hands out
    from recipe_store import RecipeStore
    store = RecipeStore(str(pipeline / "data" / "train_store"))
    ids = {int(x) for i in range(len(store)) for x in store.ingredients(i) if x}
    assert ids <= set(range(1, len(w)))

    model.eval()
    x = torch.randn(3, 3, 64, 64)
    y1, y2 = torch.randn(3, 20, DIM), torch.tensor([5, 3, 4])
    z1, z2 = torch.randint(1, 6, (3, 20)), torch.tensor([2, 3, 1])
    with torch.no_grad():
        vis, rec, vis_sem, rec_sem = model(x, y1, y2, z1, z2)
    assert vis.shape == rec.shape == (3, 12) and vis_sem.shape == rec_sem.shape == (3, 5)
    assert torch.allclose(vis.norm(dim=1), torch.ones(3), atol=1e-5)
    assert torch.allclose(rec.norm(dim=1), torch.ones(3), atol=1e-5)
    # samples are independent of their batch mates (the sort/unsort in the RNN wrappers is right)
    with torch.no_grad():
        single = model(x[1:2], y1[1:2], y2[1:2], z1[1:2], z2[1:2])
    assert torch.allclose(single[1], rec[1:2], atol=1e-5)


def test_model_without_semantic_branch_and_dim_check(pipeline):
    opts = small_opts(pipeline)
    opts.semantic_reg = False
    model = im2recipe(opts, pretrained=False).eval()
    assert not hasattr(model, "semantic_branch")
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64), torch.randn(2, 20, DIM), torch.tensor([3, 4]),
                    torch.randint(1, 6, (2, 20)), torch.tensor([2, 2]))
    assert len(out) == 2
    bad = small_opts(pipeline, "--ingrW2VDim", "7")
    with pytest.raises(ValueError, match="6-dimensional"):
        im2recipe(bad, pretrained=False)


# --------------------------------------------------------------- train.py / test.py smoke runs
def test_train_then_test_scripts_run_end_to_end(pipeline, tmp_path):
    import train as train_script
    import test as test_script  # noqa: F401  (the repo's test.py; ROOT is first on sys.path)

    assert os.path.abspath(test_script.__file__) == os.path.join(ROOT, "test.py")
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    opts = small_opts(pipeline, "--epochs", "2", "--valfreq", "1", "--snapshots", str(snaps) + "/",
                      "--path_results", str(tmp_path) + "/")
    train_script.main(opts)
    checkpoints = sorted(p for p in os.listdir(snaps) if p.endswith(".pth.tar"))
    assert checkpoints, "validation ran but no checkpoint was written"

    test_opts = small_opts(pipeline, "--model_path", str(snaps / checkpoints[-1]), "--path_results", str(tmp_path) + "/")
    test_script.main(test_opts)
    import pickle
    with open(tmp_path / "rec_embeds.pkl", "rb") as f:
        rec = pickle.load(f)
    with open(tmp_path / "rec_ids.pkl", "rb") as f:
        ids = pickle.load(f)
    assert rec.shape == (len(ids), 12) and len(ids) == 10


# ------------------------------------------------------------------------------ checkpoints
def test_checkpoint_loading_is_safe_by_default(tmp_path):
    from checkpoint_io import load_checkpoint
    cpu = torch.device("cpu")
    plain = {"epoch": 2, "state_dict": {"w": torch.ones(2)}, "best_val": 5.5}
    torch.save(plain, tmp_path / "plain.pth.tar")
    assert load_checkpoint(str(tmp_path / "plain.pth.tar"), cpu)["best_val"] == 5.5

    legacy = {**plain, "best_val": np.float64(5.5)}  # what older train.py versions wrote
    torch.save(legacy, tmp_path / "legacy.pth.tar")
    with pytest.raises(RuntimeError, match="--trust_checkpoint"):
        load_checkpoint(str(tmp_path / "legacy.pth.tar"), cpu)
    assert load_checkpoint(str(tmp_path / "legacy.pth.tar"), cpu, trust=True)["epoch"] == 2


def test_check_ingredient_vocab_script(pipeline, tmp_path, capsys):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import check_ingredient_vocab
    args = ["--w2v", str(pipeline / "vocab.bin"), "--vocab", str(pipeline / "vocab.txt")]
    assert check_ingredient_vocab.main(args) == 0  # default: 2 reserved rows, vocab.txt in file order
    assert "OK" in capsys.readouterr().out
    assert check_ingredient_vocab.main(args + ["--extra-rows", "0"]) == 1  # would be off by two rows
    assert "MISMATCH" in capsys.readouterr().out
    # a vocab.bin that already starts with the two reserved rows, which vocab.txt skips, is the --extra-rows 0 layout
    write_w2v(str(tmp_path / "reserved.bin"), ["<pad>", "</i>"] + [t.replace(" ", "_") for t in INGREDIENTS], W2V_DIM)
    args = ["--w2v", str(tmp_path / "reserved.bin"), "--vocab", str(pipeline / "vocab.txt")]
    assert check_ingredient_vocab.main(args + ["--extra-rows", "0"]) == 0
    assert check_ingredient_vocab.main(args) == 1


# -------------------------------------------------------------------------------- arguments
def test_boolean_flags_really_accept_false():
    p = get_parser()
    assert p.parse_args([]).semantic_reg is True
    assert p.parse_args(["--semantic_reg", "False"]).semantic_reg is False  # type=bool made this True
    assert p.parse_args(["--freeVision", "true", "--freeRecipe", "0"]).freeVision is True
    assert p.parse_args(["--freeVision", "true", "--freeRecipe", "0"]).freeRecipe is False
    with pytest.raises(SystemExit):
        p.parse_args(["--semantic_reg", "maybe"])


def test_extra_rows_zero_uses_the_file_as_is(pipeline):
    opts = small_opts(pipeline, "--ingr_extra_rows", "0")
    _, expected = load_word2vec_bin(str(pipeline / "vocab.bin"))
    assert im2recipe(opts, pretrained=False).ingRNN_.embs.weight.shape[0] == len(expected)
