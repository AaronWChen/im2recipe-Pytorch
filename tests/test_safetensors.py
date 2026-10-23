import json
import os
import pickle

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import train as train_script
from checkpoint_io import load_checkpoint
from safetensors_io import load_safetensors, save_safetensors
from skipinstructions import encode as skip_encode
from skipinstructions import export as skip_export
from test_trijoint import small_opts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_roundtrip_with_metadata_shared_and_noncontiguous_tensors(tmp_path):
    base = torch.randn(4, 4)
    sd = {"a": base, "b": base, "t": base.t()}  # shared storage and a transposed view: safetensors refuses both raw
    meta = {"epoch": 3, "best_val": float("inf"), "cfg": {"dim": 8, "tags": ["x", "y"]}, "name": "x", "ok": True}
    save_safetensors(str(tmp_path / "m.safetensors"), sd, meta)
    loaded, got = load_safetensors(str(tmp_path / "m.safetensors"))
    assert all(torch.equal(loaded[k], sd[k]) for k in sd) and loaded["t"].is_contiguous()
    assert got == meta and got["best_val"] == float("inf")


def test_metadata_written_by_other_tools_is_kept_as_text(tmp_path):
    save_file({"w": torch.ones(2)}, str(tmp_path / "x.safetensors"), metadata={"format": "pt", "note": "plain words"})
    _, meta = load_safetensors(str(tmp_path / "x.safetensors"))
    assert meta == {"format": "pt", "note": "plain words"}


# ------------------------------------------------------------------------------------ skipinstr
def test_skipinstr_checkpoint_has_a_safetensors_twin_that_encodes_identically(pipeline, tmp_path):
    run = pipeline / "run"
    assert (run / "skipinstructions-last.pt").exists() and (run / "skipinstructions-last.safetensors").exists()
    state, meta = load_safetensors(str(run / "skipinstructions-last.safetensors"))
    assert meta["format"] == "skipinstructions-v1" and meta["config"]["thought_size"] == 24 and meta["iter"] == 30
    text = pipeline / "text"
    outs = {}
    for name, ckpt in (("pt", "skipinstructions-last.pt"), ("st", "skipinstructions-last.safetensors")):
        skip_encode.main(["--checkpoint", str(run / ckpt), "--sentences", str(text / "instructions_val.txt"),
                          "--index", str(text / "instructions_val.index.tsv"), "--out-prefix", str(tmp_path / name),
                          "--vocab", str(run / "vocab.txt"), "--device", "cpu"])
        outs[name] = np.load(str(tmp_path / name) + ".encs.npy")
    assert np.array_equal(outs["pt"], outs["st"])


def test_export_converts_an_existing_pt_checkpoint(pipeline, tmp_path):
    out = skip_export.export(str(pipeline / "run" / "skipinstructions-last.pt"), str(tmp_path / "old.safetensors"))
    state, meta = load_safetensors(out)
    ref = torch.load(pipeline / "run" / "skipinstructions-last.pt", weights_only=True)
    assert meta["config"] == ref["config"] and all(torch.equal(state[k], ref["model"][k]) for k in state)


# ------------------------------------------------------------------------------------ trijoint
def fake_state(epoch=2, best=5.5):
    return {"epoch": epoch, "state_dict": {"w": torch.ones(3), "b": torch.zeros(2)}, "best_val": best,
            "optimizer": {"state": {}, "param_groups": []}, "valtrack": 0, "freeVision": True, "curr_val": 5.6}


@pytest.mark.parametrize("fmt, expected", [("both", {".pth.tar", ".safetensors"}), ("torch", {".pth.tar"}),
                                           ("safetensors", {".safetensors"})])
def test_save_checkpoint_formats_and_no_trailing_slash(pipeline, tmp_path, fmt, expected):
    opts = small_opts(pipeline, "--snapshots", str(tmp_path / "snaps"), "--checkpoint_format", fmt)  # no trailing "/"
    train_script.save_checkpoint(fake_state(), False, opts)
    assert not (tmp_path / "snaps").exists()  # only a new best is written
    train_script.save_checkpoint(fake_state(), True, opts)
    files = os.listdir(tmp_path / "snaps")
    assert {os.path.splitext(f)[1] if f.endswith(".safetensors") else ".pth.tar" for f in files} == expected
    assert all(f.startswith("model_e002_v-5.500") for f in files)
    if "safetensors" in fmt or fmt == "both":
        ck = load_checkpoint(str(tmp_path / "snaps" / "model_e002_v-5.500.safetensors"), torch.device("cpu"))
        assert ck["epoch"] == 2 and ck["best_val"] == 5.5 and ck["freeVision"] is True and "optimizer" not in ck
        assert torch.equal(ck["state_dict"]["w"], torch.ones(3)) and "opts" not in ck


@pytest.fixture(scope="module")
def trained(pipeline, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("trained")
    opts = small_opts(pipeline, "--epochs", "2", "--valfreq", "1", "--snapshots", str(tmp / "snaps"),
                      "--checkpoint_format", "both", "--path_results", str(tmp / "res"))
    train_script.main(opts)
    names = sorted(os.listdir(tmp / "snaps"))
    stem = [n for n in names if n.endswith(".safetensors")][-1][:-len(".safetensors")]
    return tmp, stem


def test_both_formats_hold_the_same_weights(trained):
    tmp, stem = trained
    pth = torch.load(tmp / "snaps" / (stem + ".pth.tar"), map_location="cpu", weights_only=True)
    st, meta = load_safetensors(str(tmp / "snaps" / (stem + ".safetensors")))
    assert st.keys() == pth["state_dict"].keys() and all(torch.equal(st[k], pth["state_dict"][k]) for k in st)
    assert meta["epoch"] == pth["epoch"] == 2 and meta["best_val"] == pth["best_val"]
    assert meta["format"] == "im2recipe-trijoint-v1" and meta["opts"]["checkpoint_format"] == "both"
    assert meta["opts"]["stDim"] == 24  # the settings the weights were trained with travel with them


def test_test_script_loads_a_safetensors_checkpoint(pipeline, trained):
    import test as test_script
    tmp, stem = trained
    opts = small_opts(pipeline, "--model_path", str(tmp / "snaps" / (stem + ".safetensors")),
                      "--path_results", str(tmp / "res2"))  # no trailing slash either
    test_script.main(opts)
    with open(tmp / "res2" / "rec_embeds.pkl", "rb") as f:
        rec = pickle.load(f)
    assert rec.shape == (10, 12)


def test_resume_from_safetensors_restarts_the_optimizer_in_the_right_phase(pipeline, trained, capsys):
    tmp, stem = trained
    opts = small_opts(pipeline, "--epochs", "4", "--valfreq", "1", "--snapshots", str(tmp / "snaps_resumed"),
                      "--resume", str(tmp / "snaps" / (stem + ".safetensors")), "--checkpoint_format", "safetensors")
    train_script.main(opts)
    out = capsys.readouterr().out
    assert "no optimizer state" in out and "(epoch 2)" in out
    assert "Validation" in out  # epochs 2 and 3 ran
