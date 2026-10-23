import json
import math
import os

import numpy as np
import pytest
pytest.importorskip("transformers", reason="pip install -r requirements.txt")
pytest.importorskip("tokenizers", reason="pip install -r requirements.txt")
import torch

from hf_tiny import make_gpt2, make_pickle_only
from textenc import finetune, prepare_lm
from textenc.lmdata import END_TOKEN, SPECIAL_TOKENS, prompt_for, recipe_to_text
from textenc.models import encode_texts, load_encoder

BLOCK = 256  # a tiny-tokenizer recipe is about 70 tokens


def recipes(n, offset=0):
    out = []
    for i in range(n):
        j = (i + offset) % 4
        out.append(recipe_to_text(["chop onion", "heat the oil", "add garlic", "season with salt"][j],
                                  ["1 onion", "2 tbsp oil"], ["Chop the onion.", "Heat it in the oil."], ["onion", "oil"]))
    return out


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lm")
    model = make_gpt2(str(tmp / "base"), n_positions=BLOCK)
    long = recipe_to_text("a very long one", ["1 onion"], ["chop the onion and heat the oil and add garlic. "] * 30, ["onion"])
    files = {}
    for name, texts in (("train", recipes(60) + [long]), ("val", recipes(12, 1)), ("holdout", recipes(8, 2))):
        files[name] = str(tmp / f"{name}.txt")
        open(files[name], "w").write("\n".join(texts) + "\n")
    meta = prepare_lm.prepare(model, str(tmp / "data"), files, BLOCK)
    return tmp, model, meta, files


# ---------------------------------------------------------------------------------- prepare_lm
def test_tags_become_single_tokens_and_the_tokenizer_is_saved(prepared):
    tmp, model, meta, _ = prepared
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tmp / "data" / "tokenizer"))
    ids = tok.convert_tokens_to_ids(SPECIAL_TOKENS)
    assert len(set(ids)) == 13 and all(i < len(tok) for i in ids) and None not in ids
    assert tok(recipes(1)[0], add_special_tokens=False)["input_ids"][0] == tok.convert_tokens_to_ids("<RECIPE_START>")
    assert meta["vocab_size"] == len(tok) and meta["dtype"] == "uint16" and meta["end_id"] == tok.convert_tokens_to_ids(END_TOKEN)


def test_packing_keeps_recipes_whole_and_in_order(prepared):
    tmp, model, meta, files = prepared
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tmp / "data" / "tokenizer"))
    info = meta["splits"]["train"]
    assert info["skipped_too_long"] == 1 and info["recipes"] == 60
    ids = np.fromfile(tmp / "data" / "train.bin", dtype=np.uint16).reshape(info["blocks"], BLOCK)
    lengths = np.load(tmp / "data" / "train.lengths.npy")
    assert lengths.min() > 0 and lengths.max() <= BLOCK and len(lengths) == info["blocks"]
    assert (ids[lengths < BLOCK][:, -1] == meta["end_id"]).all()  # padding is <RECIPE_END>
    real = np.concatenate([row[:n] for row, n in zip(ids, lengths)])
    expected = np.concatenate([tok(t, add_special_tokens=False)["input_ids"] for t in recipes(60)])
    assert np.array_equal(real, expected)  # nothing lost, nothing reordered
    start = tok.convert_tokens_to_ids("<RECIPE_START>")
    for row, n in zip(ids, lengths):
        assert row[0] == start  # a block never begins in the middle of a recipe
    assert info["tokens"] == len(expected)


# ---------------------------------------------------------------------------------- training
def test_schedule_and_precision_helpers():
    assert finetune.lr_at(0, 100, 1e-3, 10) == pytest.approx(1e-4)
    assert finetune.lr_at(9, 100, 1e-3, 10) == pytest.approx(1e-3)
    assert finetune.lr_at(100, 100, 1e-3, 10, 0.1) == pytest.approx(1e-4)
    assert finetune.lr_at(55, 100, 1e-3, 10, 0.1) < finetune.lr_at(20, 100, 1e-3, 10, 0.1)
    assert finetune.pick_precision("auto", torch.device("cpu")) == "fp32"
    with pytest.raises(ValueError, match="CUDA"):
        finetune.run(finetune.parse_args(["--model", "x", "--data", "x", "--out", "x", "--precision", "fp16", "--device", "cpu"]))


def args(prepared, out, *extra):
    tmp, model, _, _ = prepared
    return finetune.parse_args(["--model", model, "--data", str(tmp / "data"), "--out", str(out), "--batch-size", "4",
                                "--grad-accum", "2", "--lr", "1e-2", "--warmup-steps", "3", "--log-every", "5",
                                "--eval-every", "10", "--save-every", "10", "--eval-blocks", "6", "--device", "cpu", *extra])


@pytest.fixture(scope="module")
def finetuned(prepared, tmp_path_factory):
    out = tmp_path_factory.mktemp("ft") / "model"
    finetune.run(args(prepared, out, "--max-steps", "40", "--sample-ingredients", "onion, oil"), log=lambda *a: None)
    return out


def test_training_learns_and_writes_safetensors_folders(prepared, finetuned):
    out = finetuned
    log = [json.loads(l) for l in open(out / "log.jsonl")]
    losses = [e["loss"] for e in log if "loss" in e]
    evals = [e for e in log if "val_loss" in e]
    assert losses[-1] < 0.7 * losses[0]
    assert [e["step"] for e in evals] == [10, 20, 30, 40]
    assert all({"val_ppl", "holdout_loss", "holdout_ppl"} <= e.keys() for e in evals)
    assert evals[0]["val_ppl"] == pytest.approx(math.exp(evals[0]["val_loss"]))
    for d in ("best", "last"):
        assert "model.safetensors" in os.listdir(out / d) and "tokenizer.json" in os.listdir(out / d)
    assert (out / "last" / "trainer_state.pt").exists() and not (out / "best" / "trainer_state.pt").exists()
    best = min(e["val_loss"] for e in evals)
    assert any(e["val_loss"] == best for e in evals)
    assert not os.path.exists(out / "last.tmp") and not os.path.exists(out / "best.tmp")


def test_embeddings_were_resized_for_the_tags_and_the_saved_model_generates(prepared, finetuned):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = finetuned
    tok = AutoTokenizer.from_pretrained(str(out / "best"))
    lm = AutoModelForCausalLM.from_pretrained(str(out / "best"), use_safetensors=True)
    assert lm.get_input_embeddings().num_embeddings == len(tok) == prepared[2]["vocab_size"]
    assert lm.config.use_cache is True
    sample = open(out / "sample.txt").read()
    assert sample.startswith("<RECIPE_START> <INPUT_START> onion <NEXT_INPUT> oil <INPUT_END> <INGR_START>")
    assert prompt_for(["onion", "oil"]) in sample


def test_resume_continues_from_the_saved_step(prepared, tmp_path):
    out = tmp_path / "m"
    finetune.run(args(prepared, out, "--max-steps", "10"), log=lambda *a: None)
    seen = []
    finetune.run(args(prepared, out, "--max-steps", "20", "--resume"), log=lambda m: seen.append(m))
    assert any("resumed at update 10 of 20" in m for m in seen)
    steps = [json.loads(l)["step"] for l in open(out / "log.jsonl") if '"loss"' in l]
    assert steps[-1] == 20 and 5 in steps and 15 in steps
    with pytest.raises(FileNotFoundError):
        finetune.run(args(prepared, tmp_path / "nothing", "--resume"), log=lambda *a: None)


def test_finetuned_folder_is_a_drop_in_for_the_encoder(finetuned):
    model, tok = load_encoder(str(finetuned / "best"))  # safetensors-only policy
    vec = encode_texts(model, tok, ["chop the onion .", "heat the oil ."], pooling="mean")
    assert vec.shape == (2, 16) and np.isfinite(vec).all()


def test_pickle_only_starting_models_are_refused(prepared, tmp_path):
    tmp, model, _, _ = prepared
    pk = make_pickle_only(model, str(tmp_path / "pk"))
    a = args(prepared, tmp_path / "o", "--max-steps", "1")
    a.model = pk
    with pytest.raises(OSError):
        finetune.run(a, log=lambda *x: None)
    a.allow_pickle = True
    finetune.run(a, log=lambda *x: None)
