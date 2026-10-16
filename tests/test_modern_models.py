"""The text pipeline is not GPT-2-specific: the same steps work for a Llama-style model (RoPE, RMSNorm, grouped-query
attention, tied embeddings, BOS-adding tokenizer without a pad token) -- the design of today's Llama, Qwen and SmolLM models."""
import json
import os

import numpy as np
import pytest
import torch

from hf_tiny import make_gpt2, make_llama
from test_finetune import recipes
from textenc import encode as enc_cli
from textenc import finetune, prepare_lm
from textenc.models import encode_texts, load_encoder

TEXTS = ["heat the olive oil in a large pan .", "chop the onion .", "", "season with salt and serve warm ."]


@pytest.fixture(scope="module")
def llama(tmp_path_factory):
    return make_llama(str(tmp_path_factory.mktemp("llama") / "m"))


def test_encoder_handles_a_bos_adding_tokenizer_without_pad_token(llama):
    model, tok = load_encoder(llama)
    assert tok.pad_token_id is None and tok("chop")["input_ids"][0] == tok.bos_token_id
    for pooling in ("mean", "last", "first"):
        full = encode_texts(model, tok, TEXTS, pooling=pooling)
        assert full.shape == (4, 16) and np.isfinite(full).all()
        assert np.allclose(full, encode_texts(model, tok, TEXTS, pooling=pooling, batch_size=1), atol=1e-5)
        assert np.allclose(full[1], encode_texts(model, tok, [TEXTS[1]], pooling=pooling)[0], atol=1e-5)
    assert not np.allclose(encode_texts(model, tok, TEXTS[:2], pooling="last"), encode_texts(model, tok, TEXTS[:2], pooling="first"))
    # an empty text still gets a vector (just the BOS token)
    assert np.isfinite(encode_texts(model, tok, [""])).all()


def test_prefix_and_suffix_are_added_to_every_text(llama):
    model, tok = load_encoder(llama)
    plain = encode_texts(model, tok, ["chop the onion ."])
    with_prefix = encode_texts(model, tok, ["chop the onion ."], prefix="passage: ")
    assert not np.allclose(plain, with_prefix)
    assert np.allclose(with_prefix, encode_texts(model, tok, ["passage: chop the onion ."]), atol=1e-6)
    assert np.allclose(encode_texts(model, tok, ["chop"], suffix=" ."), encode_texts(model, tok, ["chop ."]), atol=1e-6)


def test_cli_records_prefix(llama, tmp_path):
    raw = tmp_path / "raw.txt"
    raw.write_text("chop the onion .\nheat the oil .\n")
    index = tmp_path / "i.tsv"
    index.write_text("a\t2\n")
    enc_cli.main(["--model", llama, "--sentences", str(raw), "--index", str(index), "--out-prefix", str(tmp_path / "o"),
                  "--prefix", "passage: ", "--device", "cpu"])
    assert json.load(open(tmp_path / "o.encoder.json"))["prefix"] == "passage: "


def test_prepare_finetune_and_encode_work_end_to_end_with_a_llama_style_model(llama, tmp_path):
    files = {}
    for name, texts in (("train", recipes(60)), ("val", recipes(12, 1))):
        files[name] = str(tmp_path / f"{name}.txt")
        open(files[name], "w").write("\n".join(texts) + "\n")
    meta = prepare_lm.prepare(llama, str(tmp_path / "data"), files, 256)
    assert meta["splits"]["train"]["skipped_too_long"] == 0 and meta["vocab_size"] == 300 + 13
    out = tmp_path / "ft"
    args = finetune.parse_args(["--model", llama, "--data", str(tmp_path / "data"), "--out", str(out), "--batch-size", "4",
                                "--grad-accum", "2", "--lr", "1e-2", "--warmup-steps", "3", "--max-steps", "40", "--log-every", "10",
                                "--eval-every", "20", "--device", "cpu", "--sample-ingredients", "onion, oil"])
    finetune.run(args, log=lambda *a: None)
    log = [json.loads(l) for l in open(out / "log.jsonl")]
    losses = [e["loss"] for e in log if "loss" in e]
    assert losses[-1] < 0.7 * losses[0]
    assert "model.safetensors" in os.listdir(out / "best")
    from transformers import AutoModelForCausalLM
    lm = AutoModelForCausalLM.from_pretrained(str(out / "best"), use_safetensors=True)
    assert lm.get_input_embeddings().num_embeddings == 313 and lm.config.tie_word_embeddings
    assert open(out / "sample.txt").read().startswith("<RECIPE_START> <INPUT_START> onion")
    tuned, tok = load_encoder(str(out / "best"))  # the fine-tuned folder is a drop-in encoder, as before
    assert encode_texts(tuned, tok, TEXTS).shape == (4, 16)


def test_a_second_architecture_changes_only_the_model_folder(tmp_path):
    """Same raw text, two different models: only --model differs and the output files have the same format."""
    raw = tmp_path / "raw.txt"
    raw.write_text("chop the onion .\nheat the oil .\nadd salt .\n")
    index = tmp_path / "i.tsv"
    index.write_text("a\t2\nb\t1\n")
    dims = {}
    for name, maker in (("gpt2", make_gpt2), ("llama", make_llama)):
        path = maker(str(tmp_path / name))
        enc_cli.main(["--model", path, "--sentences", str(raw), "--index", str(index), "--out-prefix", str(tmp_path / f"{name}_out"),
                      "--device", "cpu"])
        dims[name] = np.load(str(tmp_path / f"{name}_out.encs.npy")).shape
        assert json.load(open(tmp_path / f"{name}_out.index.json"))["rlens"] == [2, 1]
    assert dims["gpt2"] == dims["llama"] == (3, 16)
