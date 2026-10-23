import json
import os

import numpy as np
import pytest
pytest.importorskip("transformers", reason="pip install -r requirements.txt")
pytest.importorskip("tokenizers", reason="pip install -r requirements.txt")
import torch

import build_dataset
from hf_tiny import make_bert, make_gpt2, make_pickle_only
from recipe_store import RecipeStore
from skipinstructions.data import read_index, write_index
from synthetic import STEPS
from textenc import encode as enc_cli
from textenc import to_safetensors
from textenc.models import encode_texts, load_encoder
from textenc.pooling import pool

SENTENCES = ["heat the olive oil in a large pan .", "chop the onion .", "", "season with salt and serve warm .",
             "add garlic and onion , stir for two minutes . " * 6, "bake for twenty minutes at 350 degrees ."]


@pytest.fixture(scope="module")
def gpt2(tmp_path_factory):
    path = make_gpt2(str(tmp_path_factory.mktemp("models") / "gpt2"))
    model, tok = load_encoder(path)
    return path, model, tok


# ------------------------------------------------------------------------------------ pooling
def test_pooling_modes_ignore_padding():
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]])
    assert torch.allclose(pool(hidden, mask, "mean")[0], hidden[0, :3].mean(0))
    assert torch.equal(pool(hidden, mask, "mean")[1], hidden[1, 0])
    assert torch.equal(pool(hidden, mask, "last"), torch.stack([hidden[0, 2], hidden[1, 0]]))
    assert torch.equal(pool(hidden, mask, "first"), hidden[:, 0])
    with pytest.raises(ValueError):
        pool(hidden, mask, "max")


# ------------------------------------------------------------------------------------ encoder
def test_vectors_do_not_depend_on_batching_or_padding(gpt2):
    _, model, tok = gpt2
    for pooling in ("mean", "last", "first"):
        full = encode_texts(model, tok, SENTENCES, pooling=pooling, batch_size=64)
        assert full.shape == (len(SENTENCES), 16) and full.dtype == np.float32 and np.isfinite(full).all()
        for bs in (1, 2, 4):
            assert np.allclose(full, encode_texts(model, tok, SENTENCES, pooling=pooling, batch_size=bs), atol=1e-5)
        alone = encode_texts(model, tok, [SENTENCES[1]], pooling=pooling)
        assert np.allclose(alone[0], full[1], atol=1e-5)  # same vector next to a long, padded neighbour


def test_empty_text_gets_a_vector_and_truncation_works(gpt2):
    _, model, tok = gpt2
    vecs = encode_texts(model, tok, ["", "chop the onion ."])
    assert np.isfinite(vecs).all() and not np.allclose(vecs[0], vecs[1])
    long = "add garlic and onion , stir for two minutes . " * 20
    ids = tok(long, add_special_tokens=True)["input_ids"]
    assert len(ids) > 10
    with torch.no_grad():
        expected = model(input_ids=torch.tensor([ids[:10]])).last_hidden_state.mean(1)[0].numpy()
    assert np.allclose(encode_texts(model, tok, [long], max_length=10)[0], expected, atol=1e-5)
    with pytest.raises(ValueError, match="positions"):
        encode_texts(model, tok, ["x"], max_length=1000)


def test_layer_and_normalize_options(gpt2):
    _, model, tok = gpt2
    text = ["chop the onion ."]
    ids = torch.tensor([tok(text[0])["input_ids"]])
    with torch.no_grad():
        hs = model(input_ids=ids, output_hidden_states=True).hidden_states
    assert np.allclose(encode_texts(model, tok, text, layer=1)[0], hs[1].mean(1)[0].numpy(), atol=1e-5)
    assert not np.allclose(encode_texts(model, tok, text, layer=1), encode_texts(model, tok, text))
    norms = np.linalg.norm(encode_texts(model, tok, SENTENCES, l2_normalize=True), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_bert_style_model_uses_cls_for_first_pooling(tmp_path):
    model, tok = load_encoder(make_bert(str(tmp_path / "bert")))
    first = encode_texts(model, tok, SENTENCES[:2], pooling="first")
    ids = torch.tensor([tok(SENTENCES[1])["input_ids"]])
    assert ids[0, 0] == tok.cls_token_id
    with torch.no_grad():
        assert np.allclose(first[1], model(input_ids=ids).last_hidden_state[0, 0].numpy(), atol=1e-5)


# ------------------------------------------------------------------------- safetensors policy
def test_pickled_models_are_refused_unless_allowed_and_can_be_converted(tmp_path, gpt2):
    path, model, tok = gpt2
    pk = make_pickle_only(path, str(tmp_path / "pickled"))
    with pytest.raises(OSError, match="to_safetensors"):
        load_encoder(pk)
    loaded, _ = load_encoder(pk, allow_pickle=True)
    ref = encode_texts(model, tok, SENTENCES)
    assert np.allclose(encode_texts(loaded, tok, SENTENCES), ref, atol=1e-6)

    out = str(tmp_path / "converted")
    with pytest.raises(OSError, match="allow-pickle"):
        to_safetensors.convert(pk, out)
    assert to_safetensors.convert(pk, out, allow_pickle=True) == "GPT2LMHeadModel"  # keeps its LM head
    assert any(f.endswith(".safetensors") for f in os.listdir(out)) and "tokenizer.json" in os.listdir(out)
    converted, ctok = load_encoder(out)  # now loads under the default, safetensors-only policy
    assert np.allclose(encode_texts(converted, ctok, SENTENCES), ref, atol=1e-6)


def test_a_finetuned_model_saved_with_save_pretrained_is_a_drop_in(tmp_path, gpt2):
    """What 'fine-tune later' produces: a local folder. Change a weight, save, point --model at it."""
    from transformers import GPT2LMHeadModel
    path, _, tok = gpt2
    lm = GPT2LMHeadModel.from_pretrained(path)
    with torch.no_grad():
        # random noise: a constant shift would be cancelled by LayerNorm and change nothing
        w = lm.transformer.wte.weight
        w.add_(torch.randn(w.shape, generator=torch.Generator().manual_seed(0)) * 0.5)
    lm.save_pretrained(str(tmp_path / "ft"))
    tok.save_pretrained(str(tmp_path / "ft"))
    assert "model.safetensors" in os.listdir(tmp_path / "ft")
    tuned, ttok = load_encoder(str(tmp_path / "ft"))
    base, _ = load_encoder(path)
    assert not np.allclose(encode_texts(tuned, ttok, SENTENCES[:2]), encode_texts(base, tok, SENTENCES[:2]))


# ------------------------------------------------------------------------------------- CLI
def write_raw(tmp_path, n_recipes=5):
    lines, ids, rlens = [], [], []
    for i in range(n_recipes):
        k = 2 + i % 3
        lines += [STEPS[j % len(STEPS)] for j in range(k)]
        ids.append(f"x{i}")
        rlens.append(k)
    raw, index = tmp_path / "raw.txt", tmp_path / "raw.index.tsv"
    raw.write_text("\n".join(lines) + "\n")
    write_index(str(index), ids, rlens)
    return str(raw), str(index), ids, rlens, lines


def test_cli_writes_the_files_build_dataset_reads(tmp_path, gpt2):
    path, model, tok = gpt2
    raw, index, ids, rlens, lines = write_raw(tmp_path)
    prefix = str(tmp_path / "out" / "train")
    enc_cli.main(["--model", path, "--sentences", raw, "--index", index, "--out-prefix", prefix,
                  "--batch-size", "3", "--window", "4", "--device", "cpu", "--pooling", "last"])
    encs = np.load(prefix + ".encs.npy")
    meta = json.load(open(prefix + ".index.json"))
    assert encs.shape == (len(lines), 16) and meta["dim"] == 16 and meta["ids"] == ids and meta["rlens"] == rlens
    assert meta["starts"][1] == rlens[0]
    assert np.allclose(encs, encode_texts(model, tok, lines, pooling="last"), atol=1e-5)  # window/batch size irrelevant
    prov = json.load(open(prefix + ".encoder.json"))
    assert prov["pooling"] == "last" and prov["model"] == path and prov["dim"] == 16

    enc_cli.main(["--model", path, "--sentences", raw, "--index", index, "--out-prefix", prefix + "16",
                  "--dtype", "float16", "--device", "cpu"])
    e16 = np.load(prefix + "16.encs.npy")
    assert e16.dtype == np.float16 and np.allclose(e16, encode_texts(model, tok, lines), atol=5e-3)

    write_index(index, ids, [r + 1 for r in rlens])  # an index that disagrees with the text
    with pytest.raises(ValueError, match="covers"):
        enc_cli.main(["--model", path, "--sentences", raw, "--index", index, "--out-prefix", prefix, "--device", "cpu"])


# ------------------------------------------- whole path: raw text -> encode -> build_dataset -> loader
def test_gpt2_vectors_flow_through_build_dataset_and_loader(pipeline, gpt2, tmp_path):
    from torch.utils.data import DataLoader
    from data_loader import ImagerLoader
    from torchvision import transforms

    path = gpt2[0]
    text = pipeline / "text"
    out = tmp_path / "gpt2"
    for part in ("train", "val", "test"):
        raw = text / f"raw_instructions_{part}.txt"
        index = text / f"instructions_{part}.index.tsv"
        assert len(raw.read_text().split("\n")) - 1 == int(read_index(str(index))[1].sum())
        enc_cli.main(["--model", path, "--sentences", str(raw), "--index", str(index),
                      "--out-prefix", str(out / part), "--device", "cpu"])
    build_dataset.main(["--dataset", str(pipeline), "--vocab", str(pipeline / "vocab.txt"), "--classes", str(pipeline / "classes.pkl"),
                        "--remove", str(pipeline / "remove.txt"), "--skip-dir", str(out), "--out-dir", str(tmp_path / "data")])
    store = RecipeStore(str(tmp_path / "data" / "train_store"))
    assert store.dim == 16 and len(store) == len(RecipeStore(str(pipeline / "data" / "train_store")))
    tf = transforms.Compose([transforms.Resize(16), transforms.CenterCrop(16), transforms.ToTensor()])
    ds = ImagerLoader(str(pipeline / "images"), transform=tf, data_path=str(tmp_path / "data"), partition="train")
    inputs, _ = next(iter(DataLoader(ds, batch_size=4)))
    assert inputs[1].shape == (4, 20, 16)  # a model trained on this uses --stDim 16
