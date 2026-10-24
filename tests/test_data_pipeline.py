"""End-to-end test of the data path on a tiny synthetic Recipe1M:

tokenize_instructions -> skipinstructions.train -> skipinstructions.encode -> scripts/build_dataset -> ImagerLoader
"""

import json
import os

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

import build_dataset
import proc
import utils
from data_loader import ImagerLoader
from recipe_store import RecipeStore, write_store
from synthetic import DIM


def test_filters_and_counts(pipeline):
    stores = {
        p: RecipeStore(str(pipeline / "data" / f"{p}_store"))
        for p in ("train", "val", "test")
    }
    train_ids = [stores["train"].recipe_id(i) for i in range(len(stores["train"]))]
    assert (
        "r0" not in train_ids and "r1" not in train_ids
    )  # removal file (incl. its first entry)
    assert "r4" not in train_ids  # no images
    assert "r5" not in train_ids  # 25 instructions
    assert len(train_ids) == int(48 * 0.6) - 4
    assert (
        stores["val"].meta["n_recipes"]
        == len(stores["val"])
        == int(48 * 0.8) - int(48 * 0.6)
    )


def test_vectors_match_encoder_output(pipeline):
    store = RecipeStore(str(pipeline / "data" / "train_store"))
    encs = np.load(str(pipeline / "skipinstructions" / "train.encs.npy"))
    index = json.load(open(str(pipeline / "skipinstructions" / "train.index.json")))
    start = dict(zip(index["ids"], index["starts"]))
    rlen = dict(zip(index["ids"], index["rlens"]))
    for i in range(len(store)):
        rid = store.recipe_id(i)
        assert np.array_equal(
            store.instructions(i), encs[start[rid] : start[rid] + rlen[rid]]
        )
    assert store.dim == DIM


def test_recipe_fields(pipeline):
    store = RecipeStore(str(pipeline / "data" / "train_store"))
    i = [store.recipe_id(k) for k in range(len(store))].index("r3")
    n = 3
    assert store.recipe_class(i) == 3 % 5
    assert store.image_names(i) == [f"03abcdef{j}.jpg" for j in range(1 + 3 % 7)]
    ingrs = store.ingredients(i)
    assert ingrs.dtype == np.uint16 and len(ingrs) == 20
    # r3 detects 4 ingredients, plus the end token (id 1) last
    assert (
        np.count_nonzero(ingrs) == (1 + 3 % 4) + 1
        and ingrs[np.count_nonzero(ingrs) - 1] == 1
    )
    assert len(store.instructions(i)) == 3 + 3 % 4


def test_loader_end_to_end(pipeline):
    img_root = pipeline / "images"
    store = RecipeStore(str(pipeline / "data" / "train_store"))
    tf = transforms.Compose(
        [transforms.Resize(16), transforms.CenterCrop(16), transforms.ToTensor()]
    )

    ds = ImagerLoader(
        str(img_root),
        transform=tf,
        data_path=str(pipeline / "data"),
        partition="train",
        sem_reg=True,
    )
    assert len(ds) == len(store)
    (img, instrs, itr_ln, ingrs, igr_ln), (target, img_class, rec_class) = ds[0]
    assert (
        img.shape == (3, 16, 16)
        and instrs.shape == (20, DIM)
        and instrs.dtype == torch.float32
    )
    assert (
        3 <= itr_ln <= 6
        and ingrs.dtype == torch.long
        and 1 <= igr_ln <= 5
        and target in (1, -1)
    )
    assert (
        torch.count_nonzero(instrs[itr_ln:]) == 0
    )  # zero padding after the real instructions

    val = ImagerLoader(
        str(img_root), transform=tf, data_path=str(pipeline / "data"), partition="val"
    )
    (_, _, _, _, _), (target, img_id, rec_id) = val[0]
    assert target == 1 and img_id == rec_id  # val/test are always matching pairs

    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, num_workers=2)
    inputs, targets = next(iter(dl))
    assert (
        inputs[0].shape == (8, 3, 16, 16)
        and inputs[1].shape == (8, 20, DIM)
        and inputs[2].shape == (8,)
    )

    with pytest.raises(ValueError):
        ImagerLoader(str(img_root), data_path=str(pipeline / "data"), partition="dev")


def test_float16_build_halves_the_vectors(pipeline, tmp_path):
    out = tmp_path / "data16"
    build_dataset.main(
        [
            "--dataset",
            str(pipeline),
            "--vocab",
            str(pipeline / "vocab.txt"),
            "--classes",
            str(pipeline / "classes.pkl"),
            "--remove",
            str(pipeline / "remove.txt"),
            "--skip-dir",
            str(pipeline / "skipinstructions"),
            "--out-dir",
            str(out),
            "--dtype",
            "float16",
        ]
    )
    s16 = RecipeStore(str(out / "train_store"))
    s32 = RecipeStore(str(pipeline / "data" / "train_store"))
    assert s16.meta["dtype"] == "float16" and len(s16) == len(s32)
    assert np.allclose(s16.instructions(0), s32.instructions(0), atol=2e-3)
    assert os.path.getsize(out / "train_store" / "intrs.npy") < 0.6 * os.path.getsize(
        pipeline / "data" / "train_store" / "intrs.npy"
    )


def test_write_store_handles_empty_partition(tmp_path):
    write_store(
        str(tmp_path / "s"),
        [],
        np.zeros((0, 20), "uint16"),
        [],
        [],
        np.zeros((0, 8), "float32"),
        [],
    )
    assert len(RecipeStore(str(tmp_path / "s"))) == 0


# ------------------------------------------------- Python 3 / SciPy-free helpers
def test_proc_images_without_scipy(tmp_path):
    for name, mode in (("rgb.jpg", "RGB"), ("gray.png", "L"), ("rgba.png", "RGBA")):
        Image.new(mode, (60, 40)).save(tmp_path / name)
        img, fail = proc.process_image(str(tmp_path / name), 20)
        assert fail == 0 and img.shape == (20, 30, 3)
        assert proc.read_image(str(tmp_path / name)).shape == (224, 224, 3)
    img, fail = proc.process_image(str(tmp_path / "missing.jpg"), 20)
    assert fail == 1 and img.shape == (20, 20, 3)


def test_utils_python3():
    assert (
        utils.prepro_txt("Mix &amp; bake at 350\x92 and 1/2 cup")
        == "Mix & bake at 350' and 1/2 cup"
    )
    assert utils.prepro_txt("50%") == "50%"  # an errant % must not raise
    assert utils.align(["a", "b", "c", "d"], ["b", "c"]).tolist() == [0, 1, 1, 0]
    assert utils.vstack({"train": [[1, 2], [3, 4]]})["train"].shape == (2, 2)
    assert utils.mstack({"train": [[1], [2, 3]]}, 3)["train"].tolist() == [
        [1, 0, 0],
        [2, 3, 0],
    ]


def test_image_root_keeps_only_images_that_exist_and_drops_recipes_without_any(
    pipeline, tmp_path, capsys
):
    import build_dataset

    # the synthetic recipes r0..r47 list images f"{i:02d}abcdef{j}.jpg" for j in range(1 + i % 7); pretend only the
    # images with i + j even are in the image folder
    have = tmp_path / "partial"
    for i in range(48):
        for j in range(1 + i % 7):
            if (i + j) % 2 == 0:
                name = f"{i:02d}abcdef{j}.jpg"
                d = have.joinpath(*name[:4])
                d.mkdir(parents=True, exist_ok=True)
                (d / name).write_bytes(b"x")
    out = tmp_path / "data"
    build_dataset.main(
        [
            "--dataset",
            str(pipeline),
            "--vocab",
            str(pipeline / "vocab.txt"),
            "--classes",
            str(pipeline / "classes.pkl"),
            "--remove",
            str(pipeline / "remove.txt"),
            "--skip-dir",
            str(pipeline / "skipinstructions"),
            "--out-dir",
            str(out),
            "--image-root",
            str(have),
        ]
    )
    printed = capsys.readouterr().out
    dropped = {
        7,
        21,
        35,
    }  # odd i with a single image (i % 7 == 0): none of their images exist
    assert "no_available_image=3" in printed
    kept = 0
    for part in ("train", "val", "test"):
        base = RecipeStore(str(pipeline / "data" / f"{part}_store"))
        store = RecipeStore(str(out / f"{part}_store"))
        base_ids = [base.recipe_id(k) for k in range(len(base))]
        want = [r for r in base_ids if int(r[1:]) not in dropped]
        assert [store.recipe_id(k) for k in range(len(store))] == want
        for k in range(len(store)):
            i = int(store.recipe_id(k)[1:])
            expected = [
                f"{i:02d}abcdef{j}.jpg" for j in range(1 + i % 7) if (i + j) % 2 == 0
            ][:5]
            assert (
                store.image_names(k) == expected
            )  # only existing images, in their original order
            assert all(
                os.path.exists(have.joinpath(*n[:4], n)) for n in store.image_names(k)
            )
            kept += 1
    assert (
        kept
        == sum(
            len(RecipeStore(str(pipeline / "data" / f"{p}_store")))
            for p in ("train", "val", "test")
        )
        - 3
    )
