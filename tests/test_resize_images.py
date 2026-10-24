import io
import os
import tarfile

import numpy as np
import pytest
from PIL import Image
from torchvision import transforms

import resize_images as R
from data_loader import ImagerLoader
from recipe_store import RecipeStore


def all_names(pipeline):
    return R.wanted_names(str(pipeline / "data"), all_images=True)


def make_source(path, i):
    """Images of assorted sizes and modes, like the real ones."""
    sizes = [(640, 480), (300, 900), (200, 150), (1200, 400)]
    modes = ["RGB", "RGB", "L", "RGBA"]
    w, h = sizes[i % 4]
    arr = np.random.default_rng(i).integers(0, 255, (h, w, 3), dtype=np.uint8)
    im = Image.fromarray(arr).convert(modes[i % 4])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if modes[i % 4] == "RGBA":
        im.save(path, format="PNG")  # a .jpg name holding PNG data
    else:
        im.save(path, format="JPEG")
    return (w, h)


@pytest.fixture()
def source(pipeline, tmp_path):
    """An extracted tree for every image name, minus one missing and one corrupt, plus an unwanted file."""
    names = sorted(all_names(pipeline))
    sizes = {}
    root = tmp_path / "src"
    missing, corrupt = names[3], names[7]
    for i, n in enumerate(names):
        if n == missing:
            continue
        path = R.layout_path(str(root), n)
        if n == corrupt:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").write(b"this is not an image")
            continue
        sizes[n] = make_source(path, i)
    extra = R.layout_path(str(root), "zzzz-not-in-any-recipe.jpg")
    make_source(extra, 1)
    return root, sizes, missing, corrupt


def make_tars(root, tmp_path, names):
    """Two tar files like the real ones: split by the first folder level, with a prefix folder and a directory entry."""
    tars = []
    for k, firsts in enumerate((set("0123"), None)):
        path = tmp_path / f"images_{k}.tar"
        with tarfile.open(path, "w") as tar:
            tar.add(
                str(root), arcname="recipe1M+_images", recursive=False
            )  # a directory member
            for dirpath, _, files in os.walk(root):
                for f in files:
                    rel = os.path.relpath(os.path.join(dirpath, f), root)
                    in_first = rel[0] in "0123"
                    if (firsts is not None) == in_first:
                        tar.add(
                            os.path.join(dirpath, f),
                            arcname=os.path.join("recipe1M+_images", rel),
                        )
        tars.append(str(path))
    return tars


def read(path):
    with Image.open(path) as im:
        return im.format, im.mode, im.size


def test_wanted_names_follow_what_the_loader_reads(pipeline):
    data = str(pipeline / "data")
    wanted = R.wanted_names(data)
    store = RecipeStore(data + "/train_store")
    for i in range(len(store)):
        assert (
            set(store.image_names(i)) <= wanted
        )  # training uses up to five images per recipe
    val = RecipeStore(data + "/val_store")
    multi = [i for i in range(len(val)) if len(val.image_names(i)) > 1]
    assert multi, "the synthetic data should have val recipes with several images"
    for i in multi:
        assert val.image_names(i)[0] in wanted and not (
            set(val.image_names(i)[1:]) & wanted
        )  # val/test read only the first
    assert R.wanted_names(data, all_images=True) > wanted
    assert R.layout_path("/x", "0fa8309c13.jpg") == "/x/0/f/a/8/0fa8309c13.jpg"


def test_shrink_never_enlarges_and_always_gives_rgb_jpeg(tmp_path):
    for i, (w, h) in enumerate([(640, 480), (300, 900), (200, 150), (1200, 400)]):
        p = tmp_path / f"{i}.jpg"
        make_source(str(p), i)
        fmt, mode, size = read(io.BytesIO(R.shrink(str(p))))
        assert (fmt, mode) == ("JPEG", "RGB")
        assert min(size) == min(256, min(w, h))
        assert abs(size[0] / size[1] - w / h) < 0.02  # aspect ratio kept
    with pytest.raises(Exception):
        R.shrink(b"not an image")


def check_outputs(pipeline, out, sizes, missing, corrupt):
    wanted = R.wanted_names(str(pipeline / "data"))
    present = {n for n in wanted if n in sizes}
    found = {n for n in present if os.path.exists(R.layout_path(str(out), n))}
    assert found == present  # everything wanted and readable
    for n in found:
        fmt, mode, size = read(R.layout_path(str(out), n))
        w, h = sizes[n]
        assert (fmt, mode) == ("JPEG", "RGB") and min(size) == min(256, min(w, h))
    assert not os.path.exists(
        R.layout_path(str(out), "zzzz-not-in-any-recipe.jpg")
    )  # unwanted files are not copied
    return wanted


def test_resizing_extracted_images(pipeline, source, tmp_path):
    root, sizes, missing, corrupt = source
    out = tmp_path / "out"
    wanted = R.wanted_names(str(pipeline / "data"))
    res = R.run_extracted(wanted, str(root), str(out), workers=2, log=lambda *a: None)
    check_outputs(pipeline, out, sizes, missing, corrupt)
    assert (
        res["missing"] == [missing]
        and res["failed"] == [corrupt]
        and res["ok"] == len(wanted) - 2
    )
    again = R.run_extracted(wanted, str(root), str(out), workers=2, log=lambda *a: None)
    assert again["ok"] == 0 and again["skipped"] == len(wanted) - 2  # restartable
    redo = R.run_extracted(
        wanted, str(root), str(out), overwrite=True, workers=2, log=lambda *a: None
    )
    assert redo["ok"] == len(wanted) - 2


def test_resizing_straight_from_tars_gives_the_same_pixels(pipeline, source, tmp_path):
    root, sizes, missing, corrupt = source
    wanted = R.wanted_names(str(pipeline / "data"))
    tars = make_tars(root, tmp_path, wanted)
    out_t, out_e = tmp_path / "from_tars", tmp_path / "from_files"
    res = R.run_tars(wanted, tars, str(out_t), workers=2, log=lambda *a: None)
    R.run_extracted(wanted, str(root), str(out_e), workers=2, log=lambda *a: None)
    check_outputs(pipeline, out_t, sizes, missing, corrupt)
    assert res["missing"] == [missing] and res["failed"] == [corrupt]
    for n in wanted - {missing, corrupt}:
        a, b = (
            np.asarray(Image.open(R.layout_path(str(o), n))) for o in (out_t, out_e)
        )
        assert np.array_equal(a, b)
    again = R.run_tars(wanted, tars, str(out_t), workers=2, log=lambda *a: None)
    assert again["ok"] == 0 and again["skipped"] == len(wanted) - 2


def test_in_place_shrinks_originals_once(pipeline, source):
    root, sizes, missing, corrupt = source
    wanted = R.wanted_names(str(pipeline / "data"))
    big = [n for n in wanted if n in sizes and min(sizes[n]) > 256]
    before = {n: os.path.getsize(R.layout_path(str(root), n)) for n in big}
    res = R.run_extracted(
        wanted, str(root), str(root), in_place=True, workers=2, log=lambda *a: None
    )
    assert res["failed"] == [corrupt] and res["missing"] == [missing]
    for n in big:
        assert min(read(R.layout_path(str(root), n))[2]) == 256
        assert os.path.getsize(R.layout_path(str(root), n)) < before[n]
    again = R.run_extracted(
        wanted, str(root), str(root), in_place=True, workers=2, log=lambda *a: None
    )
    assert again["ok"] == 0  # already small enough: nothing is re-encoded
    assert os.path.exists(
        R.layout_path(str(root), "zzzz-not-in-any-recipe.jpg")
    )  # unwanted files untouched


def test_command_line_and_the_loader_reads_every_resized_file(
    pipeline, source, tmp_path, capsys
):
    root, sizes, missing, corrupt = source
    out = tmp_path / "images256"
    # replace the two bad names so that the loader below can find every file it asks for
    for n in (missing, corrupt):
        make_source(R.layout_path(str(root), n), 0)
    R.main(
        [
            "extracted",
            "--src",
            str(root),
            "--data",
            str(pipeline / "data"),
            "--out",
            str(out),
            "--workers",
            "2",
        ]
    )
    assert "unreadable 0" in capsys.readouterr().out and not os.path.exists(
        out / "missing.txt"
    )

    def strict(path):  # no silent white image: any file the loader needs must exist
        return Image.open(path).convert("RGB")

    tf = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )
    for part in ("train", "val", "test"):
        ds = ImagerLoader(
            str(out),
            transform=tf,
            loader=strict,
            data_path=str(pipeline / "data"),
            partition=part,
        )
        for i in range(len(ds)):
            for _ in range(
                3 if part == "train" else 1
            ):  # train picks random images and random negatives
                assert ds[i][0][0].shape == (3, 224, 224)


def test_missing_and_failed_lists_are_written(pipeline, source, tmp_path, capsys):
    root, sizes, missing, corrupt = source
    out = tmp_path / "o"
    R.main(
        [
            "extracted",
            "--src",
            str(root),
            "--data",
            str(pipeline / "data"),
            "--out",
            str(out),
            "--workers",
            "2",
        ]
    )
    assert open(out / "missing.txt").read().split() == [missing]
    assert open(out / "failed.txt").read().split() == [corrupt]
    assert "missing 1" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        R.main(
            [
                "extracted",
                "--src",
                str(root),
                "--data",
                str(tmp_path / "nothing"),
                "--out",
                str(out),
            ]
        )


# ------------------------------------------------------------- several source folders and odd layouts
def split_roots(root, tmp_path, n=2):
    """Move the files of one tree into n trees by first folder level, like separately extracted tar files."""
    roots = [tmp_path / f"part{k}" for k in range(n)]
    for dirpath, _, files in os.walk(root):
        for f in files:
            rel = os.path.relpath(os.path.join(dirpath, f), root)
            dest = roots[ord(rel[0]) % n] / rel
            os.makedirs(dest.parent, exist_ok=True)
            os.replace(os.path.join(dirpath, f), dest)
    return [str(r) for r in roots]


def test_images_split_across_several_folders(pipeline, source, tmp_path):
    root, sizes, missing, corrupt = source
    wanted = R.wanted_names(str(pipeline / "data"))
    whole = tmp_path / "from_whole"
    R.run_extracted(wanted, str(root), str(whole), workers=2, log=lambda *a: None)
    roots = split_roots(root, tmp_path)
    assert all(os.path.isdir(r) for r in roots) and not os.path.exists(
        R.layout_path(str(root), sorted(sizes)[0])
    )
    out = tmp_path / "from_parts"
    res = R.run_extracted(wanted, roots, str(out), workers=2, log=lambda *a: None)
    assert (
        res["missing"] == [missing]
        and res["failed"] == [corrupt]
        and res["ok"] == len(wanted) - 2
    )
    for n in wanted - {missing, corrupt}:
        assert np.array_equal(
            np.asarray(Image.open(R.layout_path(str(whole), n))),
            np.asarray(Image.open(R.layout_path(str(out), n))),
        )
    # one root alone only finds its own share
    one = R.run_extracted(
        wanted, roots[:1], str(tmp_path / "one"), workers=2, log=lambda *a: None
    )
    assert 2 < len(one["missing"]) < len(wanted)


def test_scan_finds_files_in_an_unexpected_layout(pipeline, source, tmp_path):
    root, sizes, missing, corrupt = source
    wanted = R.wanted_names(str(pipeline / "data"))
    odd = tmp_path / "odd"
    for k, n in enumerate(
        sorted(sizes)
    ):  # a wrapper folder and arbitrary sub-folders, nothing like 0/f/a/8/
        dest = odd / "Recipe1M+_extra" / f"batch{k % 3}" / n
        os.makedirs(dest.parent, exist_ok=True)
        os.replace(R.layout_path(str(root), n), dest)
    plain = R.run_extracted(
        wanted, [str(odd)], str(tmp_path / "plain"), workers=2, log=lambda *a: None
    )
    assert len(plain["missing"]) == len(wanted)  # the expected layout is not there
    logs = []
    res = R.run_extracted(
        wanted,
        [str(odd)],
        str(tmp_path / "scanned"),
        workers=2,
        scan=True,
        log=logs.append,
    )
    # the corrupt file was not moved into the odd layout, so like the never-created one it is simply not found
    assert (
        res["missing"] == sorted([missing, corrupt])
        and res["ok"] == len(wanted) - 2
        and res["failed"] == []
    )
    assert any("scan:" in m for m in logs)
    assert R.scan_roots(
        [str(odd)], frozenset({sorted(sizes)[0], "nothing-here.jpg"})
    ).keys() == {sorted(sizes)[0]}


def test_command_line_with_several_src_folders_and_in_place(
    pipeline, source, tmp_path, capsys
):
    root, sizes, missing, corrupt = source
    roots = split_roots(root, tmp_path)
    out = tmp_path / "cli"
    R.main(
        [
            "extracted",
            "--src",
            *roots,
            "--data",
            str(pipeline / "data"),
            "--out",
            str(out),
            "--workers",
            "2",
        ]
    )
    assert "missing 1" in capsys.readouterr().out and open(
        out / "missing.txt"
    ).read().split() == [missing]
    wanted = R.wanted_names(str(pipeline / "data"))
    big = [n for n in wanted if n in sizes and min(sizes[n]) > 256]
    R.main(
        [
            "extracted",
            "--src",
            *roots,
            "--data",
            str(pipeline / "data"),
            "--in-place",
            "--workers",
            "2",
        ]
    )
    found_in = {
        n: next(r for r in roots if os.path.exists(R.layout_path(r, n))) for n in big
    }
    assert (
        len(set(found_in.values())) == 2
    )  # the files stayed in whichever folder they were in
    assert all(min(read(R.layout_path(r, n))[2]) == 256 for n, r in found_in.items())


def test_missing_report_tells_missing_folders_from_an_even_gap(tmp_path):
    chars = "0123456789abcdef"
    wanted = {f"{c}{i:03d}abcdef.jpg" for c in chars for i in range(20)}
    # case 1: three whole top-level folders were never extracted
    gone = {n for n in wanted if n[0] in "cde"}
    root = tmp_path / "src"
    for c in chars:
        if c not in "cde":
            (root / c).mkdir(parents=True)
    text = "\n".join(R.missing_report(sorted(gone), wanted, [str(root)]))
    assert "c: 20/20 (100%)" in text and "0: 0/20 (0%)" in text
    assert "no folder named c, d, e" in text and "whole first-level folders" in text
    # case 2: every folder is there but about a fifth of the images are not
    spread = {n for i, n in enumerate(sorted(wanted)) if i % 5 == 0}
    for c in chars:
        (root / c).mkdir(parents=True, exist_ok=True)
    text = "\n".join(R.missing_report(sorted(spread), wanted, [str(root)]))
    assert (
        "no folder named" not in text
        and "spread evenly" in text
        and "layer2+.json" in text
    )
    # nothing missing: no verdict
    assert not any(
        l.startswith("=>") for l in R.missing_report([], wanted, [str(root)])
    )
