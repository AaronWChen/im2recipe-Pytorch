#!/usr/bin/env python
"""Shrink the Recipe1M images the model will actually read, so they take tens of GB instead of hundreds.

The loader (data_loader.py) resizes every image to a shorter side of 256 and crops 224, and it only reads the
first five images of each training recipe and the first image of each val/test recipe. This keeps exactly those
images (names come from the <data>/*_store folders written by build_dataset.py), with the shorter side reduced to
--size (never enlarged), as JPEG, in the same four-level folder layout, so the output folder can be given to
train.py as --img_path unchanged.

From the tar files, without extracting them (only the wanted files are read; one process per tar file):

    python scripts/resize_images.py tars --tars /downloads/Recipe1M+_*.tar --data data --out data/images256

From an extracted folder tree (<src>/0/f/a/8/0fa8309c13.jpg):

    python scripts/resize_images.py extracted --src /data/recipe1M+_images --data data --out data/images256
    python scripts/resize_images.py extracted --src /data/recipe1M+_images --data data --in-place   # replaces the originals

Runs can be stopped and restarted: files that already exist in --out are skipped (use --overwrite to redo them).
Names that were wanted but not found are written to <out>/missing.txt, unreadable images to <out>/failed.txt.
--in-place overwrites the originals of the wanted images only (other files are left alone); a file already at or below
--size is left as it is, so the command is safe to repeat. Test on a copy first: the originals are not recoverable.
"""
import argparse
import glob
import io
import multiprocessing
import os
import sys
import tarfile

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recipe_store import RecipeStore  # noqa: E402

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")


# ------------------------------------------------------------------------------ what to keep
def wanted_names(data_dir, all_images=False):
    """Image file names the loader can read: every image of a train recipe, the first image of a val/test recipe."""
    names = set()
    for part in ("train", "val", "test"):
        path = os.path.join(data_dir, part + "_store")
        if not os.path.isdir(path):
            continue
        store = RecipeStore(path)
        for i in range(len(store)):
            images = store.image_names(i)
            names.update(images if (part == "train" or all_images) else images[:1])
    return names


def layout_path(root, name):
    """0fa8309c13.jpg -> <root>/0/f/a/8/0fa8309c13.jpg, the layout the loader expects."""
    return os.path.join(root, *name[:4], name)


# --------------------------------------------------------------------------------- resizing
def shrink(source, size=256, quality=90):
    """Image file path or bytes -> JPEG bytes whose shorter side is at most `size`. Raises on unreadable images."""
    with Image.open(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source) as im:
        w, h = im.size
        scale = size / min(w, h)
        if scale < 1:
            new = (max(1, round(w * scale)), max(1, round(h * scale)))
            im.draft("RGB", new)  # for JPEGs: decode at reduced scale, much faster
            im = im.convert("RGB").resize(new, Image.LANCZOS)
        else:
            im = im.convert("RGB")
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=quality)
        return out.getvalue()


def already_small(path, size):
    try:
        with Image.open(path) as im:
            return min(im.size) <= size and im.format == "JPEG"
    except Exception:
        return False


def write_atomic(dest, payload):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".tmp"
    with open(tmp, "wb") as f:
        f.write(payload)
    os.replace(tmp, dest)


# ----------------------------------------------------------------------- from extracted files
def _extracted_task(args):
    name, src, out, size, quality, overwrite, in_place = args
    source = layout_path(src, name)
    if not os.path.exists(source):
        return name, "missing"
    dest = layout_path(out, name)
    try:
        if in_place:
            if already_small(source, size):
                return name, "skipped"
        elif os.path.exists(dest) and not overwrite:
            return name, "skipped"
        write_atomic(dest, shrink(source, size, quality))
        return name, "ok"
    except Exception:
        return name, "failed"


def run_extracted(names, src, out, size=256, quality=90, overwrite=False, in_place=False, workers=None, log=print):
    tasks = ((n, src, out, size, quality, overwrite, in_place) for n in sorted(names))
    results = {"ok": 0, "skipped": 0, "missing": [], "failed": []}
    done = 0
    with multiprocessing.Pool(workers or os.cpu_count() or 1) as pool:
        for name, status in pool.imap_unordered(_extracted_task, tasks, chunksize=64):
            if status in ("missing", "failed"):
                results[status].append(name)
            else:
                results[status] += 1
            done += 1
            if done % 20000 == 0:
                log(f"{done}/{len(names)} images handled")
    return results


# ------------------------------------------------------------------------------- from tar files
def _tar_task(args):
    path, wanted, out, size, quality, overwrite = args
    seen, ok, skipped, failed = set(), 0, 0, []
    with tarfile.open(path) as tar:
        for member in tar:  # lazy: a seekable tar skips over the data of files we do not want
            tar.members = []  # do not keep millions of headers in memory
            if not member.isfile():
                continue
            name = os.path.basename(member.name)
            if name not in wanted:
                continue
            seen.add(name)
            dest = layout_path(out, name)
            if os.path.exists(dest) and not overwrite:
                skipped += 1
                continue
            try:
                write_atomic(dest, shrink(tar.extractfile(member).read(), size, quality))
                ok += 1
            except Exception:
                failed.append(name)
    return path, seen, ok, skipped, failed


def run_tars(names, tars, out, size=256, quality=90, overwrite=False, workers=None, log=print):
    wanted = frozenset(names)
    tasks = [(t, wanted, out, size, quality, overwrite) for t in tars]
    results = {"ok": 0, "skipped": 0, "missing": [], "failed": []}
    seen = set()
    with multiprocessing.Pool(min(workers or os.cpu_count() or 1, max(1, len(tars)))) as pool:
        for path, s, ok, skipped, failed in pool.imap_unordered(_tar_task, tasks):
            seen |= s
            results["ok"] += ok
            results["skipped"] += skipped
            results["failed"] += failed
            log(f"{os.path.basename(path)}: {ok} written, {skipped} already there, {len(failed)} unreadable")
    results["missing"] = sorted(wanted - seen)
    return results


# ------------------------------------------------------------------------------------- command line
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", default="data", help="folder with train_store/ val_store/ test_store/")
    common.add_argument("--size", type=int, default=256, help="shorter side in pixels (default: what the loader resizes to)")
    common.add_argument("--quality", type=int, default=90, help="JPEG quality")
    common.add_argument("--all-images", action="store_true", help="also keep the non-first images of val/test recipes")
    common.add_argument("--overwrite", action="store_true")
    common.add_argument("--workers", type=int, default=None)
    t = sub.add_parser("tars", parents=[common], help="read straight from the tar files")
    t.add_argument("--tars", nargs="+", required=True, help="tar files (a shell glob such as Recipe1M+_*.tar works)")
    t.add_argument("--out", required=True)
    e = sub.add_parser("extracted", parents=[common], help="read from an extracted folder tree")
    e.add_argument("--src", required=True)
    group = e.add_mutually_exclusive_group(required=True)
    group.add_argument("--out")
    group.add_argument("--in-place", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    names = wanted_names(a.data, a.all_images)
    if not names:
        sys.exit(f"no image names found: is {a.data}/train_store there? (run scripts/build_dataset.py first)")
    print(f"{len(names)} images wanted")
    if a.mode == "tars":
        tars = [t for pattern in a.tars for t in (sorted(glob.glob(pattern)) or [pattern])]
        out = a.out
        res = run_tars(names, tars, out, a.size, a.quality, a.overwrite, a.workers)
    else:
        out = a.src if a.in_place else a.out
        res = run_extracted(names, a.src, out, a.size, a.quality, a.overwrite, a.in_place, a.workers)
    os.makedirs(out, exist_ok=True)
    for key in ("missing", "failed"):
        if res[key]:
            with open(os.path.join(out, key + ".txt"), "w") as f:
                f.write("\n".join(sorted(res[key])) + "\n")
    print(f"written {res['ok']}, already done {res['skipped']}, missing {len(res['missing'])}, unreadable {len(res['failed'])}")
    if res["missing"] or res["failed"]:
        print(f"see {out}/missing.txt and {out}/failed.txt; the loader replaces an image it cannot open with a white one")


if __name__ == "__main__":
    main()
