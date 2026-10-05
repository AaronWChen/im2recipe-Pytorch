"""Flat-array recipe store (replaces the per-recipe pickled LMDB).

One directory per partition, e.g. data/train_store/:

    meta.json          format version, sizes, dtype, maxlen
    intrs.npy          (total_instructions, dim) float32|float16, memory-mapped when read
    intr_offsets.npy   (n + 1,) int64; recipe i's vectors are intrs[off[i]:off[i+1]]
    ingrs.npy          (n, maxlen) uint16 ingredient ids, zero padded
    classes.npy        (n,) int32 class index
    ids.npy            (n,) fixed-width bytes, recipe ids
    img_names.npy      flat fixed-width bytes, image file names of all recipes
    img_offsets.npy    (n + 1,) int64; recipe i's images are img_names[off[i]:off[i+1]]

Everything except intrs.npy is small and loaded fully into memory; intrs.npy is opened
lazily (per process), so a DataLoader worker maps its own copy. No pickle anywhere.
"""
import json
import os

import numpy as np

FORMAT = "im2recipe-npy-v1"


def _bytes_array(strings):
    arr = np.array([s.encode("utf-8") for s in strings])
    return arr if len(strings) else np.zeros(0, dtype="S1")


def write_store(out_dir, ids, ingrs, classes, images, source, spans, dtype=None, extra_meta=None):
    """Write one partition.

    ids      list[str], one per recipe
    ingrs    (n, maxlen) integer array
    classes  length-n integer sequence
    images   list[list[str]] image file names per recipe
    source   (rows, dim) array (a memmap is fine) holding the instruction vectors
    spans    list[(start, length)] into `source`, one per recipe. Vectors are copied
             into a compact contiguous file, so recipes dropped by a filter cost nothing.
    dtype    storage dtype for intrs.npy (default: the source dtype)
    """
    n = len(ids)
    if not (len(ingrs) == len(classes) == len(images) == len(spans) == n):
        raise ValueError("ids, ingrs, classes, images and spans must all have one entry per recipe")
    os.makedirs(out_dir, exist_ok=True)
    ingrs = np.asarray(ingrs, dtype=np.uint16)
    if ingrs.ndim != 2:  # an empty partition arrives as a bare empty array
        ingrs = ingrs.reshape(n, 0) if n == 0 else ingrs.reshape(n, -1)
    dtype = np.dtype(dtype or source.dtype)
    dim = source.shape[1]

    lens = np.array([length for _, length in spans], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    out = np.lib.format.open_memmap(os.path.join(out_dir, "intrs.npy"), mode="w+", dtype=dtype,
                                    shape=(int(offsets[-1]), dim))
    for i, (start, length) in enumerate(spans):
        out[offsets[i]:offsets[i + 1]] = source[start:start + length]
    out.flush()
    del out

    img_lens = np.array([len(x) for x in images], dtype=np.int64)
    img_offsets = np.concatenate([[0], np.cumsum(img_lens)]).astype(np.int64)
    np.save(os.path.join(out_dir, "intr_offsets.npy"), offsets)
    np.save(os.path.join(out_dir, "img_offsets.npy"), img_offsets)
    np.save(os.path.join(out_dir, "img_names.npy"), _bytes_array([name for names in images for name in names]))
    np.save(os.path.join(out_dir, "ingrs.npy"), ingrs)
    np.save(os.path.join(out_dir, "classes.npy"), np.asarray(classes, dtype=np.int32))
    np.save(os.path.join(out_dir, "ids.npy"), _bytes_array(ids))
    meta = {"format": FORMAT, "n_recipes": n, "n_instructions": int(offsets[-1]), "dim": int(dim),
            "dtype": dtype.name, "maxlen": int(ingrs.shape[1])}
    meta.update(extra_meta or {})
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


class RecipeStore:
    def __init__(self, path):
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        if self.meta.get("format") != FORMAT:
            raise ValueError(f"{path}: unsupported store format {self.meta.get('format')!r}")
        self.path = path
        load = lambda name: np.load(os.path.join(path, name + ".npy"))
        self._ids, self._ingrs, self._classes = load("ids"), load("ingrs"), load("classes")
        self._intr_off, self._img_off, self._img_names = load("intr_offsets"), load("img_offsets"), load("img_names")
        self._intrs = None  # opened lazily so each worker process maps its own copy

    def __len__(self):
        return len(self._ids)

    @property
    def dim(self):
        return self.meta["dim"]

    def _vectors(self):
        if self._intrs is None:
            self._intrs = np.load(os.path.join(self.path, "intrs.npy"), mmap_mode="r")
        return self._intrs

    def recipe_id(self, i):
        return self._ids[i].decode("utf-8")

    def instructions(self, i):
        """(k, dim) float32 copy of recipe i's instruction vectors."""
        return np.array(self._vectors()[self._intr_off[i]:self._intr_off[i + 1]], dtype=np.float32)

    def ingredients(self, i):
        return self._ingrs[i]

    def recipe_class(self, i):
        return int(self._classes[i])

    def image_names(self, i):
        return [b.decode("utf-8") for b in self._img_names[self._img_off[i]:self._img_off[i + 1]]]
