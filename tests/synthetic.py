"""Tiny synthetic Recipe1M used by the tests."""
import json
import os
import pickle

import numpy as np
from PIL import Image

DIM = 24
STEPS = ["chop the onion", "heat the olive oil", "add garlic and stir", "cook for ten minutes",
         "season with salt", "serve warm"]
INGREDIENTS = ["onion", "olive oil", "garlic", "salt"]


def make_dataset(root, n=48):
    """Recipes r0..r{n-1}: partitions 60/20/20. Some have no images, too many instructions, etc."""
    layer1, layer2, det, classes = [], [], [], {}
    for i in range(n):
        part = "train" if i < int(n * 0.6) else ("val" if i < int(n * 0.8) else "test")
        k = 3 + i % 4
        if i == 5:
            k = 25  # too many instructions -> filtered
        instrs = [{"text": STEPS[j % len(STEPS)] + " ."} for j in range(k)]
        layer1.append({"id": f"r{i}", "partition": part, "title": "garlic onion stew", "instructions": instrs})
        if i != 4:  # r4 has no images -> filtered
            layer2.append({"id": f"r{i}", "images": [{"id": f"{i:02d}abcdef{j}.jpg"} for j in range(1 + i % 7)]})
        det.append({"id": f"r{i}", "ingredients": [{"text": t} for t in INGREDIENTS[: 1 + i % 4]],
                    "valid": [True] * (1 + i % 4)})
        classes[f"r{i}"] = i % 5
    for name, obj in (("layer1", layer1), ("layer2", layer2), ("det_ingrs", det)):
        with open(os.path.join(root, name + ".json"), "w") as f:
            json.dump(obj, f)
    with open(os.path.join(root, "vocab.txt"), "w") as f:
        f.write("\n".join(t.replace(" ", "_") for t in INGREDIENTS) + "\n")
    with open(os.path.join(root, "classes.pkl"), "wb") as f:
        pickle.dump(classes, f)
        pickle.dump({v: k for k, v in classes.items()}, f)
    with open(os.path.join(root, "remove.txt"), "w") as f:
        f.write("r0\nr1\n")  # r0 is the FIRST entry: the old dict-based lookup never removed it


def write_w2v(path, words, dim, seed=0):
    """A word2vec .bin file in the C layout (header, then word + space + float32s + newline)."""
    vecs = np.random.default_rng(seed).standard_normal((len(words), dim)).astype("<f4")
    with open(path, "wb") as f:
        f.write(f"{len(words)} {dim}\n".encode())
        for w, v in zip(words, vecs):
            f.write(w.encode() + b" " + v.tobytes() + b"\n")
    return vecs


def make_images(img_root, stores):
    """One small jpg per image name, in the four-level folder layout the loader expects."""
    for s in stores:
        for i in range(len(s)):
            for name in s.image_names(i):
                d = os.path.join(img_root, *name[:4])
                os.makedirs(d, exist_ok=True)
                Image.new("RGB", (40, 30), (i * 5 % 255, 80, 120)).save(os.path.join(d, name))


if __name__ == "__main__":
    # python tests/synthetic.py DIR  -> DIR/layer1.json, layer2.json, det_ingrs.json (+ helper files), for dry runs
    import sys
    os.makedirs(sys.argv[1], exist_ok=True)
    make_dataset(sys.argv[1])
