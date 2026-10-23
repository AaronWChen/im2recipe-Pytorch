"""Convert a skipinstructions .pt checkpoint to .safetensors (weights + config), e.g. for a model trained before this existed.

    python -m skipinstructions.export runs/skip1/skipinstructions-best.pt          # -> runs/skip1/skipinstructions-best.safetensors
"""
import argparse
import os

import torch

from safetensors_io import save_safetensors


def export(pt_path, out_path=None):
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
    out_path = out_path or os.path.splitext(pt_path)[0] + ".safetensors"
    save_safetensors(out_path, ckpt["model"], {"format": "skipinstructions-v1", "config": ckpt["config"],
                                               "iter": ckpt.get("iter"), "best": ckpt.get("best")})
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    print("wrote", export(a.checkpoint, a.out))


if __name__ == "__main__":
    main()
