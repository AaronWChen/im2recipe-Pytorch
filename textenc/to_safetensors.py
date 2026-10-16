"""Save a Hugging Face model as safetensors (e.g. RecipeNLG's 2020 checkpoint, which is a pickled .bin).

    python -m textenc.to_safetensors mbien/recipenlg --out models/recipenlg --allow-pickle

Loads the model with its original architecture (GPT2LMHeadModel stays GPT2LMHeadModel, so it can still generate and
be fine-tuned), saves weights + config + tokenizer to --out, then reloads from safetensors only and checks every
tensor is identical. --allow-pickle is only needed when the source has no safetensors file; use it for models you trust.
Models you fine-tune later with save_pretrained are written as safetensors already.
"""
import argparse
import os

import torch


def convert(src, out, allow_pickle=False):
    import transformers
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    cfg = AutoConfig.from_pretrained(src)
    archs = getattr(cfg, "architectures", None) or []
    cls = getattr(transformers, archs[0], None) if archs else None
    cls = cls or AutoModel
    try:
        model = cls.from_pretrained(src, use_safetensors=True)
    except OSError:
        if not allow_pickle:
            raise OSError(f"{src} has no safetensors weights; add --allow-pickle to load its pickled weights "
                          "(only if you trust the source)")
        model = cls.from_pretrained(src)
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(src).save_pretrained(out)
    if not any(f.endswith(".safetensors") for f in os.listdir(out)):
        raise RuntimeError(f"no .safetensors file was written to {out}")

    again = cls.from_pretrained(out, use_safetensors=True)
    a, b = model.state_dict(), again.state_dict()
    if a.keys() != b.keys() or not all(torch.equal(a[k], b[k]) for k in a):
        raise RuntimeError("the converted weights differ from the originals")
    return cls.__name__


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="Hugging Face id or local folder")
    p.add_argument("--out", required=True)
    p.add_argument("--allow-pickle", action="store_true")
    a = p.parse_args(argv)
    print(f"saved {convert(a.model, a.out, a.allow_pickle)} as safetensors in {a.out} (verified identical)")


if __name__ == "__main__":
    main()
