"""Save and load model weights as safetensors, with JSON metadata.

safetensors files hold only tensors (no pickle, so loading cannot run code) plus a string-to-string
metadata table. Every metadata value is stored JSON-encoded, so numbers, booleans, lists and dicts
survive a round trip: save_safetensors(p, sd, {"epoch": 3, "cfg": {"dim": 8}}) ->
load_safetensors(p) == (sd, {"epoch": 3, "cfg": {"dim": 8}}).
"""
import json

from safetensors import safe_open
from safetensors.torch import save_file


def save_safetensors(path, state_dict, metadata=None):
    # clone: safetensors refuses tensors that share memory (tied weights) or are non-contiguous views
    tensors = {k: v.detach().cpu().clone().contiguous() for k, v in state_dict.items()}
    meta = {str(k): json.dumps(v, default=str) for k, v in (metadata or {}).items()}
    save_file(tensors, path, metadata=meta)


def load_safetensors(path):
    """Return (state_dict on cpu, metadata dict)."""
    with safe_open(path, framework="pt", device="cpu") as f:
        raw = f.metadata() or {}
        state = {k: f.get_tensor(k) for k in f.keys()}
    meta = {}
    for k, v in raw.items():
        try:
            meta[k] = json.loads(v)
        except ValueError:  # written by another tool, e.g. {"format": "pt"}
            meta[k] = v
    return state, meta
