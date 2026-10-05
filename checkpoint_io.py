"""Checkpoint loading that does not execute arbitrary pickled code by default."""
import pickle

import torch


def load_checkpoint(path, device, trust=False):
    """Load a checkpoint written by train.py (a dict with tensors and plain numbers).

    By default only tensors and plain Python values are accepted (torch.load weights_only=True),
    so a downloaded file cannot run code on load. Checkpoints that contain anything else, such as
    the numpy scalars older versions of train.py stored, need trust=True (--trust_checkpoint),
    which unpickles with the old latin1 behaviour. Only use that for files you trust.
    """
    map_location = 'cpu' if device.type == 'cpu' else None
    if trust:
        return torch.load(path, map_location=map_location, encoding='latin1', weights_only=False)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as e:
        raise RuntimeError(
            "%s contains objects other than tensors and plain numbers (older checkpoints store numpy "
            "scalars). If you trust this file, load it with --trust_checkpoint." % path) from e
