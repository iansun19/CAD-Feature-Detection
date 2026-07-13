"""Shared device selection and seeding for train.py / overfit_check.py."""

import random

import numpy as np
import torch


def resolve_device(preference="auto") -> torch.device:
    """Pick cuda, mps, or cpu. preference: auto | cuda | mps | cpu."""
    pref = (preference or "auto").lower()
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
