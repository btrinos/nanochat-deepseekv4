"""Minimal runtime helpers for the DeepSeek-V4 benchmark."""

import os

import torch


_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _detect_compute_dtype():
    env = os.environ.get("NANOCHAT_DTYPE")
    if env is not None:
        return _DTYPE_MAP[env], f"set via NANOCHAT_DTYPE={env}"
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16, f"auto-detected: CUDA SM {capability[0]}{capability[1]}"
    return torch.float32, "auto-detected: CPU/MPS or pre-Ampere CUDA"


COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()


def print0(message="", **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(message, **kwargs)


def autodetect_device_type():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

