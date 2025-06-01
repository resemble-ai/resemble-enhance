import logging
from functools import cache

import torch

from ..inference import inference
from .safetensors_loader import load_denoiser_model, create_default_denoiser
from .hparams import HParams

logger = logging.getLogger(__name__)


@cache
def load_denoiser(run_dir, device):
    if run_dir is None:
        return create_default_denoiser(device)
    return load_denoiser_model(run_dir, device)


@torch.inference_mode()
def denoise(dwav, sr, run_dir, device):
    denoiser = load_denoiser(run_dir, device)
    return inference(model=denoiser, dwav=dwav, sr=sr, device=device)
