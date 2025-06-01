import logging
from functools import cache
from pathlib import Path

import torch

from ..inference import inference
from .download import download
from .safetensors_loader import load_enhancer_model
from .hparams import HParams

logger = logging.getLogger(__name__)


@cache
def load_enhancer(run_dir: str | Path | None, device, skip_download: bool = False):
    if not skip_download:
        run_dir = download(run_dir)
    return load_enhancer_model(run_dir, device)


@torch.inference_mode()
def denoise(dwav, sr, device, run_dir=None):
    from ..denoiser.safetensors_loader import load_denoiser_from_enhancer_checkpoint
    from ..inference import inference
    
    # Load denoiser from enhancer checkpoint since they're stored together
    denoiser = load_denoiser_from_enhancer_checkpoint(run_dir, device)
    return inference(model=denoiser, dwav=dwav, sr=sr, device=device)


@torch.inference_mode()
def enhance(dwav, sr, device, nfe=32, solver="midpoint", lambd=0.5, tau=0.5, run_dir=None, skip_download=False):
    assert 0 < nfe <= 128, f"nfe must be in (0, 128], got {nfe}"
    assert solver in ("midpoint", "rk4", "euler"), f"solver must be in ('midpoint', 'rk4', 'euler'), got {solver}"
    assert 0 <= lambd <= 1, f"lambd must be in [0, 1], got {lambd}"
    assert 0 <= tau <= 1, f"tau must be in [0, 1], got {tau}"
    enhancer = load_enhancer(run_dir, device, skip_download)
    enhancer.configurate_(nfe=nfe, solver=solver, lambd=lambd, tau=tau)
    return inference(model=enhancer, dwav=dwav, sr=sr, device=device)
