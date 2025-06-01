import logging
from pathlib import Path

import torch

RUN_NAME = "enhancer_stage2"

logger = logging.getLogger(__name__)


def get_source_url(relpath):
    return f"https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/{RUN_NAME}/{relpath}?download=true"

def get_safetensors_url(relpath):
    return f"https://huggingface.co/rsxdalv/resemble-enhance/resolve/main/{relpath}?download=true"

def get_target_path(relpath: str | Path, run_dir: str | Path | None = None):
    if run_dir is None:
        run_dir = Path(__file__).parent.parent / "model_repo" / RUN_NAME
    return Path(run_dir) / relpath


def download(run_dir: str | Path | None = None, safetensors: bool = False) -> Path:
    relpaths_safetensors = [
        "denoiser/config.json",
        "denoiser/model.safetensors",
        "denoiser/model_info.json",
        "enhancer/config.json",
        "enhancer/model.safetensors",
        "enhancer/model_info.json",
    ]
    if safetensors:
        for relpath in relpaths_safetensors:
            path = get_target_path(relpath, run_dir=run_dir)
            if path.exists():
                continue
            url = get_safetensors_url(relpath)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(url, str(path))
        return get_target_path("", run_dir=run_dir)

    relpaths = ["hparams.yaml", "ds/G/latest", "ds/G/default/mp_rank_00_model_states.pt"]
    for relpath in relpaths:
        path = get_target_path(relpath, run_dir=run_dir)
        if path.exists():
            continue
        url = get_source_url(relpath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(path))
    return get_target_path("", run_dir=run_dir)
