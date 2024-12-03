from dataclasses import dataclass
from pathlib import Path

from ..hparams import HParams as HParamsBase


@dataclass(frozen=True)
class HParams(HParamsBase):
    cfm_solver_method: str = "midpoint"
    cfm_solver_nfe: int = 64
    cfm_time_mapping_divisor: int = 4
    univnet_nc: int = 96

    lcfm_latent_dim: int = 64
    lcfm_training_mode: str = "ae"
    # This value should be carefully tuned when training. Better estimate it from the latent vectors first
    lcfm_z_scale: float = 5

    vocoder_extra_dim: int = 32

    gan_training_start_step: int | None = 5_000
    enhancer_stage1_run_dir: Path | None = None

    denoiser_run_dir: Path | None = None

    # Enable this increases the training stability (but will also disable the change of eval_tau)
    force_gaussian_prior: bool = False
