"""
Enhancer model for inference without training dependencies.
This is a copy of the enhancer module but with inference-only imports.
"""
import logging

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import Tensor, nn
from torch.distributions import Beta

from ..common import Normalizer
from ..melspec import MelSpectrogram
from .hparams import HParams
from .lcfm import CFM, IRMAE, LCFM
from .univnet import UnivNet

logger = logging.getLogger(__name__)


# No-op decorators for inference
def global_leader_only(fn):
    """No-op decorator for inference - just returns the function as-is."""
    return fn


# Simple TrainLoop replacement for inference
class TrainLoop:
    """No-op TrainLoop class for inference."""
    
    @staticmethod
    def get_running_loop():
        """Always return None for inference (no training loop)."""
        return None


def _maybe(fn):
    def _fn(*args):
        if args[0] is None:
            return None
        return fn(*args)

    return _fn


def _normalize_wav(x: Tensor):
    return x / (x.abs().max(dim=-1, keepdim=True).values + 1e-7)


class EnhancerInference(nn.Module):
    def __init__(self, hp: HParams):
        super().__init__()
        self.hp = hp

        n_mels = self.hp.num_mels
        vocoder_input_dim = n_mels + self.hp.vocoder_extra_dim
        latent_dim = self.hp.lcfm_latent_dim

        self.lcfm = LCFM(
            IRMAE(
                input_dim=n_mels,
                output_dim=vocoder_input_dim,
                latent_dim=latent_dim,
            ),
            CFM(
                cond_dim=n_mels,
                output_dim=self.hp.lcfm_latent_dim,
                solver_nfe=hp.cfm_solver_nfe,
                solver_method=hp.cfm_solver_method,
                time_mapping_divisor=hp.cfm_time_mapping_divisor,
            ),
            z_scale=self.hp.lcfm_z_scale,
        )

        self.lcfm.set_mode_(self.hp.lcfm_training_mode)

        self.mel_fn = MelSpectrogram(hp)
        self.vocoder = UnivNet(self.hp, vocoder_input_dim)
        
        # For inference, denoiser will be set separately if needed
        self.denoiser = None
        self.normalizer = Normalizer()

        self._eval_lambd = 0.0

        self.dummy: Tensor
        self.register_buffer("dummy", torch.zeros(1))

    @property
    def mel_fn(self):
        return self.vocoder.mel_fn

    def configure_denoiser_(self, denoiser):
        self.denoiser = denoiser

    def configurate_(self, **kwargs):
        """Configure model parameters for inference.
        
        Args:
            nfe: number of function evaluations
            solver: solver method
            lambd: denoiser strength [0, 1]
            tau: prior temperature [0, 1]
        """
        if "nfe" in kwargs and "solver" in kwargs:
            self.lcfm.cfm.solver.configurate_(kwargs["nfe"], kwargs["solver"])
        elif "nfe" in kwargs:
            self.lcfm.cfm.solver.configurate_(kwargs["nfe"], None)
        elif "solver" in kwargs:
            self.lcfm.cfm.solver.configurate_(None, kwargs["solver"])
            
        if "tau" in kwargs:
            self.lcfm.eval_tau_(kwargs["tau"])
        if "lambd" in kwargs:
            self._eval_lambd = kwargs["lambd"]

    def _may_denoise(self, x: Tensor, y: Tensor | None = None):
        if self.hp.lcfm_training_mode == "cfm" and self.denoiser is not None:
            return self.denoiser(x, y)
        return x

    def forward(self, x: Tensor, y: Tensor | None = None, z: Tensor | None = None):
        """Forward pass for inference.
        
        Args:
            x: (b t), mix wavs (fg + bg)
            y: (b t), fg clean wavs
            z: (b t), fg distorted wavs
        Returns:
            o: (b t), reconstructed wavs
        """
        assert x.dim() == 2, f"Expected (b t), got {x.size()}"
        assert y is None or y.dim() == 2, f"Expected (b t), got {y.size()}"

        if self.hp.lcfm_training_mode == "cfm":
            self.normalizer.eval()

        x = _normalize_wav(x)
        y = _maybe(_normalize_wav)(y)
        z = _maybe(_normalize_wav)(z)

        x_mel_original = self.normalizer(self.to_mel(x), update=False)  # (b d t)

        if self.hp.lcfm_training_mode == "cfm":
            lambd = self._eval_lambd
            if lambd == 0:
                x_mel_denoised = x_mel_original
            else:
                x_mel_denoised = self.normalizer(self.to_mel(self._may_denoise(x, z)), update=False)
                x_mel_denoised = x_mel_denoised.detach()
                x_mel_denoised = lambd * x_mel_denoised + (1 - lambd) * x_mel_original
        else:
            x_mel_denoised = x_mel_original

        y_mel = _maybe(self.to_mel)(y)  # (b d t)
        y_mel = _maybe(self.normalizer)(y_mel)

        if hasattr(self.hp, 'force_gaussian_prior') and self.hp.force_gaussian_prior:
            lcfm_decoded = self.lcfm(x_mel_denoised, y_mel, ψ0=None)  # (b d t)
        else:
            lcfm_decoded = self.lcfm(x_mel_denoised, y_mel, ψ0=x_mel_original)  # (b d t)

        if lcfm_decoded is None:
            o = None
        else:
            o = self.vocoder(lcfm_decoded, y)

        return o

    def to_mel(self, x, drop_last=True):
        """Convert waveform to mel-spectrogram.
        
        Args:
            x: (b t), wavs
        Returns:
            o: (b c t), mels
        """
        if drop_last:
            return self.mel_fn(x)[..., :-1]  # (b d t)
        return self.mel_fn(x)

    def to_mel(self, x, drop_last=True):
        if drop_last:
            return self.mel_fn(x)[..., :-1]  # (b d t)
        return self.mel_fn(x)

    @global_leader_only
    @torch.no_grad()
    def _visualize(self, original_mel, denoised_mel):
        loop = TrainLoop.get_running_loop()
        if loop is None or loop.global_step % 100 != 0:
            return

        plt.figure(figsize=(6, 6))
        plt.subplot(211)
        plt.title("Original")
        plt.imshow(original_mel[0].cpu().numpy(), origin="lower", interpolation="none")
        plt.subplot(212)
        plt.title("Denoised")
        plt.imshow(denoised_mel[0].cpu().numpy(), origin="lower", interpolation="none")

        loop.save_current_step_viz(plt.gcf(), self.__class__.__name__, ".png")
        plt.close()
