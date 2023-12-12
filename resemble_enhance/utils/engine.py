import logging
import re
from functools import cache, partial
from typing import Callable, TypeVar

import deepspeed
import pandas as pd
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.utils import clip_grad_norm_
from torch import nn

from .distributed import fix_unset_envs

logger = logging.getLogger(__name__)

T = TypeVar("T")


def flatten_dict(d):
    records = pd.json_normalize(d, sep="/").to_dict(orient="records")
    return records[0] if records else {}


def _get_named_modules(module, attrname, sep="/"):
    for name, module in module.named_modules():
        name = name.replace(".", sep)
        if hasattr(module, attrname):
            yield name, module


def gather_attribute(module, attrname, delete=True, prefix=None):
    ret = {}
    for name, module in _get_named_modules(module, attrname):
        ret[name] = getattr(module, attrname)
        if delete:
            try:
                delattr(module, attrname)
            except Exception as e:
                raise RuntimeError(f"{name} {module} {attrname}") from e
    if prefix:
        ret = {prefix: ret}
    ret = flatten_dict(ret)
    # remove consecutive /
    ret = {re.sub(r"\/+", "/", k): v for k, v in ret.items()}
    return ret


def dispatch_attribute(module, attrname, value, filter_fn: Callable[[nn.Module], bool] | None = None):
    for _, module in _get_named_modules(module, attrname):
        if filter_fn is None or filter_fn(module):
            setattr(module, attrname, value)


@cache
def update_deepspeed_logger():
    logger = logging.getLogger("DeepSpeed")
    logger.setLevel(logging.WARNING)


@cache
def init_distributed():
    update_deepspeed_logger()
    fix_unset_envs()
    deepspeed.init_distributed(get_accelerator().communication_backend_name())


def _try_each(*fns, e=None):
    if len(fns) == 0:
        raise RuntimeError("All functions failed")

    head, *tails = fns

    try:
        return head()
    except Exception as e:
        logger.warning(f"Tried {head} but failed: {e}, trying next")
        return _try_each(*tails)


class Engine(DeepSpeedEngine):
    def __init__(self, *args, ckpt_dir, **kwargs):
        init_distributed()
        super().__init__(args=None, *args, **kwargs)
        self._ckpt_dir = ckpt_dir
        self._frozen_params = set()
        self._fp32_grad_norm = None

    @property
    def path(self):
        return self._ckpt_dir

    def freeze_(self):
        for p in self.module.parameters():
            if p.requires_grad:
                p.requires_grad_(False)
                self._frozen_params.add(p)

    def unfreeze_(self):
        for p in self._frozen_params:
            p.requires_grad_(True)
        self._frozen_params.clear()

    @property
    def global_step(self):
        return self.global_steps

    def gather_attribute(self, *args, **kwargs):
        return gather_attribute(self.module, *args, **kwargs)

    def dispatch_attribute(self, *args, **kwargs):
        return dispatch_attribute(self.module, *args, **kwargs)

    def clip_fp32_gradients(self):
        self._fp32_grad_norm = clip_grad_norm_(
            parameters=self.module.parameters(),
            max_norm=self.gradient_clipping(),
            mpu=self.mpu,
        )

    def get_grad_norm(self):
        grad_norm = self.get_global_grad_norm()
        if grad_norm is None:
            grad_norm = self._fp32_grad_norm
        return grad_norm

    def save_checkpoint(self, *args, **kwargs):
        if not self._ckpt_dir.exists():
            self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        super().save_checkpoint(save_dir=self._ckpt_dir, *args, **kwargs)
        logger.info(f"Saved checkpoint to {self._ckpt_dir}")

    def load_checkpoint(self, *args, **kwargs):
        fn = partial(super().load_checkpoint, *args, load_dir=self._ckpt_dir, **kwargs)
        return _try_each(
            lambda: fn(),
            lambda: fn(load_optimizer_states=False),
            lambda: fn(load_lr_scheduler_states=False),
            lambda: fn(load_optimizer_states=False, load_lr_scheduler_states=False),
            lambda: fn(
                load_optimizer_states=False,
                load_lr_scheduler_states=False,
                load_module_strict=False,
            ),
        )
