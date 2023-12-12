from typing import Callable, TypeVar, overload

import matplotlib.pyplot as plt
import numpy as np


def save_mels(path, *, targ_mel, pred_mel, cond_mel):
    n = 3 if cond_mel is None else 4

    plt.figure(figsize=(10, n * 4))

    i = 1

    plt.subplot(n, 1, i)
    plt.imshow(pred_mel, origin="lower", interpolation="none")
    plt.title(f"Pred mel {pred_mel.shape}")
    i += 1

    plt.subplot(n, 1, i)
    plt.imshow(targ_mel, origin="lower", interpolation="none")
    plt.title(f"GT mel {targ_mel.shape}")
    i += 1

    plt.subplot(n, 1, i)
    pred_mel = pred_mel[:, : targ_mel.shape[1]]
    targ_mel = targ_mel[:, : pred_mel.shape[1]]
    plt.imshow(np.abs(pred_mel - targ_mel), origin="lower", interpolation="none")
    plt.title(f"Diff mel {pred_mel.shape}, mse={np.mean((pred_mel - targ_mel)**2):.4f}")
    i += 1

    if cond_mel is not None:
        plt.subplot(n, 1, i)
        plt.imshow(cond_mel, origin="lower", interpolation="none")
        plt.title(f"Cond mel {cond_mel.shape}")
        i += 1

    plt.savefig(path, dpi=480)
    plt.close()


T = TypeVar("T")


@overload
def tree_map(fn: Callable, x: list[T]) -> list[T]:
    ...


@overload
def tree_map(fn: Callable, x: tuple[T]) -> tuple[T]:
    ...


@overload
def tree_map(fn: Callable, x: dict[str, T]) -> dict[str, T]:
    ...


@overload
def tree_map(fn: Callable, x: T) -> T:
    ...


def tree_map(fn: Callable, x):
    if isinstance(x, list):
        x = [tree_map(fn, xi) for xi in x]
    elif isinstance(x, tuple):
        x = (tree_map(fn, xi) for xi in x)
    elif isinstance(x, dict):
        x = {k: tree_map(fn, v) for k, v in x.items()}
    else:
        x = fn(x)
    return x
