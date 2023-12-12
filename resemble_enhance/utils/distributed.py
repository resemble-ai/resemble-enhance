import os
import socket
from functools import cache, partial, wraps
from typing import Callable

import deepspeed
import torch
from deepspeed.accelerator import get_accelerator
from torch.distributed import broadcast_object_list


def get_free_port():
    sock = socket.socket()
    sock.bind(("", 0))
    return sock.getsockname()[1]


@cache
def fix_unset_envs():
    envs = dict(RANK="0", WORLD_SIZE="1", MASTER_ADDR="localhost", MASTER_PORT=str(get_free_port()), LOCAL_RANK="0")

    for key in envs:
        value = os.getenv(key)
        if value is not None:
            return

    for key, value in envs.items():
        os.environ[key] = value


@cache
def init_distributed():
    fix_unset_envs()
    deepspeed.init_distributed(get_accelerator().communication_backend_name())
    torch.cuda.set_device(local_rank())


def local_rank():
    return int(os.getenv("LOCAL_RANK", 0))


def global_rank():
    return int(os.getenv("RANK", 0))


def is_local_leader():
    return local_rank() == 0


def is_global_leader():
    return global_rank() == 0


def leader_only(leader_only_type, fn: Callable | None = None, boardcast_return=False) -> Callable:
    """
    Args:
        fn: The function to decorate
        boardcast_return: Whether to boardcast the return value to all processes
                        (may cause deadlock if the function calls another decorated function)
    """

    def wrapper(fn):
        if hasattr(fn, "__leader_only_type__"):
            raise RuntimeError(f"Function {fn.__name__} has already been decorated with {fn.__leader_only_type__}")

        fn.__leader_only_type__ = leader_only_type

        if leader_only_type == "local":
            guard_fn = is_local_leader
        elif leader_only_type == "global":
            guard_fn = is_global_leader
        else:
            raise ValueError(f"Unknown leader_only_type: {leader_only_type}")

        @wraps(fn)
        def wrapped(*args, **kwargs):
            if boardcast_return:
                init_distributed()
            obj_list = [None]
            if guard_fn():
                ret = fn(*args, **kwargs)
                obj_list[0] = ret
            if boardcast_return:
                broadcast_object_list(obj_list, src=0)
            return obj_list[0]

        return wrapped

    if fn is None:
        return wrapper

    return wrapper(fn)


local_leader_only = partial(leader_only, "local")
global_leader_only = partial(leader_only, "global")
