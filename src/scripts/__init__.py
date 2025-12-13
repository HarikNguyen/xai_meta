import os
import numpy as np
import math
import torch
import torch.multiprocessing as mp
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc

from .warm_up import warm_up
from algos.base import BaseAlgorithm
from algos.maml import MAML
from algos.utils import put_on_device

from rpc.maml import run_train_master, init_worker


def run(
    validate,
    world_size,
):
    mp.set_start_method("spawn", force=True)
    world_size = world_size
    mp.spawn(run_process, args=(world_size, validate), nprocs=world_size, join=True)


def run_process(rank, world_size, validate):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    device_maps = {
        f"worker{i}": {torch.device("cuda:0"): torch.device("cuda:0")}
        for i in range(world_size)
    }

    options = rpc.TensorPipeRpcBackendOptions(
        device_maps=device_maps,
    )

    rpc.init_rpc(
        name=f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options,
    )

    if rank == 0:
        train_loader, val_loader, algo_conf = warm_up()

        maml = MAML(**algo_conf)
        workers = [f"worker{i}" for i in range(1, world_size)]

        # Initialize worker-local algo instances so we don't send the full
        # `maml` object with every RPC.
        for w in workers:
            rpc.rpc_sync(w, init_worker, args=(algo_conf,))

        run_train_master(maml, workers, train_loader)

    # shutdown all
    rpc.shutdown()
