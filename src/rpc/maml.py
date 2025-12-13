import os
import time
import copy
import typing
from queue import Queue

import torch
import torch.distributed.rpc as rpc

from algos.utils import put_on_device
from algos.maml import MAML

# Worker-local algorithm reference (set via `init_worker`)
_GLOBAL_ALGO = None


def init_worker(algo_conf, state=None):
    """Initialize a local algo instance on the worker.

    Called remotely by the master once before training starts. This avoids
    repeatedly serializing the whole `algo_obj` to workers.
    """
    global _GLOBAL_ALGO
    _GLOBAL_ALGO = MAML(**algo_conf)
    return True


def run_train_master(algo_obj, worker_list, train_loader):
    start_time = time.time()
    for batch_id, task_batch in enumerate(train_loader):
        mean_pre_losses, mean_post_losses = train_on_meta_batch(algo_obj, worker_list, task_batch)

        if batch_id % 100 == 0:
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"Meta-batch {batch_id}: {mean_pre_losses}, {mean_post_losses} | Time: {elapsed:.3f}s")
            start_time = time.time()

        weights = algo_obj.dump_state()
        torch.save(weights, "meta_init.pt")

def train_on_meta_batch(algo_obj, worker_list, task_batch):
    num_workers = len(worker_list)
    processed = 0
    total_task = algo_obj.meta_batch_size

    zero_state = algo_obj.dump_state()

    results = []
    while processed < total_task:
        remaining = total_task - processed
        part_size = min(num_workers, remaining)
        
        futs = []
        for w in range(1, part_size + 1):
            task_data = task_batch.pop(0)
            # Send only the task data to the worker. Worker must have been
            # initialized beforehand via `init_worker` so it has a local algo.
            fut = rpc.rpc_async(
                f"worker{w}",
                run_task_remote,
                args=(task_data,zero_state),
            )
            futs.append(fut)
        
        for fut in futs:
            result = fut.wait()
            results.append(result)

        processed += part_size

    pre_losses, post_losses = [], []
    for pre_loss, post_loss in results:
        pre_losses.append(pre_loss)
        post_losses.append(post_loss)

    pre_losses = torch.stack(pre_losses)
    post_losses = torch.stack(post_losses)
    
    return algo_obj.outer_train(pre_losses, post_losses)


def run_task_remote(task_data, zero_state):
    """Run task on worker using the worker-local algo instance.

    This function should be executed on the worker process which has called
    `init_worker` earlier. It avoids passing the full `algo_obj` over RPC.
    """
    global _GLOBAL_ALGO
    if _GLOBAL_ALGO is None:
        raise RuntimeError("Worker algorithm not initialized. Call init_worker first.")

    _GLOBAL_ALGO.load_state(zero_state)

    device = _GLOBAL_ALGO.device
    support, query = task_data
    train_x, train_y, test_x, test_y = put_on_device(
        device, [support[0], support[1], query[0], query[1]]
    )

    pre_loss, post_loss, _, _ = _GLOBAL_ALGO.inner_train(
        train_x, 
        train_y, 
        test_x, 
        test_y,
        rpc_mode=True,
    )

    return pre_loss, post_loss

