import os
import time
import copy
import typing
from queue import Queue

import numpy as np
import torch
import torch.distributed.rpc as rpc

from algos.utils import put_on_device
from algos.maml import MAML

# Worker-local algorithm reference (set via `init_worker`)
_GLOBAL_ALGO = None


def init_worker(algo_conf):
    """Initialize a local algo instance on the worker.

    Called remotely by the master once before training starts. This avoids
    repeatedly serializing the whole `algo_obj` to workers.
    """
    global _GLOBAL_ALGO
    _GLOBAL_ALGO = MAML(**algo_conf)
    return True


def run_train_master(algo_obj, worker_list, train_loader, val_loader):
    total_task = algo_obj.meta_batch_size

    start_time = time.time()
    for batch_id, task_batch in enumerate(train_loader):
        mean_pre_losses, mean_post_losses = train_on_meta_batch(algo_obj, worker_list, task_batch)

        if batch_id % 100 == 0:
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"Meta-batch {batch_id}: {mean_pre_losses}, {mean_post_losses} | Time: {elapsed:.3f}s")
            
            start_time = time.time()

        if batch_id % 1000 == 0:
            zero_state_cur = algo_obj.dump_state()
            pre_accs_avg, post_accs_avg, pre_accs_max, post_accs_max = run_val_master(zero_state_cur, total_task, worker_list, val_loader)
            print(f"Meta-batch {batch_id}:\n- pre_accs_avg: {pre_accs_avg}\n- post_accs_avg: {post_accs_avg}")
            print(f"- pre_accs_max: {pre_accs_max}\n- post_accs_max: {post_accs_max}")

        algo_obj.store_file("meta_init.pt")

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
            fut = rpc.rpc_async(
                f"worker{w}",
                run_train_task_remote,
                args=(task_data,zero_state,),
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

def run_train_task_remote(task_data, zero_state):
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


def run_val_master(zero_state, total_task, worker_list, val_loader):
    try:
        task_batch = next(iter(val_loader))
    except StopIteration:
        return 0.0, 0.0, 0.0, 0.0
    
    pre_accs, post_accs = val_on_meta_batch(zero_state, total_task, worker_list, task_batch)

    pre_accs_tensor = torch.tensor(pre_accs)
    post_accs_tensor = torch.tensor(post_accs)

    return (
        pre_accs_tensor.mean().item(),
        post_accs_tensor.mean().item(),
        pre_accs_tensor.max().item(),
        post_accs_tensor.max().item(),
    )

def val_on_meta_batch(zero_state, total_task, worker_list, task_batch):
    num_workers = len(worker_list)
    processed = 0

    results = []
    while processed < total_task:
        remaining = total_task - processed
        part_size = min(num_workers, remaining)
        
        futs = []
        for w in range(1, part_size + 1):
            task_data = task_batch.pop(0)
            fut = rpc.rpc_async(
                f"worker{w}",
                run_val_task_remote,
                args=(task_data,zero_state,),
            )
            futs.append(fut)
        
        for fut in futs:
            result = fut.wait()
            results.append(result)

        processed += part_size

    pre_accs, post_accs = [], []
    for pre_acc, post_acc in results:
        pre_accs.append(pre_acc)
        post_accs.append(post_acc)

    return pre_accs, post_accs

def run_val_task_remote(task_data, zero_state):
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

    return _GLOBAL_ALGO.acc_val(
        train_x,
        train_y,
        test_x,
        test_y,
        val_mode=True,
        rpc_mode=True,
    )


def run_test_master(algo_obj, worker_list, test_loader):
    algo_obj.read_file("meta_init.pt")
    total_task = algo_obj.meta_batch_size
    all_results = []

    print("Starting Meta-Testing...")
    
    zero_state_cur = algo_obj.dump_state()

    for task_batch in test_loader:
        print(task_batch)
        print(len(task_batch))
        batch_pre_accs, batch_post_accs = val_on_meta_batch(
            zero_state_cur, total_task, worker_list, task_batch
        )
        
        all_results.append(batch_pre_accs + batch_post_accs)
        break

    all_results = np.array(all_results) # Shape: [Số lượng task, Số bước update]
    
    num_test_points = all_results.shape[0]
    means = np.mean(all_results, axis=0)
    stds = np.std(all_results, axis=0)
    ci95 = 1.96 * stds / np.sqrt(num_test_points)

    print('\nMean validation accuracy/loss, stddev, and confidence intervals')
    print(f"Means: {means}")
    print(f"Stds:  {stds}")
    print(f"CI95:  {ci95}")
