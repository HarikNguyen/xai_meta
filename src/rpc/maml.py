import time

import numpy as np
import torch
import torch.distributed.rpc as rpc

from algos.maml import MAML
from algos.utils import put_on_device

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


def _get_worker_name(worker_list, worker_idx):
    """Resolve worker name from configured list with backward-compatible fallback."""
    if worker_list:
        return worker_list[worker_idx]
    return f"worker{worker_idx + 1}"


def _dispatch_tasks(task_batch, total_task, worker_list, remote_fn, zero_state, val_mode=False):
    """Dispatch tasks to workers in chunks and gather RPC results in batches to workers."""
    num_workers = len(worker_list)
    if num_workers == 0:
        raise ValueError("worker_list must not be empty")

    if total_task > len(task_batch):
        raise ValueError(
            f"total_task ({total_task}) cannot exceed task_batch size ({len(task_batch)})"
        )

    # Dispatch tasks in chunks by index to avoid O(n^2) pop(0) overhead.
    results = []

    for chunk_start in range(0, total_task, num_workers):
        task_chunk = task_batch[chunk_start : chunk_start + num_workers]

        futs = []
        for worker_idx, task_data in enumerate(task_chunk):
            worker_name = _get_worker_name(worker_list, worker_idx)
            if val_mode:
                args = (task_data, zero_state, True)
            else:
                args = (task_data, zero_state)
            fut = rpc.rpc_async(
                worker_name,
                remote_fn,
                args=args,
            )
            futs.append(fut)

        # Gather results
        results.extend(fut.wait() for fut in futs)

    return results


def run_train_master(algo_obj, worker_list, train_loader, val_loader):
    total_task = algo_obj.meta_batch_size

    start_time = time.time()
    for batch_id, task_batch in enumerate(train_loader):
        mean_pre_losses, mean_post_losses = train_on_meta_batch(
            algo_obj, worker_list, task_batch
        )

        if batch_id % 100 == 0:
            elapsed = time.time() - start_time
            print(
                f"Meta-batch {batch_id}: {mean_pre_losses}, {mean_post_losses} | Time: {elapsed:.3f}s"
            )

            start_time = time.time()

        if batch_id % 1000 == 0:
            zero_state_cur = algo_obj.dump_state()
            pre_accs_avg, post_accs_avg, pre_accs_max, post_accs_max = run_val_master(
                zero_state_cur, total_task, worker_list, val_loader
            )
            print(
                f"Meta-batch {batch_id}:\n- pre_accs_avg: {pre_accs_avg}\n- post_accs_avg: {post_accs_avg}"
            )
            print(f"- pre_accs_max: {pre_accs_max}\n- post_accs_max: {post_accs_max}")

            # Save intermediate meta-learner state (checkpoints).
            algo_obj.store_file(f"meta_init_{batch_id}.pt")

    # Save final meta-learner state once training ends.
    algo_obj.store_file("meta_init.pt")


def train_on_meta_batch(algo_obj, worker_list, task_batch):
    total_task = algo_obj.meta_batch_size

    zero_state = algo_obj.dump_state()

    results = _dispatch_tasks(
        task_batch,
        total_task,
        worker_list,
        run_train_task_remote,
        zero_state,
    )

    pre_losses = torch.stack([pre_loss for pre_loss, _ in results])
    post_losses = torch.stack([post_loss for _, post_loss in results])

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

    pre_loss, post_losses, _, _ = _GLOBAL_ALGO.inner_train(
        train_x,
        train_y,
        test_x,
        test_y,
        rpc_mode=True,
    )

    return pre_loss, post_losses[-1]


def run_val_master(zero_state, total_task, worker_list, val_loader):
    try:
        task_batch = next(iter(val_loader))
    except StopIteration:
        return 0.0, 0.0, 0.0, 0.0

    pre_accs, post_accs = val_on_meta_batch(
        zero_state,
        total_task,
        worker_list,
        task_batch,
        val_mode=True,
    )

    pre_accs_tensor = torch.tensor(pre_accs)
    post_accs_tensor = torch.tensor(post_accs)

    return (
        pre_accs_tensor.mean().item(),
        post_accs_tensor.mean().item(),
        pre_accs_tensor.max().item(),
        post_accs_tensor.max().item(),
    )


def run_test_master(algo_obj, worker_list, test_loader):
    algo_obj.read_file("meta_init.pt")
    total_task = algo_obj.test_batch_size
    all_results = None

    print("Starting Meta-Testing...")

    zero_state_cur = algo_obj.dump_state()

    for task_batch in test_loader:
        batch_pre_accs, batch_post_accs = check_on_meta_batch(
            zero_state_cur, total_task, worker_list, task_batch
        )

        combined_accs = np.column_stack((batch_pre_accs,batch_post_accs))

        if all_results is None:
            all_results = combined_accs
        else:
            all_results += combined_accs

    print(all_results.shape)
    num_test_points = all_results.shape[0]
    means = np.mean(all_results, axis=0)
    stds = np.std(all_results, axis=0)
    ci95 = 1.96 * stds / np.sqrt(num_test_points)

    print("\nMean validation accuracy/loss, stddev, and confidence intervals")
    print(f"Means: {means}")
    print(f"Stds:  {stds}")
    print(f"CI95:  {ci95}")


def check_on_meta_batch(zero_state, total_task, worker_list, task_batch, val_mode=False):
    results = _dispatch_tasks(
        task_batch,
        total_task,
        worker_list,
        run_check_task_remote,
        zero_state,
        val_mode=val_mode,
    )

    pre_accs = [pre_res for pre_res, _ in results]
    post_accs = [post_res for _, post_res in results]
    
    return pre_accs, post_accs


def run_check_task_remote(task_data, zero_state, val_mode=False):
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

    pre_acc, post_accs = _GLOBAL_ALGO.acc_val(
        train_x,
        train_y,
        test_x,
        test_y,
        val_mode=val_mode,
        rpc_mode=True,
    )

    if val_mode:
        return pre_acc, post_accs[-1]
    else:
        return pre_acc, post_accs
