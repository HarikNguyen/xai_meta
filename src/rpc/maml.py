import time
import copy
import torch
from queue import Queue
import torch.distributed.rpc as rpc
from algos.utils import put_on_device


def run_train_master(algo_obj, worker_list, train_loader):
    for batch_id, task_batch in enumerate(train_loader):
        start_time = time.time()
        mean_pre_losses, mean_post_losses = train_on_meta_batch(algo_obj, worker_list, task_batch)
        end_time = time.time()

        if batch_id % 100 == 0:
            elapsed = end_time - start_time
            print(f"Meta-batch {batch_id}: {mean_pre_losses}, {mean_post_losses} | Time: {elapsed:.3f}s")

        weights = algo_obj.dump_state()
        torch.save(weights, "meta_init.pt")

def train_on_meta_batch(algo_obj, worker_list, task_batch):
    num_workers = len(worker_list)
    processed = 0
    total_task = algo_obj.meta_batch_size

    results = []
    while processed < total_task:
        remaining = total_task - processed
        part_size = min(num_workers, remaining)
        
        futs = []
        for w in range(1, part_size + 1):
            task_data = task_batch.pop(0)
            fut = rpc.rpc_async(
                f"worker{w}",
                run_task,
                args=(algo_obj, task_data),
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


def run_task(algo_obj, task_data):
    # algo_obj = copy.deepcopy(algo_obj)
    device = algo_obj.device
    support, query = task_data
    train_x, train_y, test_x, test_y = put_on_device(
        device, [support[0], support[1], query[0], query[1]]
    )

    pre_loss, post_loss, _, _ = algo_obj.inner_train(train_x, train_y, test_x, test_y)

    return pre_loss, post_loss

