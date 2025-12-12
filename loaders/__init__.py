import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from .datasets import MiniImagenetDataset
from .transforms import make_transform
from .samplers import BatchTaskSampler


def encode_labels(labels):
    mapping = {}
    counter = 0
    result = []

    for item in labels:
        if item not in mapping:
            mapping[item] = counter
            counter += 1
        result.append(mapping[item])

    return torch.tensor(result)


def _task_collate(batch):
    """Collate function that handles collection type of element
    within batch_support and batch_query in each batch.

    Args
    ----------
    batch: List[batch_support, batch_query]
        A batch to be collated

    Returns
    ----------
    Tuple[batch_support_collated, batch_query_collated]
    """
    task_batches_collated = []
    for task_batch in batch:
        batch_support = task_batch[0]
        batch_query = task_batch[1]

        batch_support = default_collate(batch_support)
        batch_query = default_collate(batch_query)
        batch_support[1] = encode_labels(batch_support[1])
        batch_query[1] = encode_labels(batch_query[1])

        _, batch_support_labels = torch.unique(batch_support[1], return_inverse=True)
        _, batch_query_labels = torch.unique(batch_query[1], return_inverse=True)

        batch_support[1] = batch_support_labels
        batch_query[1] = batch_query_labels

        task_batches_collated.append((batch_support, batch_query))

    return task_batches_collated


def get_dataloader(
    data_root,
    dataset,
    dataset_type,
    num_workers,
    out_path=False,
    sample=None,
    seed=None,
):
    # Get transform
    transform = make_transform()

    # Get dataset
    dataset = MiniImagenetDataset(
        data_root,
        dataset,
        dataset_type,
        out_path=out_path,
        transform=transform,
    )

    # Create dataloader
    sampler = BatchTaskSampler(dataset.classes, seed=seed, **sample)

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_task_collate,
    )
    # Return
    return loader
