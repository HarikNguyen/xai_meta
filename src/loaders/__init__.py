import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, default_collate
from .datasets import FewShotDataset
from .transforms import make_transform
from .samplers import BatchTaskSampler


def encode_labels(labels, mapping=None):
    # 1. Initialize or generate the mapping
    if mapping is None:
        unique_labels = sorted(list(set(labels)))
        mapping = {label: i for i, label in enumerate(unique_labels)}

    num_classes = len(mapping)

    # 2. Convert labels to integer indices
    try:
        indices = [mapping[item] for item in labels]
    except KeyError as e:
        raise ValueError(f"Label {e} not found in the provided mapping.")

    indices_tensor = torch.tensor(indices)

    # 3. Convert indices to One-Hot vectors
    # F.one_hot returns a tensor of shape [N, num_classes]
    one_hot_tensor = F.one_hot(indices_tensor, num_classes=num_classes)

    return one_hot_tensor.float()


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

        # _, batch_support_labels = torch.unique(batch_support[1], return_inverse=True)
        # _, batch_query_labels = torch.unique(batch_query[1], return_inverse=True)

        # batch_support[1] = batch_support_labels
        # batch_query[1] = batch_query_labels

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
    dataset = FewShotDataset(
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
