import torch
from loaders import get_dataloader


def main():
    loader = get_dataloader(
        data_root="miniImagenet",
        dataset="miniImagenet",
        dataset_type="train",
        num_workers=1,
        sample={
            "metatrain_iterations": 2,
            "n_way": 2,
            "k_shot": 2,
            "k_query": 3,
            "meta_batch_size": 1,
            "shuffle": True,
        },
    )

    for batch in loader:
        print(batch)
        break


if __name__ == "__main__":
    main()
