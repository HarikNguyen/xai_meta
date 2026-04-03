import torch
from loaders import get_dataloader
from models import Conv4

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
    model = Conv4(
        device="cuda",
        criterion=torch.nn.CrossEntropyLoss(),
        train_classes=64,
    )
    for id_, batch in enumerate(loader):
        print(f"Batch {id_}")
        # print(batch)
        print("==" * 60)
        for task in batch:
            support, query = task[0], task[1]
            sup_x, sup_y = support
            que_x, que_y = query
            print(sup_x.shape, sup_y.shape)
            print(que_x.shape, que_y.shape)
            print("**" * 60)
            pred = model.forward(sup_x)
            print(pred.shape)
            print(pred)

if __name__ == "__main__":
    main()
