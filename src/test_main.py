import torch
from loaders import get_dataloader
from models import Conv4
from algos.utils import put_on_device

def main():
    loader = get_dataloader(
        data_root="miniImagenet",
        dataset="miniImagenet",
        dataset_type="train",
        num_workers=1,
        sample={
            "metatrain_iterations": 2,
            "n_way": 5,
            "k_shot": 1,
            "k_query": 15,
            "meta_batch_size": 4,
            "shuffle": True,
        },
    )
    model = Conv4(
        device="cuda",
        criterion=torch.nn.CrossEntropyLoss(),
        train_classes=2,
    )
    weights = [p.clone().detach().to("cuda") for p in model.parameters()]
    for id_, batch in enumerate(loader):
        print(f"Batch {id_}")
        # print(batch)
        print("==" * 60)
        ls = []
        for task in batch:
            support, query = task[0], task[1]
            sup_x, sup_y = support
            que_x, que_y = query
            print("dtype: sup - {}:{}, que - {}:{}".format(sup_x.dtype, sup_y.dtype, que_x.dtype, que_y.dtype))
            sup_x, sup_y, que_x, que_y = put_on_device("cuda", [sup_x, sup_y, que_x, que_y])
            print(sup_x.shape, sup_y.shape)
            print(que_x.shape, que_y.shape)
            print("**" * 60)
            pred = model.forward_weights(sup_x, weights)
            print(pred.shape)
            print(pred)
            print("++" * 60)
            print("pred dtype: {}".format(pred.dtype))
            print("sup_y dtype: {}".format(sup_y.dtype))
            l, g = get_loss_with_grad(model, sup_x, sup_y, weights)
            print(l)
            w = update_w(weights, g)
            
            print("test with que")
            pred = model.forward_weights(que_x, w)
            print(pred.shape)
            print(pred)
            print("++" * 60)
            print("pred dtype: {}".format(pred.dtype))
            print("que_y dtype: {}".format(que_y.dtype))
            l, g = get_loss_with_grad(model, que_x, que_y, w)
            print(l)
            ls.append(l)

            print("--" * 60)
        print(ls)
        # update init weights by Adam
        

        print("==" * 60)


def update_w(w, g, al=0.001):
    for w_i, g_i in zip(w, g):
        w_i = w_i - al * g_i
    return w

def get_loss_with_grad(model, x, y, weights):
    pred = model.forward_weights(x, weights)
    loss = model.criterion(pred, y)

    grads = torch.autograd.grad(
        loss, weights, create_graph=create_graph, retain_graph=retain_graph
    )

    gradients = list(grads)
    return loss, gradients



    return loss

if __name__ == "__main__":
    main()
