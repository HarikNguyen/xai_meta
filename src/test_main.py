import torch
from loaders import get_dataloader
from models import Conv4
from algos.utils import put_on_device
import os


def main():
    loader = get_dataloader(
        data_root="miniImagenet",
        dataset="miniImagenet",
        dataset_type="train",
        num_workers=1,
        sample={
            "metatrain_iterations": 10000,
            "n_way": 5,
            "k_shot": 1,
            "k_query": 15,
            "meta_batch_size": 4,
            "shuffle": True,
        },
    )
    test_loader = get_dataloader(
        data_root="miniImagenet",
        dataset="miniImagenet",
        dataset_type="test",
        num_workers=1,
        sample={
            "metatrain_iterations": 10,
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
        train_classes=5,
    )
    # weights = [p.clone().to("cuda") for p in model.parameters()]
    weights = [p.clone().to("cuda").detach().requires_grad_(True) for p in model.parameters()]
    optim = torch.optim.Adam(weights, lr=0.001)
    losses = []
    checks = []
    for id_, batch in enumerate(loader):
        print()
        print("=*=" * 60)
        print(f"Batch {id_}")
        # print(batch)
        # print("==" * 60)
        ls = []
        for task in batch:
            support, query = task[0], task[1]
            sup_x, sup_y = support
            que_x, que_y = query
            # print("dtype: sup - {}:{}, que - {}:{}".format(sup_x.dtype, sup_y.dtype, que_x.dtype, que_y.dtype))
            sup_x, sup_y, que_x, que_y = put_on_device("cuda", [sup_x, sup_y, que_x, que_y])
            # print(sup_x.shape, sup_y.shape)
            # print(que_x.shape, que_y.shape)
            # print("**" * 60)
            pred = model.forward_weights(sup_x, weights)
            # print(f"pred shape: {pred.shape}")
            # print("pred", pred)
            # print("++" * 60)
            # print("pred dtype: {}".format(pred.dtype))
            # print("sup_y dtype: {}".format(sup_y.dtype))
            w = [p.clone() for p in weights]
            for _ in range(5):
                l, g = get_loss_with_grad(model, sup_x, sup_y, w)
                # print(l)
                w = update_w(w, g)
            
            # print("test with que")
            pred = model.forward_weights(que_x, w)
            # print("pred que shape: ", pred.shape)
            # print("pred que", pred)
            # print("++" * 60)
            # print("pred dtype: {}".format(pred.dtype))
            # print("que_y dtype: {}".format(que_y.dtype))
            l = get_loss_with_grad(model, que_x, que_y, w, True)
            # print(l)
            ls.append(l)

            # print("--" * 60)
        print(ls)
        # update init weights by Adam
        avg_l = torch.mean(torch.stack(ls))
        losses.append(avg_l.item())
        print(avg_l)
        # print("==" * 60)
        optim.zero_grad()
        avg_l.backward()
        
        # clip gradient
        for p in weights:
            if p.grad is not None:
                p.grad.data.clamp_(-10, 10)

        optim.step()

        print("=*=" * 60)
        print()
        if id_ % 1000 == 0:
            # test with loader
            avg_pred, avg_post = test(model, weights, iter(test_loader))
            print(avg_pred)
            print(avg_post)
            checks.apped((avg_pred[0], avg_pred[1], avg_post[0], avg_post[1]))

    # write to file
    with open("losses.txt", "w") as f:
        f.write(str(losses))
    with open("checks.txt", "w") as f:
        f.write(str(checks))

def test(model, weights, loader_iter):
    batch = next(loader_iter)
    preds = []
    posts = []
    for task in batch:
        support, query = task[0], task[1]
        sup_x, sup_y = support
        que_x, que_y = query
        sup_x, sup_y, que_x, que_y = put_on_device("cuda", [sup_x, sup_y, que_x, que_y])
        
        # call pre accs
        sup_pred = model.forward_weights(sup_x, weights)
        sup_acc = get_accuracy(sup_pred, sup_y)
        que_pred = model.forward_weights(que_x, weights)
        que_acc = get_accuracy(que_pred, que_y)

        preds.append((sup_acc, que_acc))

        # update 5 times
        w = [p.clone() for p in weights]
        for _ in range(5):
            l, g = get_loss_with_grad(model, sup_x, sup_y, w)
            w = update_w(w, g)
        
        # call post accs
        sup_pred = model.forward_weights(sup_x, w)
        sup_acc = get_accuracy(sup_pred, sup_y)
        que_pred = model.forward_weights(que_x, w)
        que_acc = get_accuracy(que_pred, que_y)

        posts.append((sup_acc, que_acc))
    
    avg_pred = (sum(p[0] for p in preds) / len(preds), sum(p[1] for p in preds) / len(preds))
    avg_post = (sum(p[0] for p in posts) / len(posts), sum(p[1] for p in posts) / len(posts))
    return avg_pred, avg_post
        

def get_accuracy(preds, y):
    _, pred_idx = torch.max(preds, dim=1)
    _, true_idx = torch.max(y, dim=1)
    
    accuracy = (pred_idx == true_idx).float().mean()

    return accuracy.item()

def update_w(w, g, al=0.01):
    # for w_i, g_i in zip(w, g):
        # w_i = w_i - al * g_i
    # clip gradient
    g = [torch.clamp(p, -10, 10) for p in g]

    return [w_i - al * g_i for w_i, g_i in zip(w, g)]

def get_loss_with_grad(model, x, y, weights, r_l=False):
    model.zero_grad()
    pred = model.forward_weights(x, weights)
    loss = model.criterion(pred, y)

    grads = torch.autograd.grad(
        loss, weights, create_graph=True, retain_graph=True
    )

    gradients = list(grads)
    if r_l:
        return loss 
    return loss, gradients



    return loss

if __name__ == "__main__":
    main()
