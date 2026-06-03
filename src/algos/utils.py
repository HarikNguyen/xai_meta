import torch


def put_on_device(dev, tensors):
    """Put tensors on specific device
    """
    for i in range(len(tensors)):
        if not tensors[i] is None:
            tensors[i] = tensors[i].to(dev)
    return tensors

def get_loss_n_preds(weights, learner, x, y):
    preds = learner.forward_weights(x, weights)
    loss = learner.criterion(preds, y)
    return loss, preds

def calc_accuracy(preds, y):
    """Computes accuracy of predictions
    """
    _, pred_idx = torch.max(preds, dim=1)
    _, true_idx = torch.max(y, dim=1)

    accuracy = (pred_idx == true_idx).float().mean()

    return accuracy.item()

