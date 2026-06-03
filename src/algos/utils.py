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

def accuracy(y_pred, y):
    """Computes accuracy of predictions

    Compute the ratio of correct predictions on the true labels y.

    Parameters
    ----------
    y_pred : torch.Tensor
        Tensor of label predictions
    y : torch.Tensor
        Tensor of ground-truth labels

    Returns
    ----------
    accuracy
        Float accuracy score in [0,1]
    """

    # return ((y_pred == y).float().sum() / len(y)).item()
    _, pred_idx = torch.max(y_pred, dim=1)
    _, true_idx = torch.max(y, dim=1)

    accuracy = (pred_idx == true_idx).float().mean()

    return accuracy.item()

