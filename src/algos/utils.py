import torch


def put_on_device(dev, tensors):
    """Put arguments on specific device

    Places the positional arguments onto the user-specified device

    Parameters
    ----------
    dev : str
        Device identifier
    tensors : sequence/list
        Sequence of torch.Tensor variables that are to be put on the device
    """
    for i in range(len(tensors)):
        if not tensors[i] is None:
            tensors[i] = tensors[i].to(dev)
    return tensors


def process_cross_entropy(
    preds, targets, class_map, apply_softmax, dev, log=False, single_input=False
):
    """Converts the predictions and targets into a format that CrossEntropy expects
    Every row of the new input will consist of [one-hot encoding for preds, one-hot encoding for target]

    Args:
        preds (torch.Tensor): tensor of predictiopns of shape [num_examples,]
        targets (torch.Tensor): ground-truth labels of shape [num_examples,]
        class_map (dict): maps classes to column positions (ints) in the one-hot encoding
        apply_softmax (bool): whether to apply the softmax to the predictions
        dev (str): device identifier to put the inputs on
        log (bool): whether to take the log of inputs

    Returns:
        torch.Tensor: tensor of one-hot encoded predictions and targets of shape [num_examples, in_dim]
    """

    one_hot = torch.zeros((preds.size(0), 2 * len(class_map.keys())), device=dev)
    # this is the case of binary classification (only 1 output node, but 2 classes)
    if len(class_map.keys()) == 2:
        class_a, class_b = list(class_map.keys())
        # do the predictions
        one_hot[:, 0] = preds.view(-1)
        one_hot[:, 1] = 1 - preds.view(-1)
        if apply_softmax:
            one_hot[:, :2] = torch.softmax(one_hot[:, :2].clone(), dim=1)
        one_hot[targets == class_a, 2] = 1
        one_hot[targets == class_b, 3] = 1
        if log and not single_input:
            one_hot = torch.log(one_hot + 1e-5)

        outputs = one_hot[:, 2].detach().float().view(-1, 1)
        if single_input:
            if not log:
                one_hot = (one_hot[:, :2] * one_hot[:, 2:]).sum(dim=1).unsqueeze(1)
            else:
                one_hot = torch.log(
                    (one_hot[:, :2] * one_hot[:, 2:]).sum(dim=1).unsqueeze(1)
                )

    else:
        outputs = torch.zeros(targets.size(), dtype=torch.long, device=dev)
        num_classes = len(class_map.keys())
        for c, column in class_map.items():
            column = class_map[c]
            one_hot[:, column] = preds[:, column]
            one_hot[targets == c, num_classes + column] = 1
            outputs[targets == c] = column
        if apply_softmax:
            one_hot[:, :num_classes] = torch.softmax(
                one_hot[:, :num_classes].clone(), dim=1
            )
        if log and not single_input:
            one_hot = torch.log(one_hot + 1e-5)
        if single_input:
            if not log:
                one_hot = (
                    (one_hot[:, :num_classes] * one_hot[:, num_classes:])
                    .sum(dim=1)
                    .unsqueeze(1)
                )
            else:
                one_hot = torch.log(
                    (one_hot[:, :num_classes] * one_hot[:, num_classes:])
                    .sum(dim=1)
                    .unsqueeze(1)
                )

    return one_hot, outputs


def get_loss_and_grads(
    model,
    train_x,
    train_y,
    flat=True,
    weights=None,
    item_loss=True,
    create_graph=False,
    retain_graph=False,
    rt_only_loss=False,
    meta_loss=False,
    class_map=None,
    loss_net=None,
    loss_params=None,
):
    """Computes loss and gradients

    Apply model to data (train_x, train_y), compute the loss
    and obtain the gradients.

    Parameters
    ----------
    model : nn.Module
        Neural network. We assume the model has a criterion attribute
    train_x : torch.Tensor
        Training inputs
    train_y : torch.Tensor
        Training targets

    Returns
    ----------
    loss
        torch.Tensor of size (#params in model) containing the loss value
        if flat=True, else a single float
    gradients
        torch.Tensor of size (#params in model) containing gradients w.r.t.
        all model parameters if flat=True, else a structured list
    """
    model.zero_grad()
    if weights is None:
        weights = model.parameters()
        out = model(train_x)
    else:
        out = model.forward_weights(train_x, weights)

    if not meta_loss:
        loss = model.criterion(out, train_y)
    else:
        meta_inputs, _ = process_cross_entropy(
            out, train_y, class_map=class_map, apply_softmax=True, dev=model.dev
        )
        loss = loss_net(meta_inputs, weights=loss_params)

    if rt_only_loss:
        return loss, None

    grads = torch.autograd.grad(
        loss, weights, create_graph=create_graph, retain_graph=retain_graph
    )

    if flat:
        gradients = torch.cat([p.reshape(-1) for p in grads])
        loss = torch.zeros(gradients.size()).to(train_x.device) + loss.item()
    else:
        gradients = list(grads)
        if item_loss:
            loss = loss.item()
    return loss, gradients


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

