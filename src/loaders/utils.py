import torch

def boT_to_stack(boT):
    """
    Convert a batch of tasks (boT) to stacked tensors (sup_x, sup_y, que_x, que_y).
    Parameters
    ----------
    boT : list of tasks
    ----------
    Returns
    sup_x, sup_y, que_x, que_y : torch.Tensor
    """
    # destructure boT into lists
    supports, queries = zip(*boT)
    sup_x, sup_y = zip(*supports)
    que_x, que_y = zip(*queries)

    # stack all
    sup_x = torch.stack(sup_x)
    sup_y = torch.stack(sup_y)
    que_x = torch.stack(que_x)
    que_y = torch.stack(que_y)
    return sup_x, sup_y, que_x, que_y
