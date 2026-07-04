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

def get_stratified_bootstrap_batches(
    que_x,
    que_y,
    num_bootstraps: int,
    samples_per_class: int,
):
    if que_y.dim() > 1 and que_y.shape[1] > 1:
        que_y_labels = torch.argmax(que_y, dim=1)
    else:
        que_y_labels = que_y
    classes = torch.unique(que_y_labels)

    class_indices = {
        c.item(): (que_y_labels == c).nonzero(as_tuple=True)[0] 
        for c in classes
    }

    for _ in range(num_bootstraps):
        bootstrap_idx_list = []

        for c in classes:
            idx_of_class = class_indices[c.item()]
            num_available = len(idx_of_class)
            
            # random sampling WITH REPLACEMENT
            rand_picks = torch.randint(low=0, high=num_available, size=(samples_per_class,))
            bootstrap_idx_list.append(idx_of_class[rand_picks])

        # Combine into a single complete batch
        final_bootstrap_indices = torch.cat(bootstrap_idx_list)
        
        yield que_x[final_bootstrap_indices], que_y[final_bootstrap_indices]

