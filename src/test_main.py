import torch
from loaders import get_dataloader
from models import Conv4
from algos.utils import put_on_device
import os
from sklearn.decomposition import PCA

def main():
    loader = get_dataloader(
        data_root="miniImagenet",
        dataset="miniImagenet",
        dataset_type="train",
        num_workers=1,
        sample={
            "metatrain_iterations": 1000,
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
            "metatrain_iterations": 11,
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
        if id_ % 100 == 0:
            # store weights
            if not os.path.exists("weights"):
                os.makedirs("weights")
            elif id_ == 0:
                for f in os.listdir("weights"):
                    os.remove(os.path.join("weights", f))
                    os.rmdir("weights")
                os.makedirs("weights")
            else:
                pass

            torch.save(weights, f"weights/weights_{id_}.pt")
            # test with loader
            avg_pred, avg_post = test(model, weights, iter(test_loader))
            print(avg_pred)
            print(avg_post)
            checks.append((avg_pred[0], avg_pred[1], avg_post[0], avg_post[1]))

    # write to file
    if os.path.exists("losses.txt"):
        os.remove("losses.txt")
    if os.path.exists("checks.txt"):
        os.remove("checks.txt")

    with open("losses.txt", "w") as f:
        f.write(str(losses))
    with open("checks.txt", "w") as f:
        f.write(str(checks))

    export_landscape_data(model, loader, "weights")

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

def generate_landscape_data(model, loader, checkpoint_dir, output_file="meta_landscape.npz", resolution=25):
    print("Server: Starting Meta-Loss Landscape calculation...")

    # --- 1. Extract PCA Directions from Checkpoints ---
    all_flat_weights = []
    files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')], 
                   key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    for f in files:
        weights = torch.load(os.path.join(checkpoint_dir, f), map_location='cuda')
        # Flattening all parameters into a single vector
        flat_w = torch.cat([p.data.view(-1) for p in weights]).detach().cpu().numpy()
        all_flat_weights.append(flat_w)
    
    all_flat_weights = np.array(all_flat_weights)
    pca = PCA(n_components=2)
    pca.fit(all_flat_weights)
    
    d1, d2 = pca.components_
    center_weights = all_flat_weights[-1] # Centering around the final checkpoint

    # --- 2. Initialize Grid ---
    alphas = np.linspace(-1.5, 1.5, resolution)
    betas = np.linspace(-1.5, 1.5, resolution)
    Z_meta_loss = np.zeros((resolution, resolution))
    
    # Grab one representative batch of tasks for consistent evaluation
    task_batch = next(iter(loader)) 

    # --- 3. The Grid Scan (GPU Intensive) ---
    print(f"Scanning {resolution}x{resolution} points on GPU...")
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            # Compute new flat weights for this coordinate
            current_flat = center_weights + a * d1 + b * d2
            
            # Reconstruct weight list for the model
            curr_weights = []
            ptr = 0
            for p in model.parameters():
                numel = p.numel()
                curr_weights.append(torch.from_numpy(current_flat[ptr:ptr+numel]).view(p.shape).cuda())
                ptr += numel
            
            # Compute Meta-Loss (Inner Loop adaptation + Outer Loop evaluation)
            batch_losses = []
            for task in task_batch:
                sup_x, sup_y = put_on_device("cuda", task[0])
                que_x, que_y = put_on_device("cuda", task[1])
                
                # Fast Adaptation (e.g., 5 steps)
                w_adapted = [p.clone() for p in curr_weights]
                for _ in range(5):
                    # logic from your 'get_loss_with_grad' and 'update_w' functions
                    # Assuming these are available in your namespace
                    l, g = get_loss_with_grad(model, sup_x, sup_y, w_adapted)
                    w_adapted = [wi - 0.01 * gi for wi, gi in zip(w_adapted, g)]
                
                # Query Loss (Meta-Loss)
                with torch.no_grad():
                    query_pred = model.forward_weights(que_x, w_adapted)
                    q_loss = model.criterion(query_pred, que_y)
                    batch_losses.append(q_loss.item())
            
            Z_meta_loss[i, j] = np.mean(batch_losses)

    # --- 4. Transform Training Trajectory to PCA space ---
    traj_coords = pca.transform(all_flat_weights)

    # --- 5. Export for Client ---
    np.savez(output_file, 
             alphas=alphas, 
             betas=betas, 
             Z=Z_meta_loss, 
             traj_x=traj_coords[:, 0], 
             traj_y=traj_coords[:, 1])
    print(f"Exported landscape data to {output_file}")

if __name__ == "__main__":
    main()
