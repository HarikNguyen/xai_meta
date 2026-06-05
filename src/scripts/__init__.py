import os
import numpy as np
import math
import torch
from tqdm import tqdm
from collections import defaultdict

from .warm_up import warm_up
from .utils import log_to_csv, compute_stats
from algos.maml import MAML

from loaders.utils import boT_to_stack


############################################################################################
### Main Func
############################################################################################

VAL_AFTER = 1000
TRAIN_MODE = "train"
TEST_MODE = "test"
LOG_DIR = "logs"

def run(args):
    if args.algo == "maml":
        algo_class = MAML
    else:
        raise NotImplementedError(f"Algorithm {algo} not implemented.")

    # warm up
    train_loader, val_loader, test_loader, algo_conf = warm_up()
    algo_conf["vmap_chunk_size"] = args.vmap_chunk_size

    checkpoint_dir = args.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if not os.path.exists(os.path.join(checkpoint_dir, algo_class.__name__)):
        os.makedirs(os.path.join(checkpoint_dir, algo_class.__name__))

    checkpoint_dir = os.path.join(checkpoint_dir, algo_class.__name__)

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    if args.mode == TRAIN_MODE:
        run_train(args, algo_class, train_loader, val_loader, algo_conf, checkpoint_dir)

    elif args.mode == TEST_MODE:
        run_test(args, algo_class, test_loader, algo_conf, checkpoint_dir)

def run_train(args, algo_class, train_loader, val_loader, algo_conf, checkpoint_dir="checkpoints"):
    # define algo_obj for manage training and validating strategies
    algo_mgr = algo_class(**algo_conf)

    # inits
    best_val_queacc = 0.0
    csv_path = f"{LOG_DIR}/{algo_class.__name__}_train_log.csv"
    header = ["Step",
              "Pre_Sup_Loss", "Pre_Que_Loss", "Pre_Sup_Acc", "Pre_Que_Acc",
    meta_loss_list = []
    loss_csv = f"{LOG_DIR}/{algo_class.__name__}_meta_loss.csv"
    header = ["step", "meta_loss"]
    val_iter = iter(val_loader)

    # training + validation loop
    train_pbar = tqdm(train_loader, desc="Training", position=1, leave=True)
    for id_, batch in enumerate(train_pbar):
        meta_loss = train_on_metabatch(algo_mgr, batch)

        train_pbar.set_postfix({"Meta Loss": f"{meta_loss:.4f}"})
        log_to_csv(loss_csv, [id_, meta_loss.detach().cpu().numpy()], header=header)

        if id_ % VAL_AFTER == 0 or id_ == len(train_loader) - 1:
            val_boT = next(val_iter)
            val_pbar = tqdm(val_boT, desc="Validating", position=0, leave=False)
            pre_valres, post_valres = val_on_metabatch(algo_mgr, val_pbar)

            # close val bar (remove from screen)
            val_pbar.close()

            # print validation results (Must be printed by tqdm.write to avoid interference with progress bars)
            val_result_str = f"""[Step {id_}] Validation Results
            Pre-update: Sup Loss: {pre_valres["sup_loss"]:.4f}, Que Loss: {pre_valres["que_loss"]:.4f}, Sup Acc: {pre_valres["sup_acc"]:.4f}, Que Acc: {pre_valres["que_acc"]:.4f}
            Post-update: Sup Loss: {post_valres["sup_loss"]:.4f}, Que Loss: {post_valres["que_loss"]:.4f}, Sup Acc: {post_valres["sup_acc"]:.4f}, Que Acc: {post_valres["que_acc"]:.4f}
            """
            tqdm.write(val_result_str)
            log_to_csv(
                csv_path, 
                [id_,
                 pre_valres["sup_loss"], pre_valres["que_loss"], pre_valres["sup_acc"], pre_valres["que_acc"],
                 post_valres["sup_loss"], post_valres["que_loss"], post_valres["sup_acc"], post_valres["que_acc"]],
                header=header,)

            # save checkpoint if best
            if post_valres["que_acc"] > best_val_queacc:
                best_val_queacc = post_valres["que_acc"]
                checkpoint_path = os.path.join(checkpoint_dir, f"best_checkpoint.pt")
                torch.save(algo_mgr.dump_state(), checkpoint_path)

    # save last checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"last_checkpoint.pt")
    torch.save(algo_mgr.dump_state(), checkpoint_path)

def run_test(args, algo_class, test_loader, algo_conf, checkpoint_dir="checkpoints"):
    # define algo_obj for manage training and validating strategies
    algo_mgr = algo_class(**algo_conf)

    # load checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"last_checkpoint.pt")
    print("Loading checkpoint from", checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please run training first.")
    algo_mgr.read_file(checkpoint_path)

    # test on whole test set
    all_sup_losses, all_que_losses, all_sup_accs, all_que_accs = test_on_wholeset(algo_mgr, test_loader)

    # compute stats
    num_steps = len(all_sup_losses)
    total_tasks = len(all_sup_losses[0])

    metrics_dict = {
        "SUPPORT LOSS": all_sup_losses,
        "QUERY LOSS": all_que_losses,
        "SUPPORT ACCURACY": all_sup_accs,
        "QUERY ACCURACY": all_que_accs
    }

    # print and save
    print_n_log_test(metrics_dict, num_steps, total_tasks)

############################################################################################
### Helper Funcs
############################################################################################


def train_on_metabatch(algo_mgr, boT):
    sup_x, sup_y, que_x, que_y = boT_to_stack(boT) # stack of meta_batch_size tasks
    return algo_mgr.train(sup_x, sup_y, que_x, que_y)

def val_on_metabatch(algo_mgr, boT):
    sup_x, sup_y, que_x, que_y = boT_to_stack(boT) # stack of meta_batch_size tasks
    return algo_mgr.val(sup_x, sup_y, que_x, que_y)

def test_on_wholeset(algo_mgr, test_loader):
    all_sup_losses = defaultdict(list)
    all_que_losses = defaultdict(list)
    all_sup_accs = defaultdict(list)
    all_que_accs = defaultdict(list)

    test_pbar = tqdm(test_loader, desc="Testing", leave=True)
    # with torch.no_grad():
    for boT in test_pbar:
        # fast-adaptation for each task in meta-batch
        sup_x, sup_y, que_x, que_y = boT_to_stack(boT) # stack of meta_batch_size tasks
        sup_losses, que_losses, sup_accs, que_accs = algo_mgr.test(sup_x, sup_y, que_x, que_y)
        
        # store results
        for step in range(len(sup_losses)):
            all_sup_losses[step].extend(sup_losses[step].detach().cpu().numpy())
            all_que_losses[step].extend(que_losses[step].detach().cpu().numpy())
            all_sup_accs[step].extend(sup_accs[step].detach().cpu().numpy())
            all_que_accs[step].extend(que_accs[step].detach().cpu().numpy())

    return all_sup_losses, all_que_losses, all_sup_accs, all_que_accs   

def print_n_log_test(metrics_dict, num_steps, total_tasks):
    print("\n\n" + "#" * 70)
    print(f"META-TESTING RESULTS OVER {total_tasks} TASKS")
    print("#" * 70)

    final_results = {}

    for metric_name, data_by_step in metrics_dict.items():
        print(f"\n\n{'='*60}")
        print(f" TABLE: {metric_name} ")
        print(f"{'='*60}")
        print(f"{'Step':<15} | {'Mean':<10} | {'Std':<10} | {'CI 95%':<10}")
        print(f"{'-'*60}")
        
        csv_filename = f"test_{metric_name.replace(' ', '_')}.csv"
        csv_path = os.path.join(LOG_DIR, csv_filename)
        if os.path.exists(csv_path):
            os.remove(csv_path)
        csv_header = ["Step", "Mean", "Std", "CI_95"]

        for step in range(num_steps):
            mean, std, ci95 = compute_stats(data_by_step[step])
            
            step_label = "Pre-update" if step == 0 else f"Update {step}"
            
            print(f"{step_label:<15} | {mean:<10.4f} | {std:<10.4f} | ± {ci95:<10.4f}")

            log_row = [step_label, mean, std, ci95]
            log_to_csv(csv_path, log_row, header=csv_header)
            
    print(f"\n{'='*60}\n")
