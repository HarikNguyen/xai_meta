import os
import numpy as np
import math
import torch
from tqdm import tqdm

from .warm_up import warm_up
from .utils import log_to_csv
from algos.maml import MAML

from loaders.utils import boT_to_stack


############################################################################################
### Main Func
############################################################################################

VAL_AFTER = 1000
TRAIN_MODE = "train"
VAL_MODE = "val"
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

def run_train(args, algo_class, train_loader, val_loader, algo_conf, checkpoint_dir="checkpoints"):
    # define algo_obj for manage training and validating strategies
    algo_mgr = algo_class(**algo_conf)

    # inits
    best_val_queacc = 0.0
    csv_path = f"{LOG_DIR}/{algo_class.__name__}_train_log.csv"
    header = ["Step",
              "Pre_Sup_Loss", "Pre_Que_Loss", "Pre_Sup_Acc", "Pre_Que_Acc",
              "Post_Sup_Loss", "Post_Que_Loss", "Post_Sup_Acc", "Post_Que_Acc"]
    val_iter = iter(val_loader)

    # training + validation loop
    train_pbar = tqdm(train_loader, desc="Training", position=1, leave=True)
    for id_, batch in enumerate(train_pbar):
        meta_loss = train_on_metabatch(algo_mgr, batch)

        train_pbar.set_postfix({"Meta Loss": f"{meta_loss:.4f}"})

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
                checkpoint_path = os.path.join(checkpoint_dir, f"best_checkpoint_step_{id_}.pt")
                torch.save(algo_mgr.state_dict(), checkpoint_path)

    # save last checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"last_checkpoint_step_{id_}.pt")
    torch.save(algo_mgr.state_dict(), checkpoint_path)


############################################################################################
### Helper Funcs
############################################################################################


def train_on_metabatch(algo_mgr, boT):
    sup_x, sup_y, que_x, que_y = boT_to_stack(boT) # stack of meta_batch_size tasks
    return algo_mgr.train(sup_x, sup_y, que_x, que_y)

def val_on_metabatch(algo_mgr, boT):
    sup_x, sup_y, que_x, que_y = boT_to_stack(boT) # stack of meta_batch_size tasks
    return algo_mgr.val(sup_x, sup_y, que_x, que_y)

def test_on_wholeset(algo_mgr, iter_loader):
    pass
