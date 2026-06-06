import os
import numpy as np
import torch
from tqdm import tqdm

from .warm_up import warm_up
from .trains import run_train
from .tests import run_test
from algos.maml import MAML
from loaders.utils import boT_to_stack


############################################################################################
### Main Func
############################################################################################

VAL_AFTER = 1000
TRAIN_MODE = "train"
TEST_MODE = "test"

def run(args):
    if args.algo == "maml":
        algo_class = MAML
    else:
        raise NotImplementedError(f"Algorithm {algo} not implemented.")

    # warm up
    train_loader, val_loader, test_loader, algo_conf = warm_up(args.yaml_config)
    algo_conf["vmap_chunk_size"] = args.vmap_chunk_size

    checkpoint_dir = args.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if not os.path.exists(os.path.join(checkpoint_dir, algo_class.__name__)):
        os.makedirs(os.path.join(checkpoint_dir, algo_class.__name__))

    checkpoint_dir = os.path.join(checkpoint_dir, algo_class.__name__)
    print("Checkpoint directory:", checkpoint_dir)

    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    # run
    if args.mode == TRAIN_MODE:
        run_train(args, algo_class, train_loader, val_loader, algo_conf, checkpoint_dir, log_dir)

    elif args.mode == TEST_MODE:

        run_test(args, algo_class, test_loader, algo_conf, args.use_best, args.use_last, checkpoint_dir, log_dir)
