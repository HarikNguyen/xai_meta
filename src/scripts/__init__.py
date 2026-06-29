import os
import numpy as np
import torch
from tqdm import tqdm

from .warm_up import warm_up
from .train import run_train
from .test import run_test
from .explain import explain
from .check_explain import check_explain
from algos.maml import MAML
from loaders.utils import boT_to_stack


############################################################################################
### Main Func
############################################################################################

TRAIN_MODE = "train"
TEST_MODE = "test"
EXPLAIN_MODE = "explain"
CHECK_EXPLAIN_MODE = "check_explain"

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

    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # run
    if args.mode == TRAIN_MODE:
        run_train(args, algo_class, train_loader, val_loader, algo_conf, checkpoint_dir, log_dir)

    elif args.mode == TEST_MODE:
        run_test(args, algo_class, test_loader, algo_conf, args.use_best, args.use_last, checkpoint_dir, log_dir)

    elif args.mode == EXPLAIN_MODE:
        explain(args.algo, algo_class, test_loader, algo_conf, args.use_best, args.use_last, checkpoint_dir, log_dir)

    elif args.mode == CHECK_EXPLAIN_MODE:
        check_explain(args.algo, algo_class, test_loader, algo_conf, args.check_method, args.use_best, args.use_last, checkpoint_dir, log_dir)
