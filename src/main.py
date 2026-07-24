import yaml
import argparse

import torch

from scripts import run


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Script for running the model.")

    # Add arguments
    parser.add_argument(
        "--mode",
        default="train",
        type=str,
        help="Mode to run the script in.\n - train: start meta-training.\n - val: meta-testing on validation set.\n - test: meta-testing on test set.\nDefault: train",
        choices=["train", "val", "test", "explain", "check_explain"],
    )

    parser.add_argument(
        "--algo",
        default="maml",
        type=str,
        help="Algorithm to use. Default: MAML",
        choices=["maml",],
    )

    parser.add_argument(
        "--config",
        default="configs/mini2cub/conv4.yaml",
        type=str,
        help="Path to configuration file. Default: configs/mini2cub/conv4.yaml",
    )

    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints",
        type=str,
        help="Directory to save checkpoints. Default: checkpoints",
    )

    parser.add_argument(
        "--vmap_chunk_size",
        default=None,
        type=int,
        help="Chunk size for vmap. Default: Equal to meta_batch_size",
    )

    parser.add_argument(
        "--log_dir",
        default="logs",
        type=str,
        help="Directory to save logs. Default: logs",
    )

    parser.add_argument(
        "--use_best",
        action="store_true",
        help="Use the best checkpoint from the checkpoint directory. Default: False",
    )

    parser.add_argument(
        "--use_last",
        action="store_true",
        help="Use the last checkpoint from the checkpoint directory. Default: False",
    )

    parser.add_argument(
        "--check_method",
        default="biADT",
        type=str,
        help="Method to check the explaination.",
    )

    parser.add_argument(
        "--flip_ratio",
        default=None,
        type=float,
        help="Flip ratio for the noisy label. Default: None\n(no noise + No use for explaination)",
    )

    parser.add_argument(
        "--blur",
        action="store_true",
        help="Blur the sup_x while explaining. Default: False",
    )
    
    # Return args
    return parser.parse_args()

def load_config(yaml_path):
    """Load configuration from YAML file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Get arguments from the command line
    args = parse_args()

    # Load config from YAML file
    yaml_config = load_config(args.config)
    args.yaml_config = yaml_config

    # n_way/k_shot/k_query/image size are fixed for the whole run, so cuDNN can
    # safely benchmark and cache the fastest conv algorithm per shape; TF32
    # lets matmul/conv use Tensor Cores on Ampere+/Ada GPUs at negligible
    # precision cost.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Call the run function with the arguments
    run(args)


if __name__ == "__main__":
    main()
