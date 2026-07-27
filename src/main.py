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
        help="train/val/test: chunk size for vmap (default: equal to meta_batch_size). "
             "explain/check_explain (no vmap there): overrides metatest_batch_size and "
             "the GPU worker/stream pool size, i.e. how many tasks run concurrently.",
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

    parser.add_argument(
        "--illustrate_label",
        default=None,
        type=str,
        help="check_explain only: if set (e.g. 'conv4', 'resnet10'), also save "
             "illustrative plots (masking grid for biADT, corrupted-layer grid "
             "for sanity_params, original-vs-perturbed grid for "
             "sanity_support_set) for the max-gain and min-gain tasks "
             "into check_explain_storage/<illustrate_label>/<method>/. "
             "Default: None (no illustration plots saved).",
    )

    parser.add_argument(
        "--illustrate_n_tasks",
        default=3,
        type=int,
        help="check_explain only: number of tasks to save illustration plots "
             "for when --illustrate_label is set. Default: 3",
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

    # fixed shapes -> cuDNN caches the fastest conv algo; TF32 uses Tensor Cores cheaply
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Call the run function with the arguments
    run(args)


if __name__ == "__main__":
    main()
