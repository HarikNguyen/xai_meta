import yaml
import argparse

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
        choices=["train", "val", "test"],
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
        default="configs/mini2cub.yaml",
        type=str,
        help="Path to configuration file. Default: configs/mini2cub.yaml",
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

    # Call the run function with the arguments
    run(args)


if __name__ == "__main__":
    main()
