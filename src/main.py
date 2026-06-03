from scripts import run
import argparse


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

    # Return args
    return parser.parse_args()


def main():
    # Get arguments from the command line
    args = parse_args()

    # Call the run function with the arguments
    run(args)


if __name__ == "__main__":
    main()
