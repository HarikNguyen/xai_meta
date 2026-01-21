from scripts import run
import argparse


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Script for running the model.")
    
    # Add validate and world_size arguments
    parser.add_argument(
        '--validate', 
        type=bool, 
        default=False, 
        action='store_true',
        help='Flag to indicate whether to validate the model.',
    )
    parser.add_argument(
        '--world_size', 
        type=int, 
        default=1, 
        help='Size of the world for distributed training.',
    )
    
    return parser.parse_args()


def main():
    # Get arguments from the command line
    args = parse_args()

    # Call the run function with the arguments
    run(args.validate, args.world_size)


if __name__ == "__main__":
    main()
