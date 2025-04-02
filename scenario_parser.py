import argparse


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Select traffic scenario (1, 2, or 3)")
    return parser.parse_args()
