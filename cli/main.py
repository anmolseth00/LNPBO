import argparse
import sys

from .checkpoint import add_checkpoint_command
from .encode import add_encode_command
from .register import add_register_command
from .suggest import add_suggest_command


def main():
    parser = argparse.ArgumentParser(
        prog="lnpbo",
        description="LNP Bayesian Optimization Toolkit",
    )

    subparsers = parser.add_subparsers(dest="command")

    add_encode_command(subparsers)
    add_suggest_command(subparsers)
    add_register_command(subparsers)
    add_checkpoint_command(subparsers)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)
