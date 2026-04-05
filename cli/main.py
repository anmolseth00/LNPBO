"""Entry point for the ``lnpbo`` command-line interface."""

import argparse
import sys

from .checkpoint import add_checkpoint_command
from .encode import add_encode_command
from .propose_ils import add_propose_ils_command
from .register import add_register_command
from .suggest import add_suggest_command


def main():
    """Parse CLI arguments and dispatch to the selected subcommand.

    Registers all available subcommands (encode, suggest, propose-ils,
    register, checkpoint), then invokes the handler bound to the chosen
    subcommand.  Prints help and exits with code 1 if no subcommand is
    provided.
    """
    parser = argparse.ArgumentParser(
        prog="lnpbo",
        description="LNP Bayesian Optimization Toolkit",
    )

    subparsers = parser.add_subparsers(dest="command")

    add_encode_command(subparsers)
    add_suggest_command(subparsers)
    add_propose_ils_command(subparsers)
    add_register_command(subparsers)
    add_checkpoint_command(subparsers)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
