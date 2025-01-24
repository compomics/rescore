"""Entrypoint for MS²Rescore GUI."""

import contextlib
import multiprocessing
import os

from ms2rescore.gui.app import app


def main():
    """Entrypoint for MS²Rescore GUI."""
    multiprocessing.freeze_support()
    # Redirect stdout when running GUI (packaged app might not have console attached)
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        app()


if __name__ == "__main__":
    main()
