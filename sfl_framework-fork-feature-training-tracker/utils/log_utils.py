"""Verbosity-controlled print wrapper for USFL experiments."""

import os

_VERBOSITY: int = int(os.environ.get("USFL_VERBOSITY", "1"))
TQDM_DISABLED: bool = _VERBOSITY < 2


def vprint(message, level: int = 1):
    """Print message if current verbosity >= level.

    Levels:
        0: Errors, final results (always shown)
        1: Important progress info (default)
        2: Debug, per-round details, data stats
    """
    if _VERBOSITY >= level:
        print(message)
