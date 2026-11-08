"""Compatibility import for the historical LIFTS wrapper path."""

import sys

if __name__ == "__main__":
    if not __package__:
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from baselines.lifts_wrapper import _run_demo

    _run_demo()
else:
    from . import lifts_wrapper

    # Keep historical imports and module-level patches pointed at the same code.
    sys.modules[__name__] = lifts_wrapper
