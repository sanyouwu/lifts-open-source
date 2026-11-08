"""Compatibility import for the historical LIFTS state-space model path."""

import sys

from . import lifts_ssm_joint

# Keep historical imports and pickled class names resolvable.
sys.modules[__name__] = lifts_ssm_joint
