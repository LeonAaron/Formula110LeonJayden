"""Improved submission module: the v3 reactive controller.

Re-exports ``controllers.reactive_v3`` under the name the autograder requires.
"""

from controllers.reactive_v3 import RACING_COLOR, RACING_NAME, control

__all__ = ["RACING_COLOR", "RACING_NAME", "control"]
