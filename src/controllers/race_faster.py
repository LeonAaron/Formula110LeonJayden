"""Improved submission module: the reactive controller.

Re-exports ``controllers.reactive`` under the name the autograder requires.
"""

from controllers.reactive import RACING_COLOR, RACING_NAME, control

__all__ = ["RACING_COLOR", "RACING_NAME", "control"]
