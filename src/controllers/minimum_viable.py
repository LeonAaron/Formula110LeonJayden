"""Minimum-viable submission module: the neuroevolution controller.

Re-exports ``controllers.neuro`` under the name the autograder requires.
"""

from controllers.neuro import RACING_COLOR, RACING_NAME, control

__all__ = ["RACING_COLOR", "RACING_NAME", "control"]
