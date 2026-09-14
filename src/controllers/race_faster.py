"""Improved submission module: the Apex map-localized racing-line controller.

Re-exports ``controllers.apex`` under the name the autograder requires. The
runtime prefers the ``create_controller`` factory (fresh state per car and
race); ``control`` is kept for callers that only know the function interface.
"""

from controllers.apex import RACING_COLOR, RACING_NAME, Controller, create_controller
from racing import RobotCommand, RobotSensors

__all__ = ["RACING_COLOR", "RACING_NAME", "control", "create_controller"]

_shared: Controller | None = None


def control(sensors: RobotSensors) -> RobotCommand:
    """Function-interface entry point backed by one lazily created controller."""
    global _shared
    if _shared is None or sensors.tick == 0:
        _shared = create_controller()
    return _shared(sensors)
