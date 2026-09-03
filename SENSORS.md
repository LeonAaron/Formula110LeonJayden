# Formula 110 Sensor Reference

# IMPORTANT COMMAND
uv run python scripts/export_student_controllers.py controllers.reactive2   

A controller receives one immutable `RobotSensors` snapshot on every simulation
tick:

```python
from racing import RobotCommand, RobotSensors


def control(sensors: RobotSensors) -> RobotCommand:
    speed = sensors.odometry.speed_mps
    front_wall = sensors.wall_lidar.front_m
    return RobotCommand(throttle=0.2, steer=0.0)
```

This reference describes the snapshots produced by the simulator, rather than
every value that could be passed manually to the public dataclass constructors.
All measurements use SI units unless the field name explicitly says
`degrees`. Numeric fields are Python `float` values and tuples are immutable.

## Range notation

- `[a, b]` includes both endpoints; `(a, b)` excludes them.
- `unbounded` means the simulator does not clamp the field to a documented
  numeric interval. Normal physics produces finite values, but controllers
  should choose suitable clipping and normalization for their own models.
- The only intentional infinite values are LiDAR no-hit readings. The simulator
  does not intentionally emit `NaN` sensor values.

## Snapshot overview

| Attribute | Python type | Contents |
| --- | --- | --- |
| `sensors.dt_s` | `float` | Time since the previous control snapshot |
| `sensors.tick` | `int` | Zero-based control tick number |
| `sensors.imu` | `ImuSensors` | Orientation, turn rate, and acceleration |
| `sensors.odometry` | `OdometrySensors` | Signed speed and accumulated travel distance |
| `sensors.lidar` | `LidarSensors` | Ranges to walls, cars, and other blockers |
| `sensors.wall_lidar` | `LidarSensors` | Ranges to track barriers only |
| `sensors.camera` | `CameraSensors` | Processed track geometry and opponent readings |
| `sensors.contact` | `ContactSensors` | Contact durations and accumulated damage |

The standard simulator timestep is `1 / 60` second, so controllers normally
receive 60 snapshots per simulated second.

## Timing

| Field | Type | Simulator-produced range | Meaning |
| --- | --- | --- | --- |
| `dt_s` | `float` | `[0.0, +inf)`; normally `1 / 60` | Simulation seconds represented by this tick. Supported game and race entrypoints require a positive configured timestep; the sensor builder defensively clamps a negative value to `0.0`. |
| `tick` | `int` | `[0, +inf)` | Starts at `0` for a new car/controller run and increases by one per snapshot. |

The first snapshot has `tick == 0`. Because no earlier measurement exists, its
derived yaw rate and acceleration values are `0.0`.

## `sensors.imu`: `ImuSensors`

| Field | Type | Unit | Range | Sign and meaning |
| --- | --- | --- | --- | --- |
| `heading_degrees` | `float` | degrees | Circular; not explicitly clamped | World yaw. `0.0` points along world `+Z`; positive turns toward the car's right. Values differing by `360` degrees represent the same direction. |
| `yaw_rate_degrees_per_s` | `float` | degrees/second | Unbounded | Positive means rotating right. Computed from the wrapped heading change divided by `dt_s`; `0.0` on the first tick or when `dt_s == 0.0`. |
| `pitch_degrees` | `float` | degrees | Circular; not explicitly clamped | Vehicle nose-up/nose-down body orientation. |
| `roll_degrees` | `float` | degrees | Circular; not explicitly clamped | Side-to-side vehicle body orientation. |
| `forward_acceleration_mps2` | `float` | m/s² | Unbounded | Change in signed forward speed divided by `dt_s`. Positive means acceleration in the forward direction. `0.0` on the first tick or when `dt_s == 0.0`. |
| `lateral_acceleration_mps2` | `float` | m/s² | Unbounded | Estimated as signed forward speed times yaw rate in radians/second. Positive points toward the car's right. |

Heading, pitch, and roll are angles rather than linear values. For an MLP,
encoding a circular angle as its sine and cosine avoids a discontinuity at the
wrap boundary.

## `sensors.odometry`: `OdometrySensors`

| Field | Type | Unit | Range | Meaning |
| --- | --- | --- | --- | --- |
| `speed_mps` | `float` | m/s | Unbounded signed value | Forward speed reported by the vehicle physics. Positive is forward and negative is reverse. |
| `distance_m` | `float` | m | `[0.0, +inf)` | Accumulated absolute distance traveled since the controller run began. It never decreases during a run and is not official lap progress. |

At each tick, odometry distance increases by approximately
`abs(speed_mps) * dt_s`. Driving backward therefore still increases
`distance_m`.

## `sensors.lidar` and `sensors.wall_lidar`: `LidarSensors`

Both LiDAR groups have the same fields and beam layout:

| Field/property | Type | Unit | Default/range | Meaning |
| --- | --- | --- | --- | --- |
| `angles_degrees` | `tuple[float, ...]` | degrees | Default `(-90, -45, -20, 0, 20, 45, 90)` | Beam directions relative to the car. Negative is left, `0` is forward, and positive is right. |
| `distances_m` | `tuple[float, ...]` | m | One `[0.0, 1000.0]` hit distance or `math.inf` no-hit value per angle | Beam results aligned by index with `angles_degrees`. |
| `max_distance_m` | `float` | m | `math.inf` in standard snapshots | The public no-hit sentinel and configured maximum range. |
| `front_m` | `float` property | m | Same as a distance reading | Reading whose configured angle is nearest `0` degrees. |
| `front_left_m` | `float` property | m | Same as a distance reading | Reading nearest `-20` degrees. |
| `front_right_m` | `float` property | m | Same as a distance reading | Reading nearest `20` degrees. |
| `left_m` | `float` property | m | Same as a distance reading | Reading nearest `-90` degrees. |
| `right_m` | `float` property | m | Same as a distance reading | Reading nearest `90` degrees. |
| `distance_at_angle_degrees(angle)` | method returning `float` | m | Same as a distance reading | Returns the configured beam nearest the requested angle; it does not interpolate between beams. |

The standard infinite-range configuration uses a 1,000-meter physics ray. A
hit therefore returns a value from `0.0` through `1000.0`; no accepted hit
returns `math.inf`. Distances are measured from a ray origin approximately
`0.62` meters ahead of the car's center, not from the center itself.

The two groups differ only in what counts as a hit:

- `sensors.lidar` detects track barriers, other cars, blockers, and other
  non-ignored collision geometry. It ignores the sensing car and the shared
  grass/track floor.
- `sensors.wall_lidar` filters readings to track barriers only. It is useful for
  learning the track boundary without treating another car as a wall.

Do not feed `math.inf` directly into most ML models. A common preprocessing
choice is to replace it with a chosen sensor cap and then scale all ranges to
`[0.0, 1.0]`. Human demonstration JSON writes infinite LiDAR values as JSON
`null`.

## `sensors.camera`: `CameraSensors`

These are processed geometric readings, not raw pixels.

| Field | Type | Unit | Range/default | Meaning |
| --- | --- | --- | --- | --- |
| `visible` | `bool` | — | Currently always `True` in simulator-generated snapshots | Whether processed track-center readings are available. Controllers should still respect the flag for compatibility. |
| `center_offset_m` | `float` | m | Unbounded signed value | Lateral component from the car to the nearest track-center point. Negative is left of the car and positive is right. |
| `heading_error_degrees` | `float` | degrees | `[-180.0, 180.0)` | Smallest signed turn from the car's heading to the local track heading. Negative is left and positive is right. |
| `lookahead_offsets_m` | `tuple[float, ...]` | m | Three unbounded signed values by default | Lateral components from the car to future track-center points. Negative is left and positive is right. |
| `lookahead_distances_m` | `tuple[float, ...]` | m | Default `(4.0, 9.0, 16.0)` | Distances ahead along the track centerline used to select the corresponding lookahead points. |
| `competitors` | `tuple[CameraCompetitorReading, ...]` | — | Length `[0, 3]` | Up to the three nearest non-eliminated opponent cars with a track-barrier-clear line of sight, sorted nearest first. |

`lookahead_offsets_m[i]` corresponds to `lookahead_distances_m[i]`. The offsets
are measured in the car's current left/right coordinate frame, so they reflect
both upcoming curvature and the car's current pose.

### Each `CameraCompetitorReading`

| Field | Type | Unit | Range | Meaning |
| --- | --- | --- | --- | --- |
| `distance_m` | `float` | m | `(0.05, +inf)` | Horizontal straight-line distance between car centers. Cars at or below `0.05` meters are omitted. |
| `angle_degrees` | `float` | degrees | `[-180.0, 180.0]` | Bearing from this car's forward direction to the opponent. Negative is left, `0.0` is ahead, and positive is right; values near either 180-degree endpoint are behind. |
| `relative_heading_degrees` | `float` | degrees | `[-180.0, 180.0)` | Smallest signed rotation from this car's heading to the opponent's heading. |
| `speed_mps` | `float` | m/s | Unbounded signed value | Opponent's forward speed; negative means it is reversing. |
| `closing_speed_mps` | `float` | m/s | Unbounded signed value | This car's signed forward speed minus the opponent's signed forward speed. Positive means this car is gaining by that simplified speed comparison. |

Competitor readings are not restricted to an ahead-only camera cone or a fixed
maximum distance. A car behind may therefore appear with an angle near
`-180` or `180` degrees. Track barriers can occlude an opponent.

## `sensors.contact`: `ContactSensors`

| Field | Type | Unit | Range | Meaning |
| --- | --- | --- | --- | --- |
| `wall` | `float` | seconds | `[0.0, +inf)` | Duration of uninterrupted current contact with a track barrier. Resets to `0.0` when wall contact ends. |
| `robot` | `float` | seconds | `[0.0, +inf)` | Duration of uninterrupted current contact with another robot or race blocker. Resets to `0.0` when that contact ends. |
| `any_contact` | `float` | seconds | `[0.0, +inf)` | Current contact duration, at least as large as `wall` and `robot`. In simulator snapshots it is the greater of those two durations. |
| `damage` | `float` | normalized | `[0.0, 1.0]` | Accumulated collision damage. `0.0` is undamaged and `1.0` means fully damaged/eliminated. |

Contact durations usually change in increments of `dt_s`. Use comparisons such
as `sensors.contact.wall > 0.0` to test whether contact is active.

## Compact type outline

```python
RobotSensors(
    dt_s: float,
    tick: int,
    imu: ImuSensors,
    odometry: OdometrySensors,
    lidar: LidarSensors,
    wall_lidar: LidarSensors,
    camera: CameraSensors,
    contact: ContactSensors,
)
```

The public dataclass definitions live in
[`src/racing/student/api.
py`](src/racing/student/api.py), and the simulator-side
measurement calculations live in
[`src/racing/race/sensors.py`](src/racing/race/sensors.py).