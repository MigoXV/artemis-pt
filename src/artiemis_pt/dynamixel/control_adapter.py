from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from .contronller import DualDynamixelController, DynamixelConfig

COUNTS_PER_REV = 4096
PITCH_RANGE_RAD = math.pi / 2.0
YAW_ZERO_POSITION = 1024
PITCH_ZERO_POSITION = 1024
PITCH_INDEX = 0
YAW_INDEX = 1
PULSE_PER_RAD = COUNTS_PER_REV / (2.0 * math.pi)
RAD_PER_PULSE = (2.0 * math.pi) / COUNTS_PER_REV
HOME_TARGET_RADIANS = np.array([0.0, 0.0], dtype=np.float64)


def _normalize_yaw_rad(yaw_rad: float) -> float:
    return ((yaw_rad + math.pi) % (2.0 * math.pi)) - math.pi


def _validate_target_radians(target_radians: np.ndarray) -> np.ndarray:
    radians = np.asarray(target_radians, dtype=np.float64)
    if radians.shape != (2,):
        raise ValueError(f"expected target_radians shape (2,), got {radians.shape}")
    return radians


def pitch_rad_to_position(pitch_rad: float) -> int:
    if not 0.0 <= pitch_rad <= PITCH_RANGE_RAD:
        raise ValueError(f"pitch must be within [0, pi/2], got {pitch_rad}")
    return int(round(PITCH_ZERO_POSITION + pitch_rad * PULSE_PER_RAD))


def yaw_rad_to_position(yaw_rad: float) -> int:
    normalized_yaw = _normalize_yaw_rad(yaw_rad)
    return int(round(YAW_ZERO_POSITION + normalized_yaw * PULSE_PER_RAD)) % COUNTS_PER_REV


def position_to_pitch_rad(position: int) -> float:
    return (int(position) - PITCH_ZERO_POSITION) * RAD_PER_PULSE


def position_to_yaw_rad(position: int) -> float:
    wrapped_position = int(position) % COUNTS_PER_REV
    yaw_rad = (wrapped_position - YAW_ZERO_POSITION) * RAD_PER_PULSE
    return _normalize_yaw_rad(yaw_rad)


def pitch_yaw_rad_to_positions(target_radians: np.ndarray) -> np.ndarray:
    radians = _validate_target_radians(target_radians)
    return np.array(
        [
            pitch_rad_to_position(float(radians[PITCH_INDEX])),
            yaw_rad_to_position(float(radians[YAW_INDEX])),
        ],
        dtype=np.int64,
    )


def positions_to_pitch_yaw_rad(positions: np.ndarray) -> np.ndarray:
    motor_positions = np.asarray(positions, dtype=np.int64)
    if motor_positions.shape != (2,):
        raise ValueError(f"expected positions shape (2,), got {motor_positions.shape}")

    return np.array(
        [
            position_to_pitch_rad(int(motor_positions[PITCH_INDEX])),
            position_to_yaw_rad(int(motor_positions[YAW_INDEX])),
        ],
        dtype=np.float64,
    )


def pitch_position_out_of_range(position: int) -> bool:
    pitch_rad = position_to_pitch_rad(position)
    return pitch_rad < 0.0 or pitch_rad > PITCH_RANGE_RAD


def velocity_to_pitch_yaw_rad_per_sec(velocities: np.ndarray) -> np.ndarray:
    motor_velocities = np.asarray(velocities, dtype=np.float64)
    if motor_velocities.shape != (2,):
        raise ValueError(f"expected velocities shape (2,), got {motor_velocities.shape}")

    return motor_velocities * RAD_PER_PULSE


def telemetry_to_pitch_yaw_feedback(raw_telemetry: np.ndarray) -> tuple[np.ndarray, bool]:
    telemetry = np.asarray(raw_telemetry, dtype=np.float64)
    if telemetry.shape != (2, 5):
        raise ValueError(f"expected raw_telemetry shape (2, 5), got {telemetry.shape}")

    feedback = telemetry.copy()
    feedback[:, 0] = positions_to_pitch_yaw_rad(telemetry[:, 0].astype(np.int64))
    feedback[:, 1] = velocity_to_pitch_yaw_rad_per_sec(telemetry[:, 1])
    pitch_out_of_range = pitch_position_out_of_range(int(telemetry[PITCH_INDEX, 0]))
    return feedback, pitch_out_of_range


def rad_command_stream_to_position_stream(
    rad_command_stream: Iterable[np.ndarray],
) -> Iterator[np.ndarray]:
    for target_radians in rad_command_stream:
        yield pitch_yaw_rad_to_positions(target_radians)


@dataclass(frozen=True)
class PitchYawCommand:
    pitch: float
    yaw: float

    def as_array(self) -> np.ndarray:
        return np.array([self.pitch, self.yaw], dtype=np.float64)


@dataclass(frozen=True)
class PitchYawFeedback:
    state_radians: np.ndarray
    telemetry: np.ndarray
    pitch_out_of_range: bool


class PitchYawControlAdapter:
    """Accept pitch/yaw radians and forward motor position commands to the controller."""

    def __init__(self, config: DynamixelConfig | None = None):
        self.controller = DualDynamixelController(config)

    def open(self) -> None:
        self.controller.open()

    def close(self) -> None:
        self.controller.close()

    def __enter__(self) -> "PitchYawControlAdapter":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _coerce_target(self, target: np.ndarray | PitchYawCommand) -> np.ndarray:
        command = target.as_array() if isinstance(target, PitchYawCommand) else target
        return _validate_target_radians(command)

    def write_target(self, target: np.ndarray | PitchYawCommand) -> None:
        command = self._coerce_target(target)
        self.controller.write_positions(pitch_yaw_rad_to_positions(command))

    def write_target_and_read_feedback(
        self, target: np.ndarray | PitchYawCommand
    ) -> PitchYawFeedback:
        command = self._coerce_target(target)
        raw_telemetry = self.controller.write_and_read(pitch_yaw_rad_to_positions(command))
        telemetry, pitch_out_of_range = telemetry_to_pitch_yaw_feedback(raw_telemetry)
        return PitchYawFeedback(
            state_radians=telemetry[:, 0].copy(),
            telemetry=telemetry,
            pitch_out_of_range=pitch_out_of_range,
        )

    def read_feedback(self) -> PitchYawFeedback:
        raw_telemetry = self.controller.read_telemetry()
        telemetry, pitch_out_of_range = telemetry_to_pitch_yaw_feedback(raw_telemetry)
        return PitchYawFeedback(
            state_radians=telemetry[:, 0].copy(),
            telemetry=telemetry,
            pitch_out_of_range=pitch_out_of_range,
        )

    def stream_target_feedback(
        self, target_stream: Iterable[np.ndarray | PitchYawCommand]
    ) -> Iterator[PitchYawFeedback]:
        for target in target_stream:
            yield self.write_target_and_read_feedback(target)


__all__ = [
    "COUNTS_PER_REV",
    "PITCH_ZERO_POSITION",
    "YAW_ZERO_POSITION",
    "HOME_TARGET_RADIANS",
    "PitchYawCommand",
    "PitchYawFeedback",
    "PitchYawControlAdapter",
    "pitch_rad_to_position",
    "yaw_rad_to_position",
    "pitch_position_out_of_range",
    "position_to_pitch_rad",
    "position_to_yaw_rad",
    "pitch_yaw_rad_to_positions",
    "positions_to_pitch_yaw_rad",
    "telemetry_to_pitch_yaw_feedback",
    "velocity_to_pitch_yaw_rad_per_sec",
    "rad_command_stream_to_position_stream",
]
