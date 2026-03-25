from __future__ import annotations

"""连续平滑转动"""
import argparse
import threading
import time
from pathlib import Path

import numpy as np

from artiemis_pt.dynamixel.control_adapter import PITCH_RANGE_RAD
from artiemis_pt.dynamixel.control_loop import LatestValueControlLoop
from artiemis_pt.dynamixel.contronller import DynamixelConfig


def format_array(value: np.ndarray | None) -> str:
    if value is None:
        return "None"
    return np.array2string(value, precision=4, suppress_small=True)


def validate_command(command: np.ndarray) -> np.ndarray:
    target = np.asarray(command, dtype=np.float64)
    if target.shape != (2,):
        raise ValueError(f"expected command shape (2,), got {target.shape}")

    pitch = float(target[0])
    yaw = float(target[1])
    if not 0.0 <= pitch <= PITCH_RANGE_RAD:
        raise ValueError(f"pitch must be within [0, pi/2], got {pitch}")
    if not -np.pi <= yaw < np.pi:
        raise ValueError(f"yaw must be within [-pi, pi), got {yaw}")

    return target.copy()


def build_demo_waypoints() -> list[np.ndarray]:
    commands = [
        np.array([0.25, 0.00], dtype=np.float64),
        np.array([0.25, -0.80], dtype=np.float64),
        np.array([0.50, 0.00], dtype=np.float64),
        np.array([0.90, 1.20], dtype=np.float64),
        np.array([1.20, 2.80], dtype=np.float64),
        np.array([1.00, 3.00], dtype=np.float64),
        np.array([0.00, 0.00], dtype=np.float64),
    ]
    return [validate_command(command) for command in commands]


def build_demo_commands(
    update_hz: float = 5.0,
    segment_duration_sec: float = 1.0,
) -> tuple[list[np.ndarray], float]:
    if update_hz <= 0.0:
        raise ValueError(f"update_hz must be > 0, got {update_hz}")
    if segment_duration_sec <= 0.0:
        raise ValueError(
            f"segment_duration_sec must be > 0, got {segment_duration_sec}"
        )

    waypoints = build_demo_waypoints()
    interval_sec = 1.0 / update_hz
    steps_per_segment = max(1, int(round(segment_duration_sec * update_hz)))
    commands = [waypoints[0].copy()]

    for start, end in zip(waypoints[:-1], waypoints[1:], strict=True):
        for step in range(1, steps_per_segment + 1):
            alpha = step / steps_per_segment
            command = ((1.0 - alpha) * start) + (alpha * end)
            commands.append(validate_command(command))

    return commands, interval_sec


def producer_worker(
    loop: LatestValueControlLoop,
    commands: list[np.ndarray],
    interval_sec: float,
) -> None:
    for index, command in enumerate(commands, start=1):
        loop.submit_target(command)
        print(f"[producer {index}] submit command={format_array(command)}")
        if index < len(commands):
            time.sleep(interval_sec)


def print_state(tag: str, loop: LatestValueControlLoop) -> None:
    state = loop.get_state()
    print(tag)
    print(f"latest_target : {format_array(state.latest_target)}")
    print(f"last_target   : {format_array(state.last_target)}")
    print(f"latest_state  : {format_array(state.latest_state)}")
    print(f"telemetry     : {format_array(state.latest_telemetry)}")
    print(f"pitch_oor     : {state.latest_pitch_out_of_range}")
    print(f"last_write_ok : {state.last_write_ok}")
    print(f"last_telem_ok : {state.last_telemetry_ok}")
    print(f"last_error    : {state.last_error}")
    print(f"faulted       : {state.is_faulted}")
    print(f"error_count   : {state.consecutive_error_count}")
    print(f"tick_count    : {state.tick_count}")
    print(f"last_tick     : {state.last_tick_time}")
    print(f"worker_error  : {state.worker_error}")
    print("-" * 60)


def wait_for_initial_state(
    loop: LatestValueControlLoop,
    timeout_sec: float,
    poll_interval_sec: float = 0.05,
) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        state = loop.get_state()
        if state.latest_state is not None:
            return True
        time.sleep(poll_interval_sec)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send one target per second to the real Dynamixel control loop."
    )
    parser.add_argument("--device", default="COM9")
    parser.add_argument("--baudrate", type=int, default=57600)
    parser.add_argument("--frequency", type=float, default=20.0)
    parser.add_argument("--update-hz", type=float, default=5.0)
    parser.add_argument("--segment-duration", type=float, default=1.0)
    parser.add_argument("--max-errors", type=int, default=3)
    args = parser.parse_args()

    config = DynamixelConfig(
        device_name=args.device, baudrate=args.baudrate, dxl_ids=(1, 2)
    )
    loop = LatestValueControlLoop(
        config=config,
        period_sec=1.0 / args.frequency,
        max_consecutive_errors=args.max_errors,
    )
    commands, interval_sec = build_demo_commands(
        update_hz=args.update_hz,
        segment_duration_sec=args.segment_duration,
    )

    producer = threading.Thread(
        target=producer_worker,
        args=(loop, commands, interval_sec),
        name="external-command-producer",
        daemon=False,
    )

    loop.start()
    if wait_for_initial_state(loop, timeout_sec=max(interval_sec, 1.0)):
        print_state("[initial]", loop)
    else:
        print_state("[initial-timeout]", loop)
    producer.start()

    try:
        for index in range(1, len(commands) + 1):
            time.sleep(interval_sec)
            print_state(f"[observe {index}]", loop)
    finally:
        producer.join()
        try:
            loop.stop()
        finally:
            print_state("[final]", loop)


if __name__ == "__main__":
    main()
