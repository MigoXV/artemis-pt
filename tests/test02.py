from __future__ import annotations

"""大角度转动测试"""
import argparse
import os
import time

import numpy as np

from artiemis_pt.dynamixel.control_loop import LatestValueControlLoop
from artiemis_pt.dynamixel.contronller import DynamixelConfig

YAW_DELTA_DEG = 100.0
STARTUP_DELAY_SEC = 2.0
OBSERVE_DURATION_SEC = 3.0
OBSERVE_INTERVAL_SEC = 0.2
YAW_DEG_ENV_VAR = "ARTEMIS_TEST02_YAW_DEG"


def format_array(value: np.ndarray | None) -> str:
    if value is None:
        return "None"
    return np.array2string(value, precision=4, suppress_small=True)


def normalize_yaw(yaw_rad: float) -> float:
    return ((yaw_rad + np.pi) % (2.0 * np.pi)) - np.pi


def validate_target_shape(command: np.ndarray) -> np.ndarray:
    target = np.asarray(command, dtype=np.float64)
    if target.shape != (2,):
        raise ValueError(f"expected command shape (2,), got {target.shape}")
    return target.copy()


def get_default_yaw_deg() -> float:
    raw_value = os.getenv(YAW_DEG_ENV_VAR)
    if raw_value is None:
        return YAW_DELTA_DEG

    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{YAW_DEG_ENV_VAR} must be a valid float, got {raw_value!r}"
        ) from exc


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


def build_yaw_turn_target(
    current_state: np.ndarray, yaw_delta_deg: float
) -> np.ndarray:
    state = validate_target_shape(current_state)
    yaw_delta_rad = np.deg2rad(yaw_delta_deg)
    target = np.array(
        [
            float(state[0]),
            normalize_yaw(float(state[1]) + yaw_delta_rad),
        ],
        dtype=np.float64,
    )
    return validate_target_shape(target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start the control loop, wait 2 seconds, then rotate yaw by 45 degrees."
    )
    parser.add_argument("--device", default="COM9")
    parser.add_argument("--baudrate", type=int, default=57600)
    parser.add_argument("--frequency", type=float, default=20.0)
    parser.add_argument("--max-errors", type=int, default=3)
    parser.add_argument("--startup-delay", type=float, default=STARTUP_DELAY_SEC)
    parser.add_argument(
        "--yaw-deg",
        type=float,
        default=get_default_yaw_deg(),
        help=(
            f"Yaw rotation in degrees. Defaults to env {YAW_DEG_ENV_VAR} "
            f"or {YAW_DELTA_DEG} if unset."
        ),
    )
    parser.add_argument("--observe-duration", type=float, default=OBSERVE_DURATION_SEC)
    parser.add_argument("--observe-interval", type=float, default=OBSERVE_INTERVAL_SEC)
    args = parser.parse_args()

    config = DynamixelConfig(
        device_name=args.device, baudrate=args.baudrate, dxl_ids=(1, 2)
    )
    loop = LatestValueControlLoop(
        config=config,
        period_sec=1.0 / args.frequency,
        max_consecutive_errors=args.max_errors,
    )

    loop.start()

    try:
        if not wait_for_initial_state(loop, timeout_sec=max(args.startup_delay, 1.0)):
            raise RuntimeError("failed to read initial motor state before timeout")

        print_state("[initial]", loop)

        time.sleep(args.startup_delay)

        state = loop.get_state()
        if state.latest_state is None:
            raise RuntimeError("latest_state is unavailable after startup delay")

        current_state = state.latest_state.copy()
        target = build_yaw_turn_target(current_state, args.yaw_deg)
        print(f"[command] current_state={format_array(current_state)}")
        print(
            f"[command] yaw += {args.yaw_deg:.2f} deg -> target={format_array(target)}"
        )
        loop.submit_target(target)

        observe_count = max(
            1, int(np.ceil(args.observe_duration / args.observe_interval))
        )
        for index in range(1, observe_count + 1):
            time.sleep(args.observe_interval)
            print_state(f"[observe {index}]", loop)
    finally:
        try:
            loop.stop()
        finally:
            print_state("[final]", loop)


if __name__ == "__main__":
    main()
