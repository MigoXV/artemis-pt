from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from .control_adapter import PitchYawControlAdapter, PitchYawFeedback
from .contronller import DynamixelConfig

CONTROL_FREQUENCY_HZ = 100.0
CONTROL_PERIOD_SEC = 1.0 / CONTROL_FREQUENCY_HZ
DEFAULT_MAX_CONSECUTIVE_ERRORS = 3
YAW_INDEX = 1
MAX_YAW_DIRECT_STEP_RAD = np.deg2rad(75.0)
YAW_SLIDING_WINDOW_STEP_RAD = np.deg2rad(60.0)


def _validate_target(target_rad: np.ndarray) -> np.ndarray:
    target = np.asarray(target_rad, dtype=np.float64)
    if target.shape != (2,):
        raise ValueError(f"expected target shape (2,), got {target.shape}")
    return target.copy()


def _normalize_yaw_rad(yaw_rad: float) -> float:
    return ((yaw_rad + np.pi) % (2.0 * np.pi)) - np.pi


def _shortest_yaw_delta_rad(current_yaw_rad: float, target_yaw_rad: float) -> float:
    return _normalize_yaw_rad(target_yaw_rad - current_yaw_rad)


def _build_windowed_target(target_rad: np.ndarray, present_rad: np.ndarray) -> np.ndarray:
    command = _validate_target(target_rad)
    present = _validate_target(present_rad)
    yaw_delta_rad = _shortest_yaw_delta_rad(
        float(present[YAW_INDEX]),
        float(command[YAW_INDEX]),
    )
    if abs(yaw_delta_rad) <= MAX_YAW_DIRECT_STEP_RAD:
        return command

    command[YAW_INDEX] = _normalize_yaw_rad(
        float(present[YAW_INDEX]) + np.sign(yaw_delta_rad) * YAW_SLIDING_WINDOW_STEP_RAD
    )
    return command


class ControlAdapterProtocol(Protocol):
    def open(self) -> None: ...

    def close(self) -> None: ...

    def write_target(self, target: np.ndarray) -> None: ...

    def read_feedback(self) -> PitchYawFeedback: ...


@dataclass(frozen=True)
class ControlLoopState:
    is_running: bool
    is_faulted: bool
    latest_target: Optional[np.ndarray]
    last_target: Optional[np.ndarray]
    latest_state: Optional[np.ndarray]
    latest_telemetry: Optional[np.ndarray]
    latest_pitch_out_of_range: Optional[bool]
    last_write_ok: Optional[bool]
    last_telemetry_ok: Optional[bool]
    last_error: Optional[str]
    consecutive_error_count: int
    tick_count: int
    last_tick_time: Optional[float]
    worker_error: Optional[str]


class LatestTargetProducer:
    """Thread-safe latest-value command producer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_target: Optional[np.ndarray] = None

    def submit_target(self, target_rad: np.ndarray) -> None:
        target = _validate_target(target_rad)
        with self._lock:
            self._latest_target = target

    def get_latest_target(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_target is None:
                return None
            return self._latest_target.copy()


class ControlLoopConsumer:
    """Thread-safe command consumer and feedback/state holder."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_target: Optional[np.ndarray] = None
        self._latest_state: Optional[np.ndarray] = None
        self._latest_telemetry: Optional[np.ndarray] = None
        self._latest_pitch_out_of_range: Optional[bool] = None
        self._last_write_ok: Optional[bool] = None
        self._last_telemetry_ok: Optional[bool] = None
        self._last_error: Optional[str] = None
        self._consecutive_error_count = 0
        self._is_faulted = False
        self._tick_count = 0
        self._last_tick_time: float | None = None

    def clear_fault(self) -> None:
        with self._lock:
            self._is_faulted = False

    def get_last_target(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._last_target is None:
                return None
            return self._last_target.copy()

    def get_latest_state(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_state is None:
                return None
            return self._latest_state.copy()

    def get_latest_telemetry(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_telemetry is None:
                return None
            return self._latest_telemetry.copy()

    def select_command(self, latest_target: Optional[np.ndarray]) -> Optional[np.ndarray]:
        with self._lock:
            if latest_target is not None:
                if self._latest_state is None:
                    command = latest_target.copy()
                else:
                    command = _build_windowed_target(latest_target, self._latest_state)
                self._last_target = command.copy()
                return command

            if self._last_target is not None:
                return self._last_target.copy()

            return None

    def update_feedback(
        self,
        present_radians: np.ndarray,
        telemetry: np.ndarray,
        pitch_out_of_range: bool,
    ) -> None:
        with self._lock:
            self._latest_state = np.asarray(present_radians, dtype=np.float64).copy()
            self._latest_telemetry = np.asarray(telemetry, dtype=np.float64).copy()
            self._latest_pitch_out_of_range = bool(pitch_out_of_range)

    def record_write_success(self) -> None:
        with self._lock:
            self._last_write_ok = True
            self._last_telemetry_ok = None
            self._last_error = None

    def record_telemetry_success(self, *, write_ok: Optional[bool]) -> None:
        with self._lock:
            self._last_write_ok = write_ok
            self._last_telemetry_ok = True
            self._last_error = None
            self._consecutive_error_count = 0

    def record_step_error(
        self,
        *,
        write_ok: Optional[bool],
        telemetry_ok: bool,
        error: Exception,
        phase: str,
        max_consecutive_errors: int,
    ) -> RuntimeError | None:
        with self._lock:
            self._last_write_ok = write_ok
            self._last_telemetry_ok = telemetry_ok
            self._last_error = f"{phase} error: {error!r}"
            self._consecutive_error_count += 1
            if self._consecutive_error_count < max_consecutive_errors:
                return None

            self._is_faulted = True
            return RuntimeError(
                f"control loop faulted after {self._consecutive_error_count} consecutive errors; "
                f"last_error={self._last_error}"
            )

    def record_tick(self) -> None:
        with self._lock:
            self._tick_count += 1
            self._last_tick_time = time.monotonic()

    def build_state(
        self,
        *,
        latest_target: Optional[np.ndarray],
        is_running: bool,
        worker_error: Optional[str],
    ) -> ControlLoopState:
        with self._lock:
            last_target = None if self._last_target is None else self._last_target.copy()
            latest_state = None if self._latest_state is None else self._latest_state.copy()
            latest_telemetry = (
                None if self._latest_telemetry is None else self._latest_telemetry.copy()
            )
            latest_pitch_out_of_range = self._latest_pitch_out_of_range
            last_write_ok = self._last_write_ok
            last_telemetry_ok = self._last_telemetry_ok
            last_error = self._last_error
            consecutive_error_count = self._consecutive_error_count
            tick_count = self._tick_count
            last_tick_time = self._last_tick_time
            is_faulted = self._is_faulted

        return ControlLoopState(
            is_running=is_running,
            is_faulted=is_faulted,
            latest_target=latest_target,
            last_target=last_target,
            latest_state=latest_state,
            latest_telemetry=latest_telemetry,
            latest_pitch_out_of_range=latest_pitch_out_of_range,
            last_write_ok=last_write_ok,
            last_telemetry_ok=last_telemetry_ok,
            last_error=last_error,
            consecutive_error_count=consecutive_error_count,
            tick_count=tick_count,
            last_tick_time=last_tick_time,
            worker_error=worker_error,
        )


class LatestValueControlLoop:
    """Background control loop with separate producer and consumer roles."""

    def __init__(
        self,
        adapter: ControlAdapterProtocol | None = None,
        config: DynamixelConfig | None = None,
        period_sec: float = CONTROL_PERIOD_SEC,
        max_consecutive_errors: int = DEFAULT_MAX_CONSECUTIVE_ERRORS,
    ) -> None:
        if period_sec <= 0.0:
            raise ValueError(f"period_sec must be > 0, got {period_sec}")
        if max_consecutive_errors <= 0:
            raise ValueError(
                f"max_consecutive_errors must be > 0, got {max_consecutive_errors}"
            )
        if adapter is not None and config is not None:
            raise ValueError("adapter and config cannot be provided at the same time")

        self._adapter = adapter if adapter is not None else PitchYawControlAdapter(config)
        self._period_sec = float(period_sec)
        self._max_consecutive_errors = int(max_consecutive_errors)
        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._worker_error: BaseException | None = None
        self.producer = LatestTargetProducer()
        self.consumer = ControlLoopConsumer()

    @property
    def latest_target(self) -> Optional[np.ndarray]:
        return self.producer.get_latest_target()

    @property
    def last_target(self) -> Optional[np.ndarray]:
        return self.consumer.get_last_target()

    @property
    def latest_state(self) -> Optional[np.ndarray]:
        return self.consumer.get_latest_state()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._worker_error = None
            self.consumer.clear_fault()
            self._thread = threading.Thread(
                target=self._run,
                name="dynamixel-control-loop",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._stop_event.set()
            thread = self._thread

        if thread is not None:
            thread.join()

        with self._lifecycle_lock:
            if self._thread is thread:
                self._thread = None

            if self._worker_error is not None:
                error = self._worker_error
                self._worker_error = None
                raise RuntimeError("control loop stopped due to worker error") from error

    def submit_target(self, target_rad: np.ndarray) -> None:
        self.producer.submit_target(target_rad)

    def get_latest_target(self) -> Optional[np.ndarray]:
        return self.latest_target

    def get_latest_state(self) -> Optional[np.ndarray]:
        return self.latest_state

    def get_latest_telemetry(self) -> Optional[np.ndarray]:
        return self.consumer.get_latest_telemetry()

    def get_state(self) -> ControlLoopState:
        latest_target = self.producer.get_latest_target()
        with self._lifecycle_lock:
            is_running = self._thread is not None and self._thread.is_alive()
            worker_error = None if self._worker_error is None else repr(self._worker_error)
        return self.consumer.build_state(
            latest_target=latest_target,
            is_running=is_running,
            worker_error=worker_error,
        )

    def _run(self) -> None:
        next_tick = time.monotonic()

        try:
            self._adapter.open()
            self._read_feedback_only()
            while not self._stop_event.is_set():
                command = self._select_operation_for_tick()
                if command is not None:
                    self._execute_control_step(
                        command,
                        max_consecutive_errors=self._max_consecutive_errors,
                    )
                self._record_tick()

                next_tick += self._period_sec
                remaining = next_tick - time.monotonic()
                if remaining > 0.0:
                    self._stop_event.wait(remaining)
                else:
                    next_tick = time.monotonic()
        except BaseException as exc:
            with self._lifecycle_lock:
                self._worker_error = exc
            self._stop_event.set()
        finally:
            self._adapter.close()

    def _select_operation_for_tick(self) -> Optional[np.ndarray]:
        return self.consumer.select_command(self.producer.get_latest_target())

    def _update_feedback(
        self,
        present_radians: np.ndarray,
        telemetry: np.ndarray,
        pitch_out_of_range: bool,
    ) -> None:
        self.consumer.update_feedback(
            present_radians,
            telemetry,
            pitch_out_of_range,
        )

    def _execute_control_step(
        self,
        command: np.ndarray,
        *,
        max_consecutive_errors: int | None = None,
    ) -> Optional[PitchYawFeedback]:
        error_threshold = (
            self._max_consecutive_errors
            if max_consecutive_errors is None
            else int(max_consecutive_errors)
        )
        try:
            self._adapter.write_target(command)
        except Exception as exc:
            self._record_step_error(
                write_ok=False,
                telemetry_ok=False,
                error=exc,
                phase="write",
                max_consecutive_errors=error_threshold,
            )
            return None

        self.consumer.record_write_success()

        try:
            feedback = self._adapter.read_feedback()
        except Exception as exc:
            self._record_step_error(
                write_ok=True,
                telemetry_ok=False,
                error=exc,
                phase="telemetry",
                max_consecutive_errors=error_threshold,
            )
            return None

        self._update_feedback(
            feedback.state_radians,
            feedback.telemetry,
            feedback.pitch_out_of_range,
        )
        self.consumer.record_telemetry_success(write_ok=True)
        return feedback

    def _read_feedback_only(self) -> Optional[PitchYawFeedback]:
        try:
            feedback = self._adapter.read_feedback()
        except Exception as exc:
            self._record_step_error(
                write_ok=None,
                telemetry_ok=False,
                error=exc,
                phase="telemetry",
                max_consecutive_errors=self._max_consecutive_errors,
            )
            return None

        self._update_feedback(
            feedback.state_radians,
            feedback.telemetry,
            feedback.pitch_out_of_range,
        )
        self.consumer.record_telemetry_success(write_ok=None)
        return feedback

    def _record_step_error(
        self,
        *,
        write_ok: Optional[bool],
        telemetry_ok: bool,
        error: Exception,
        phase: str,
        max_consecutive_errors: int,
    ) -> None:
        fault_error = self.consumer.record_step_error(
            write_ok=write_ok,
            telemetry_ok=telemetry_ok,
            error=error,
            phase=phase,
            max_consecutive_errors=max_consecutive_errors,
        )
        if fault_error is None:
            return

        with self._lifecycle_lock:
            self._worker_error = fault_error
        self._stop_event.set()

    def _record_tick(self) -> None:
        self.consumer.record_tick()


__all__ = [
    "CONTROL_FREQUENCY_HZ",
    "CONTROL_PERIOD_SEC",
    "DEFAULT_MAX_CONSECUTIVE_ERRORS",
    "ControlLoopConsumer",
    "ControlLoopState",
    "LatestTargetProducer",
    "LatestValueControlLoop",
    "MAX_YAW_DIRECT_STEP_RAD",
    "YAW_SLIDING_WINDOW_STEP_RAD",
]
