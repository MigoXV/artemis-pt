from __future__ import annotations

import unittest

import numpy as np

from artemis_pt.dynamixel.control_loop import LatestValueControlLoop


def deg(value: float) -> float:
    return float(np.deg2rad(value))


class ControlLoopWindowingTest(unittest.TestCase):
    def test_large_yaw_command_is_split_into_sliding_windows(self) -> None:
        loop = LatestValueControlLoop(adapter=_NoopAdapter())
        loop._update_feedback(_target(0.0, 0.0), _telemetry(), False)

        loop.submit_target(_target(0.0, 170.0))

        first_command = loop._select_operation_for_tick()
        self.assertIsNotNone(first_command)
        np.testing.assert_allclose(first_command, _target(0.0, 60.0), atol=1e-8)

        # The next command must be based on fresh feedback, not on the previous write target.
        loop._update_feedback(_target(0.0, 15.0), _telemetry(), False)
        second_command = loop._select_operation_for_tick()
        self.assertIsNotNone(second_command)
        np.testing.assert_allclose(second_command, _target(0.0, 75.0), atol=1e-8)

    def test_new_large_command_replaces_previous_window_plan(self) -> None:
        loop = LatestValueControlLoop(adapter=_NoopAdapter())
        loop._update_feedback(_target(0.0, 0.0), _telemetry(), False)

        loop.submit_target(_target(0.0, 170.0))
        first_command = loop._select_operation_for_tick()
        self.assertIsNotNone(first_command)
        np.testing.assert_allclose(first_command, _target(0.0, 60.0), atol=1e-8)

        loop._update_feedback(_target(0.0, 60.0), _telemetry(), False)
        loop.submit_target(_target(0.0, -90.0))

        redirected_command = loop._select_operation_for_tick()
        self.assertIsNotNone(redirected_command)
        np.testing.assert_allclose(redirected_command, _target(0.0, 0.0), atol=1e-8)


def _target(pitch_deg: float, yaw_deg: float) -> np.ndarray:
    return np.array([deg(pitch_deg), deg(yaw_deg)], dtype=np.float64)


def _telemetry() -> np.ndarray:
    return np.zeros((2, 5), dtype=np.float64)


class _NoopAdapter:
    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def write_target(self, target: np.ndarray) -> None:
        return None

    def read_feedback(self):
        raise AssertionError("read_feedback should not be called in this unit test")


if __name__ == "__main__":
    unittest.main()
