from __future__ import annotations

"""双电机 Dynamixel 流式控制器。

这个模块提供一个单线程控制类，用于持续接收长度为 2 的 numpy 向量，
并把它们作为两个电机的目标位置写入总线。

每次写入目标位置后，控制器会同步读取两个电机的当前位置、速度、电流、电压、温度，
并返回一个形状为 ``(2, 5)`` 的数组。
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
from dynamixel_sdk import GroupSyncRead, GroupSyncWrite, PacketHandler, PortHandler


ADDR_OPERATING_MODE = 11
ADDR_MAX_POSITION_LIMIT = 48
ADDR_MIN_POSITION_LIMIT = 52
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_INPUT_VOLTAGE = 144
ADDR_PRESENT_TEMPERATURE = 146

PROTOCOL_VERSION = 2.0
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
POSITION_MODE = 3
GOAL_POSITION_LENGTH = 4
TELEMETRY_START_ADDR = ADDR_PRESENT_CURRENT
TELEMETRY_LENGTH = ADDR_PRESENT_TEMPERATURE - ADDR_PRESENT_CURRENT + 1
COUNTS_PER_REV = 4096
MOTOR1_MIN_POSITION_LIMIT = 1024
MOTOR1_MAX_POSITION_LIMIT = 2048


def _split_u32(value: int) -> list[int]:
    """把 32 位无符号整数拆成 4 个字节。"""
    return [
        value & 0xFF,
        (value >> 8) & 0xFF,
        (value >> 16) & 0xFF,
        (value >> 24) & 0xFF,
    ]


def _to_signed(value: int, bits: int) -> int:
    """把寄存器原始值转换为有符号整数。"""
    sign_bit = 1 << (bits - 1)
    full_scale = 1 << bits
    return value - full_scale if value & sign_bit else value


@dataclass(frozen=True)
class DynamixelConfig:
    """控制器基础配置。"""

    device_name: str = "COM9"
    baudrate: int =1000000
    dxl_ids: tuple[int, int] = (1, 2)


class DualDynamixelController:
    """双电机位置控制器。

    这个类面向两个 Dynamixel 电机，输入为 shape=(2,) 的目标位置向量。
    调用 `stream_positions()` 时，会按迭代器产出的数据持续下发控制命令，
    并在每次下发后返回两个电机的五维状态。
    """

    def __init__(self, config: DynamixelConfig | None = None):
        self.config = config or DynamixelConfig()
        self.port = PortHandler(self.config.device_name)
        self.packet = PacketHandler(PROTOCOL_VERSION)
        # 使用同步写一次下发两个电机的位置目标，减少总线开销。
        self.group_sync_write = GroupSyncWrite(
            self.port,
            self.packet,
            ADDR_GOAL_POSITION,
            GOAL_POSITION_LENGTH,
        )
        # 126~146 是 X 系列一段连续的状态寄存器，便于统一读取。
        self.group_sync_read = GroupSyncRead(
            self.port,
            self.packet,
            TELEMETRY_START_ADDR,
            TELEMETRY_LENGTH,
        )
        self._is_open = False

    def _check_result(self, dxl_comm_result: int, dxl_error: int, action: str) -> None:
        if dxl_comm_result != 0:
            raise RuntimeError(f"{action} comm error: {self.packet.getTxRxResult(dxl_comm_result)}")
        if dxl_error != 0:
            raise RuntimeError(f"{action} dxl error: {self.packet.getRxPacketError(dxl_error)}")

    def _write1(self, dxl_id: int, address: int, value: int, action: str) -> None:
        dxl_comm_result, dxl_error = self.packet.write1ByteTxRx(
            self.port, dxl_id, address, value
        )
        self._check_result(dxl_comm_result, dxl_error, f"{action} (id={dxl_id})")

    def _write4(self, dxl_id: int, address: int, value: int, action: str) -> None:
        dxl_comm_result, dxl_error = self.packet.write4ByteTxRx(
            self.port, dxl_id, address, value
        )
        self._check_result(dxl_comm_result, dxl_error, f"{action} (id={dxl_id})")

    def open(self) -> None:
        """打开串口并初始化两个电机为位置模式。"""
        if self._is_open:
            return

        if not self.port.openPort():
            raise RuntimeError(f"failed to open port: {self.config.device_name}")
        if not self.port.setBaudRate(self.config.baudrate):
            self.port.closePort()
            raise RuntimeError(f"failed to set baudrate: {self.config.baudrate}")

        try:
            for dxl_id in self.config.dxl_ids:
                self._write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, "disable torque")
                self._write1(dxl_id, ADDR_OPERATING_MODE, POSITION_MODE, "set position mode")
                if dxl_id == self.config.dxl_ids[0]:
                    self._write4(
                        dxl_id,
                        ADDR_MAX_POSITION_LIMIT,
                        MOTOR1_MAX_POSITION_LIMIT,
                        "set max position limit",
                    )
                    self._write4(
                        dxl_id,
                        ADDR_MIN_POSITION_LIMIT,
                        MOTOR1_MIN_POSITION_LIMIT,
                        "set min position limit",
                    )
                self._write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE, "enable torque")
                if not self.group_sync_read.addParam(dxl_id):
                    raise RuntimeError(f"failed to register telemetry read for motor id={dxl_id}")
        except Exception:
            self.port.closePort()
            raise

        self._is_open = True

    def close(self) -> None:
        """关闭扭矩并关闭串口。"""
        if not self._is_open:
            return

        for dxl_id in self.config.dxl_ids:
            try:
                self._write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, "disable torque")
            except RuntimeError:
                pass

        self.port.closePort()
        self._is_open = False

    def __enter__(self) -> "DualDynamixelController":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def write_positions(self, target_positions: np.ndarray) -> None:
        """写入一帧双电机目标位置。"""
        if not self._is_open:
            raise RuntimeError("controller is not open")

        positions = np.asarray(target_positions, dtype=np.int64)
        if positions.shape != (2,):
            raise ValueError(f"expected target_positions shape (2,), got {positions.shape}")

        self.group_sync_write.clearParam()

        for dxl_id, goal in zip(self.config.dxl_ids, positions.tolist(), strict=True):
            if goal < 0:
                raise ValueError(f"goal position must be >= 0, got {goal}")
            if goal > 0xFFFFFFFF:
                raise ValueError(f"goal position exceeds uint32 range: {goal}")

            if not self.group_sync_write.addParam(dxl_id, _split_u32(int(goal))):
                self.group_sync_write.clearParam()
                raise RuntimeError(f"failed to queue goal position for motor id={dxl_id}")

        dxl_comm_result = self.group_sync_write.txPacket()
        if dxl_comm_result != 0:
            raise RuntimeError(
                f"write goal positions comm error: {self.packet.getTxRxResult(dxl_comm_result)}"
            )

    def read_telemetry(self) -> np.ndarray:
        """读取两个电机的五维状态。

        返回数组的列顺序为：
        1. 位置
        2. 速度
        3. 电流
        4. 电压
        5. 温度
        """
        if not self._is_open:
            raise RuntimeError("controller is not open")

        dxl_comm_result = self.group_sync_read.txRxPacket()
        if dxl_comm_result != 0:
            raise RuntimeError(
                f"read telemetry comm error: {self.packet.getTxRxResult(dxl_comm_result)}"
            )

        telemetry = np.zeros((2, 5), dtype=np.float64)
        for row, dxl_id in enumerate(self.config.dxl_ids):
            if not self.group_sync_read.isAvailable(dxl_id, ADDR_PRESENT_CURRENT, 2):
                raise RuntimeError(f"telemetry current unavailable for motor id={dxl_id}")
            if not self.group_sync_read.isAvailable(dxl_id, ADDR_PRESENT_VELOCITY, 4):
                raise RuntimeError(f"telemetry velocity unavailable for motor id={dxl_id}")
            if not self.group_sync_read.isAvailable(dxl_id, ADDR_PRESENT_POSITION, 4):
                raise RuntimeError(f"telemetry position unavailable for motor id={dxl_id}")
            if not self.group_sync_read.isAvailable(dxl_id, ADDR_PRESENT_INPUT_VOLTAGE, 2):
                raise RuntimeError(f"telemetry voltage unavailable for motor id={dxl_id}")
            if not self.group_sync_read.isAvailable(dxl_id, ADDR_PRESENT_TEMPERATURE, 1):
                raise RuntimeError(f"telemetry temperature unavailable for motor id={dxl_id}")

            current_raw = self.group_sync_read.getData(dxl_id, ADDR_PRESENT_CURRENT, 2)
            velocity_raw = self.group_sync_read.getData(dxl_id, ADDR_PRESENT_VELOCITY, 4)
            position_raw = self.group_sync_read.getData(dxl_id, ADDR_PRESENT_POSITION, 4)
            voltage_raw = self.group_sync_read.getData(dxl_id, ADDR_PRESENT_INPUT_VOLTAGE, 2)
            temperature_raw = self.group_sync_read.getData(dxl_id, ADDR_PRESENT_TEMPERATURE, 1)

            telemetry[row] = np.array(
                [
                    _to_signed(position_raw, 32),
                    _to_signed(velocity_raw, 32),
                    _to_signed(current_raw, 16),
                    float(voltage_raw),
                    float(temperature_raw),
                ],
                dtype=np.float64,
            )

        return telemetry

    def write_and_read(self, target_positions: np.ndarray) -> np.ndarray:
        """写入一帧目标位置，并立即返回一帧状态。"""
        self.write_positions(target_positions)
        return self.read_telemetry()

    def stream_positions(self, position_stream: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
        """顺序消费目标位置流，并持续返回状态流。

        `position_stream` 需要持续产出长度为 2 的 numpy 向量，
        每个向量对应两个电机当前时刻的目标位置。
        每消费一帧输入，就会 `yield` 一个形状为 ``(2, 5)`` 的状态数组。
        """
        for target_positions in position_stream:
            yield self.write_and_read(target_positions)


__all__ = ["DualDynamixelController", "DynamixelConfig"]
