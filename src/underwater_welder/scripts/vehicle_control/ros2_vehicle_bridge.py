#!/usr/bin/env python3
"""
UDP 소켓 수신 → Jackal DriveAPI 브릿지 (Isaac Sim Python 3.11용)

ros_to_socket_bridge.py 로부터 UDP :9999 으로 (linear_x, angular_z) 수신.
rclpy 불필요 → Python 버전 충돌 없음.
"""

import socket
import struct

import omni.usd
from pxr import UsdPhysics

# ── 파라미터 ─────────────────────────────────────────────────────────────
WHEEL_RADIUS  = 0.098
TRACK_WIDTH   = 0.430
MAX_WHEEL_VEL = 100.0
UDP_PORT      = 9999

WHEEL_JOINT_PATHS = [
    "/jackal/front_left_wheel_joint",
    "/jackal/front_right_wheel_joint",
    "/jackal/rear_left_wheel_joint",
    "/jackal/rear_right_wheel_joint",
]
LEFT_WHEELS = {"front_left", "rear_left"}


class JackalVehicleBridge:
    def __init__(self, port: int = UDP_PORT):
        self._linear_x  = 0.0
        self._angular_z = 0.0

        # non-blocking UDP 소켓
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", port))
        self._sock.setblocking(False)
        print(f"  ✓ UDP 수신 대기: 0.0.0.0:{port}")

    def update(self):
        """시뮬레이션 루프마다 호출: UDP 수신 + 바퀴 적용"""
        # 논블로킹 수신 (데이터 없으면 이전 값 유지)
        try:
            data, _ = self._sock.recvfrom(16)
            self._linear_x, self._angular_z = struct.unpack("dd", data)
        except BlockingIOError:
            pass

        v   = self._linear_x
        w_z = self._angular_z

        left_vel  = (v - w_z * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
        right_vel = (v + w_z * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
        left_vel  = max(-MAX_WHEEL_VEL, min(MAX_WHEEL_VEL, left_vel))
        right_vel = max(-MAX_WHEEL_VEL, min(MAX_WHEEL_VEL, right_vel))

        self._apply(left_vel, right_vel)

    def _apply(self, left: float, right: float):
        stage = omni.usd.get_context().get_stage()
        for jpath in WHEEL_JOINT_PATHS:
            prim = stage.GetPrimAtPath(jpath)
            if not prim.IsValid():
                continue
            vel = left if any(s in jpath for s in LEFT_WHEELS) else right
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.GetStiffnessAttr().Set(0.0)
                drive.GetDampingAttr().Set(10_000_000.0)
                drive.GetMaxForceAttr().Set(10_000_000.0)
            drive.GetTargetVelocityAttr().Set(vel)

    def stop(self):
        self._linear_x  = 0.0
        self._angular_z = 0.0
        self._apply(0.0, 0.0)

    def close(self):
        self._sock.close()


# ── 공개 API ─────────────────────────────────────────────────────────────

def init_bridge(port: int = UDP_PORT) -> JackalVehicleBridge:
    return JackalVehicleBridge(port)

def spin_once(bridge: JackalVehicleBridge):
    bridge.update()

def shutdown_bridge(bridge: JackalVehicleBridge):
    bridge.stop()
    bridge.close()
