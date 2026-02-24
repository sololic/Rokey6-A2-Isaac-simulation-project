#!/usr/bin/env python3
"""
test_tool_target_controller.py
/tool_target_pose 수신 → UR10 IK → 팔 제어 테스트

실행:
  ~/.local/share/ov/pkg/isaac_sim-*/python.sh \
      .../hello_world/test_tool_target_controller.py

테스트 명령 (다른 터미널):
  ros2 topic pub /tool_target_pose geometry_msgs/msg/PoseStamped \
      '{header: {frame_id: World}, pose: {position: {x: 1.0, y: 0.5, z: 1.0}}}' --once
"""

# ── Python 3.11용 ROS2 패키지 경로 추가 (SimulationApp 보다 반드시 먼저) ──────
import sys
_ROS2_PY311 = "/home/rokey/isaacsim/exts/isaacsim.ros2.bridge/humble/rclpy"
if _ROS2_PY311 not in sys.path:
    sys.path.insert(0, _ROS2_PY311)
# ────────────────────────────────────────────────────────────────────────────

# ── Isaac Sim 반드시 먼저 시작 ──────────────────────────────────────────────
from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
    "extra_args": ["--enable", "omni.isaac.ros2_bridge"],
})

import numpy as np
import omni.usd
from pxr import UsdGeom, Gf, UsdPhysics
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.prims import XFormPrim

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

# ─── 균열 마커 상수 ─────────────────────────────────────────────────────────
MARKER_XY = [(-0.2, 1.0)]   # 마커 1개만 생성

# ─── REACH 자세 (홈 → 이 자세로 이동) ─────────────────────────────────────
REACH_POSE_DEG = {
    "shoulder_pan_joint":   90.0,
    "shoulder_lift_joint": -90.0,
    "elbow_joint":          90.0,
    "wrist_1_joint":         0.0,
    "wrist_2_joint":         90.0,
    "wrist_3_joint":          0.0,
}

# ─── 설정 상수 ──────────────────────────────────────────────────────────────
USD_PATH       = "/home/rokey/hamtaro/src/underwater_welder/jackal_and_ur10_2.usd"
ROBOT_PATH     = "/jackal"
EE_PRIM_PATH   = "/jackal/ur10/ee_link"
UR10_BASE_PATH = "/jackal/ur10"

UR10_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# IK 파라미터
IK_GAIN     = 0.4    # 수렴 속도 (0.1~1.0)
IK_MAX_STEP = 0.06   # 스텝당 최대 관절 변화 [rad] (~3.4°)
IK_TOL      = 0.01   # 수렴 허용 거리 [m]
UR10_SCALE  = 0.5    # USD 파일에서 UR10이 0.5배 스케일

# ─── UR10 DH 파라미터 (표준 UR10, UR10_SCALE 적용) ──────────────────────────
# 출처: Universal Robots UR10 스펙 (d1=127.3mm, a2=612mm, a3=572.3mm, ...)
_S = UR10_SCALE
DH_D     = [0.1273*_S, 0.0,       0.0,        0.1639*_S, 0.1157*_S, 0.0922*_S]
DH_A     = [0.0,      -0.612*_S, -0.5723*_S,  0.0,       0.0,       0.0      ]
DH_ALPHA = [np.pi/2,   0.0,       0.0,         np.pi/2,  -np.pi/2,  0.0      ]


# ─── FK / Jacobian ──────────────────────────────────────────────────────────
def _dh(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1],
    ])


def ur10_fk(q: np.ndarray) -> np.ndarray:
    """관절 각도 6개 → EE 변환 행렬 4x4 (UR10 base frame 기준)"""
    T = np.eye(4)
    for i in range(6):
        T = T @ _dh(q[i], DH_D[i], DH_A[i], DH_ALPHA[i])
    return T


def position_jacobian(q: np.ndarray, T_wb: np.ndarray) -> np.ndarray:
    """수치 미분으로 위치 Jacobian 계산 (3×6, world frame)"""
    eps = 0.001
    ee0 = (T_wb @ ur10_fk(q))[:3, 3]
    J = np.zeros((3, 6))
    for i in range(6):
        q2 = q.copy(); q2[i] += eps
        J[:, i] = ((T_wb @ ur10_fk(q2))[:3, 3] - ee0) / eps
    return J


def xform_to_mat4(prim: XFormPrim) -> np.ndarray:
    """Isaac Sim XFormPrim → 4×4 변환 행렬 (column-vector convention)"""
    pos, quat = prim.get_world_pose()   # quat = [w, x, y, z]
    qw, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    R = np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)  ],
        [2*(qx*qy+qw*qz),   1-2*(qx**2+qz**2),  2*(qy*qz-qw*qx)  ],
        [2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),    1-2*(qx**2+qy**2)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = pos[:3]
    return T


# ─── ROS2 구독 노드 ──────────────────────────────────────────────────────────
class ToolTargetSubscriber(Node):
    def __init__(self):
        super().__init__('tool_target_controller')
        self._target_pos: np.ndarray | None = None
        self.create_subscription(PoseStamped, '/tool_target_pose', self._cb, 10)
        self.get_logger().info('/tool_target_pose 구독 시작')

    def _cb(self, msg: PoseStamped):
        self._target_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        self.get_logger().info(
            f'새 목표: x={msg.pose.position.x:.3f} '
            f'y={msg.pose.position.y:.3f} '
            f'z={msg.pose.position.z:.3f}'
        )

    @property
    def target(self) -> np.ndarray | None:
        return self._target_pos


# ─── IK 팔 제어기 ────────────────────────────────────────────────────────────
class ArmIKController:
    """
    /tool_target_pose 목표 → Jacobian Transpose IK → ArticulationAction 적용
    """

    def __init__(self, robot: Articulation, ros_node: ToolTargetSubscriber):
        self._robot = robot
        self._ros   = ros_node
        self._ee    = XFormPrim(EE_PRIM_PATH)
        self._base  = XFormPrim(UR10_BASE_PATH)

        # DOF 인덱스 추출 (UR10_JOINT_NAMES 순서 유지)
        dof_names = robot.dof_names
        self._arm_idx: list[int] = []
        for name in UR10_JOINT_NAMES:
            for i, dname in enumerate(dof_names):
                if dname == name:
                    self._arm_idx.append(i)
                    break

        if len(self._arm_idx) != 6:
            print(f"[ArmIK] ⚠ {len(self._arm_idx)}/6 관절만 찾음 – DOF 이름 확인 필요")
            print(f"[ArmIK]   전체 DOF: {list(dof_names)}")
        else:
            print(f"[ArmIK] ✓ UR10 DOF 인덱스: {self._arm_idx}")

        # UR10 관절 position drive 명시적 설정
        stage = omni.usd.get_context().get_stage()
        drive_count = 0
        for prim in stage.Traverse():
            ppath = str(prim.GetPath())
            for jname in UR10_JOINT_NAMES:
                if ppath.endswith(jname):
                    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                    if not drive:
                        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                    drive.GetStiffnessAttr().Set(10000.0)
                    drive.GetDampingAttr().Set(500.0)
                    drive.GetMaxForceAttr().Set(100000.0)
                    drive_count += 1
                    break
        print(f"[ArmIK] ✓ Position DriveAPI 설정: {drive_count}/6 관절")

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────
    def _arm_q(self) -> np.ndarray:
        return self._robot.get_joint_positions()[self._arm_idx]

    def _ee_pos(self) -> np.ndarray:
        pos, _ = self._ee.get_world_pose()
        return np.array(pos[:3], dtype=float)

    def _apply(self, q_arm: np.ndarray):
        all_pos = self._robot.get_joint_positions().copy()
        for i, idx in enumerate(self._arm_idx):
            all_pos[idx] = q_arm[i]
        self._robot.apply_action(ArticulationAction(joint_positions=all_pos))

    # ── 메인 업데이트 ──────────────────────────────────────────────────────
    def update(self):
        """매 시뮬레이션 스텝에서 호출"""
        # 1. ROS2 메시지 처리 (논블로킹)
        rclpy.spin_once(self._ros, timeout_sec=0)

        target = self._ros.target
        if target is None:
            return

        # 2. 현재 상태 읽기
        q       = self._arm_q()
        ee_pos  = self._ee_pos()
        T_base  = xform_to_mat4(self._base)

        # 3. 오차 계산
        error    = target - ee_pos
        err_norm = float(np.linalg.norm(error))

        # 디버그: 항상 출력 (수렴 후 억제)
        print(f"[ArmIK] err={err_norm:.4f}m  ee={ee_pos.round(3)}  target={target.round(3)}")

        if err_norm < IK_TOL:
            return  # 목표 도달

        # 4. Jacobian Transpose IK 한 스텝
        J  = position_jacobian(q, T_base)
        dq = IK_GAIN * (J.T @ error)
        dq = np.clip(dq, -IK_MAX_STEP, IK_MAX_STEP)

        # 5. 적용
        self._apply(q + dq)


# ─── 균열 마커 생성 ──────────────────────────────────────────────────────────
def create_crack_markers():
    stage = omni.usd.get_context().get_stage()
    S, T, CZ = 0.2, 0.025, 1.24
    color       = Gf.Vec3f(0.0, 1.0, 0.0)
    black_color = Gf.Vec3f(0.0, 0.0, 0.0)
    margin = 0.3

    for i, (cx, cy) in enumerate(MARKER_XY):
        path_bg = f'/World/CrackMarker_{i}_bg'
        bg = UsdGeom.Cube.Define(stage, path_bg)
        bg.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(path_bg))
        xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy + T, CZ))
        xf.AddScaleOp().Set(Gf.Vec3d(S + margin, T, S + margin))
        bg.GetDisplayColorAttr().Set([black_color])

        path = f'/World/CrackMarker_{i}'
        box = UsdGeom.Cube.Define(stage, path)
        box.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, CZ))
        xf.AddScaleOp().Set(Gf.Vec3d(S, T, S))
        box.GetDisplayColorAttr().Set([color])

    print(f"[Marker] 균열 마커 {len(MARKER_XY)}개 생성")


# ─── Isaac Sim 메인 루프 ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Tool Target Controller Test")
    print("  /tool_target_pose → UR10 IK 제어")
    print("=" * 60)

    # USD 로드
    omni.usd.get_context().open_stage(USD_PATH)

    # 균열 마커 생성
    create_crack_markers()

    # World 생성
    world = World()
    world.scene.add_default_ground_plane()
    robot = Articulation(prim_path=ROBOT_PATH)

    # ROS2 초기화
    rclpy.init()
    ros_node = ToolTargetSubscriber()

    # 시뮬레이션 시작
    world.reset()
    robot.initialize()
    print(f"[Main] DOF 수: {robot.num_dof}")
    for i, n in enumerate(robot.dof_names):
        print(f"  [{i:2d}] {n}")

    # 홈 자세 설정
    home_deg = {
        "shoulder_pan_joint":   0.0,
        "shoulder_lift_joint":  0.0,
        "elbow_joint":          0.0,
        "wrist_1_joint":        0.0,
        "wrist_2_joint":        0.0,
        "wrist_3_joint":        0.0,
    }
    dof_names = robot.dof_names
    init_pos = robot.get_joint_positions().copy()
    for jname, deg in home_deg.items():
        for i, n in enumerate(dof_names):
            if n == jname:
                init_pos[i] = np.deg2rad(deg)
    robot.set_joint_positions(init_pos)
    print("[Main] ✓ 홈 자세 설정 완료 (0,0,0,0,0,0)")

    # REACH 자세로 이동
    reach_pos = init_pos.copy()
    for jname, deg in REACH_POSE_DEG.items():
        for i, n in enumerate(dof_names):
            if n == jname:
                reach_pos[i] = np.deg2rad(deg)
    robot.apply_action(ArticulationAction(joint_positions=reach_pos))
    print("[Main] ✓ REACH_POSE 적용")

    # IK 제어기 생성
    controller = ArmIKController(robot, ros_node)

    print("\n[Main] 시뮬레이션 실행 중. 다른 터미널에서 목표 발행:")
    print("  ros2 topic pub /tool_target_pose geometry_msgs/msg/PoseStamped \\")
    print("    '{header: {frame_id: World}, pose: {position: {x: 1.0, y: 0.5, z: 1.0}}}' --once")
    print("Ctrl+C로 종료\n")

    # 메인 루프
    try:
        while simulation_app.is_running():
            world.step(render=True)
            controller.update()
    except KeyboardInterrupt:
        print("\n[Main] 종료 중...")
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
