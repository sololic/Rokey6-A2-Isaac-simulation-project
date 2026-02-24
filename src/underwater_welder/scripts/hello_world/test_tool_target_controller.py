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

import math
import random
from enum import Enum, auto

import numpy as np
import omni.usd
from pxr import UsdGeom, Gf, UsdPhysics, UsdLux
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.prims import XFormPrim

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty

# ─── 용접 판 / 마커 상수 ────────────────────────────────────────────────────
WELD_TRIGGER_DIST    = 0.04   # 용접 트리거 거리 [m]

# ── 용접 대기 시간 (프레임 단위, 60fps 기준 1프레임≈16ms) ──────────────────────
WELD_HEAT_STEPS      = 30/50    # ← 변경 가능: 가열 대기 (30f ≈ 0.5초)
WELD_ACTIVE_STEPS    = 20/50   # ← 변경 가능: 용접 활성 (20f ≈ 0.3초)
WELD_COOLDOWN_STEPS  = 15/50    # ← 변경 가능: 냉각 대기 (15f ≈ 0.25초)
# 한 점당 총 대기 = HEAT + ACTIVE + COOLDOWN = 65f ≈ 1.1초
# ────────────────────────────────────────────────────────────────────────────

# ── 변당 웨이포인트 수 (green_marker_pnp_node.py의 WAYPOINTS_PER_EDGE와 동일해야 함) ─
WELD_WAYPOINTS_PER_EDGE = 80  # ← 변경 시 green_marker_pnp_node.py도 같이 수정
# 한 사이클 총 용접 점 수 = WELD_WAYPOINTS_PER_EDGE × 4
# ────────────────────────────────────────────────────────────────────────────

BEAD_POOL_RADIUS     = 0.015
BEAD_HEIGHT          = 0.003
BEAD_COLOR           = Gf.Vec3f(0.78, 0.78, 0.82)

SPARK_BATCH_SIZE     = 4
SPARK_MAX_ALIVE      = 6
SPARK_INTENSITY_PEAK = 50000000.0
SPARK_RADIUS_LIGHT   = 0.003
SPARK_COLOR          = Gf.Vec3f(0.4, 0.6, 1.0)
SPARK_LIFETIME_STEPS = 8
SPARK_JITTER         = 0.008
SPARK_GAP_RATIO      = 0.95

BUBBLE_COUNT          = 3
BUBBLE_RADIUS         = 0.008
BUBBLE_COLOR          = Gf.Vec3f(0.82, 0.92, 1.0)
BUBBLE_RISE_SPEED     = 0.05
BUBBLE_LIFETIME_STEPS = 60

PLATE_PATH       = "/World/WeldPlate"
PLATE_SIZE_X     = 0.5
PLATE_SIZE_Z     = 0.5
PLATE_THICKNESS  = 0.025
PLATE_MASS       = 60.0
PLATE_COLOR      = Gf.Vec3f(0.0, 0.0, 0.0)

MARKER_PATH      = PLATE_PATH + "/GreenMarker"
MARKER_SIZE      = 0.2
MARKER_THICKNESS = 0.001
MARKER_COLOR     = Gf.Vec3f(0.0, 1.0, 0.0)

# 마커 위치: (x=-0.2, y=1.0, z=1.24)  ← MARKER_XY와 동일
PLATE_POS      = np.array([-0.2, 1.0, 1.24])
PLATE_FRONT_Y  = float(PLATE_POS[1]) - PLATE_THICKNESS / 2.0   # 로봇이 접근하는 면

# 용접봉 (ee_link 자식 prim → 팔과 같이 움직임)
ELECTRODE_PRIM_PATH = "/jackal/ur10/ee_link/WeldRod"
ELECTRODE_TIP_PATH  = "/jackal/ur10/ee_link/WeldRod/Tip"
ELECTRODE_RADIUS    = 0.004   # 봉 반지름 [m]
ELECTRODE_HEIGHT    = 0.08    # 봉 길이 [m] (ee_link local-x 방향으로 뻗음)

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
IK_GAIN     = 0.5    # 수렴 속도 (0.1~1.0)
IK_MAX_STEP = 0.03   # 스텝당 최대 관절 변화 [rad] (~3.4°)
IK_TOL      = 0.003   # 수렴 허용 거리 [m]
UR10_SCALE  = 0.5    # USD 파일에서 UR10이 0.5배 스케일
# standoff = ELECTRODE_HEIGHT + WELD_TRIGGER_DIST*0.8 (update()에서 동적 계산)

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


def rotation_matrix_from_quaternion(quat: np.ndarray) -> np.ndarray:
    """쿼터니언 [w,x,y,z] → 3×3 회전 행렬"""
    w, x, y, z = quat
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


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
        self._locked: bool = False   # True이면 이동 중 → 새 목표 무시
        self.corner_count: int = 0   # 완료된 코너 수 (0~4)
        self.create_subscription(PoseStamped, '/tool_target_pose', self._cb, 10)
        self._corner_ack_pub      = self.create_publisher(Empty, '/corner_ack',      10)
        self._weld_cycle_done_pub = self.create_publisher(Empty, '/weld_cycle_done', 10)
        self.get_logger().info('/tool_target_pose 구독 시작')

    def _cb(self, msg: PoseStamped):
        if self._locked:
            return  # 이동 중: 새 목표 무시
        self._target_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        self._locked = True          # 목표 수신 즉시 잠금
        self.get_logger().info(
            f'새 목표: x={msg.pose.position.x:.3f} '
            f'y={msg.pose.position.y:.3f} '
            f'z={msg.pose.position.z:.3f}'
        )

    def unlock(self):
        """다음 목표 수신 허용 (이전 목표 삭제 → 새 목표 올 때까지 IK 대기)"""
        self._target_pos = None
        self._locked = False

    def publish_corner_ack(self):
        self._corner_ack_pub.publish(Empty())

    def publish_weld_cycle_done(self):
        self._weld_cycle_done_pub.publish(Empty())

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

        # UR10 관절 DriveAPI 저장 + position drive 명시적 설정
        stage = omni.usd.get_context().get_stage()
        self._drives: dict[str, object] = {}   # jname → DriveAPI
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
                    self._drives[jname] = drive
                    break
        print(f"[ArmIK] ✓ Position DriveAPI 설정: {len(self._drives)}/6 관절")

        # ── 경험적 Jacobian 보정 상태 ──────────────────────────────────────
        self._J_emp:      np.ndarray | None = None   # 완성되면 세팅
        self._J_building: np.ndarray        = np.zeros((3, 6))
        self._cal_q0:     np.ndarray | None = None
        self._cal_ee0:    np.ndarray | None = None
        self._cal_i:      int               = 0      # 현재 관절 인덱스 (0~5)
        self._cal_phase:  int               = 0      # 0=섭동적용 1=대기(perturbed) 2=측정+복원 3=대기(restored)
        self._cal_wait:   int               = 0
        self._CAL_EPS    = np.deg2rad(8.0)   # 섭동 크기 [rad]
        self._CAL_WAIT   = 8                # 물리 수렴 대기 프레임
        self._settle_wait = 80              # REACH_POSE 도달 대기 프레임 (보정 전)

        self._at_target: bool = False   # 목표 도달 플래그 (용접 완료 후 unlock)
        self._returning: bool = False   # REACH_POSE 복귀 중 플래그

        self.count = 0

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────
    def _arm_q(self) -> np.ndarray:
        return self._robot.get_joint_positions()[self._arm_idx]

    def _ee_pos(self) -> np.ndarray:
        pos, _ = self._ee.get_world_pose()
        return np.array(pos[:3], dtype=float)

    def _apply(self, q_arm: np.ndarray):
        # DriveAPI targetPosition 직접 설정 (degree)
        for i, jname in enumerate(UR10_JOINT_NAMES):
            if jname in self._drives:
                deg = float(np.degrees(q_arm[i]))
                self._drives[jname].GetTargetPositionAttr().Set(deg)

    # ── 경험적 Jacobian 보정 ───────────────────────────────────────────────
    def _calibrate(self) -> bool:
        """
        경험적 Jacobian을 계산한다.
        보정 중이면 True 반환 (IK 실행 금지), 완료 후 False 반환.
        """
        if self._J_emp is not None:
            return False   # 이미 완료

        # REACH_POSE 도달 대기
        if self._settle_wait > 0:
            self._settle_wait -= 1
            if self._settle_wait == 0:
                print("[Cal] REACH_POSE 안정화 완료 → 보정 시작")
            return True

        if self._cal_i >= 6:
            self._J_emp = self._J_building.copy()
            print("[Cal] ✓ 경험적 Jacobian 완성")
            print(f"[Cal] J =\n{self._J_emp.round(4)}")
            return False

        if self._cal_phase == 0:
            # 현재 상태 저장 + 섭동 적용
            self._cal_q0  = self._arm_q().copy()
            self._cal_ee0 = self._ee_pos().copy()
            q_pert = self._cal_q0.copy()
            q_pert[self._cal_i] += self._CAL_EPS
            self._apply(q_pert)
            self._cal_wait = 0
            self._cal_phase = 1

        elif self._cal_phase == 1:
            # 섭동 후 물리 수렴 대기
            self._cal_wait += 1
            if self._cal_wait >= self._CAL_WAIT:
                self._cal_phase = 2

        elif self._cal_phase == 2:
            # EE 변위 측정 → J 열 기록 → 관절 복원
            ee_pert = self._ee_pos()
            dee = ee_pert - self._cal_ee0
            self._J_building[:, self._cal_i] = dee / self._CAL_EPS
            print(f"[Cal] joint {self._cal_i} ({UR10_JOINT_NAMES[self._cal_i]}): "
                  f"dee={dee.round(4)}")
            self._apply(self._cal_q0)
            self._cal_wait = 0
            self._cal_phase = 3

        elif self._cal_phase == 3:
            # 복원 후 물리 수렴 대기 → 다음 관절
            self._cal_wait += 1
            if self._cal_wait >= self._CAL_WAIT:
                self._cal_i += 1
                self._cal_phase = 0

        return True   # 아직 보정 중

    # ── 메인 업데이트 ──────────────────────────────────────────────────────
    def update(self):
        """매 시뮬레이션 스텝에서 호출"""
        # 1. ROS2 메시지 처리 (논블로킹)
        rclpy.spin_once(self._ros, timeout_sec=0)
        # 2. 경험적 Jacobian 보정 (완료되기 전에는 IK 실행 안 함)
        if self._calibrate():
            return
        # 3. 복귀 중에는 IK 실행 안 함 (DriveAPI로 REACH_POSE 유지)
        if self._returning:
            return

        target = self._ros.target
        if target is None:
            return
        self.count += 1
        # 4. 현재 상태 읽기
        q      = self._arm_q()
        ee_pos = self._ee_pos()

        # 5. 용접봉 방향(elec_dir) 기반 standoff (봉 끝이 마커를 뚫지 않도록)
        _, quat_ee   = self._ee.get_world_pose()
        R_ee         = rotation_matrix_from_quaternion(np.array(quat_ee))
        elec_dir     = R_ee @ np.array([1.0, 0.0, 0.0])   # ee_link local-x → world
        standoff_dist    = ELECTRODE_HEIGHT + WELD_TRIGGER_DIST * 0.8
        effective_target = target - elec_dir * standoff_dist

        # 6. 오차 계산
        error    = effective_target - ee_pos
        err_norm = float(np.linalg.norm(error))
        if self.count > 10:
            print(f"[ArmIK] err={err_norm:.4f}m  ee={ee_pos.round(3)}  eff_target={effective_target.round(3)}")
            self.count = 0

        if err_norm < IK_TOL:
            if not self._at_target:
                self._at_target = True
                print(f"[ArmIK] 목표 도달 → 용접 완료 대기")
            return  # unlock은 용접 COOLDOWN→IDLE 후 main에서 처리

        # 7. Jacobian Transpose IK 한 스텝 (경험적 J 사용)
        J  = self._J_emp
        dq = IK_GAIN * (J.T @ error)
        dq = np.clip(dq, -IK_MAX_STEP, IK_MAX_STEP)
        self._apply(q + dq)


# ─── 용접봉 생성 (ee_link 자식 → 팔과 같이 움직임) ──────────────────────────
def spawn_weld_electrode():
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(ELECTRODE_PRIM_PATH).IsValid():
        print("[Weld] WeldElectrode already exists, skipping.")
        return

    # 봉 몸체 (검은 원기둥)
    # Cylinder 기본 축 = local-Z. RotateY=90° 하면 → ee_link local-X 방향으로 정렬
    # 중심을 ee_link local-X의 HEIGHT/2 위치에 놓으면 봉이 0 ~ HEIGHT 범위에 걸침
    rod = UsdGeom.Cylinder.Define(stage, ELECTRODE_PRIM_PATH)
    rod.CreateRadiusAttr(ELECTRODE_RADIUS)
    rod.CreateHeightAttr(ELECTRODE_HEIGHT)
    rod.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.1, 0.1)])
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ELECTRODE_PRIM_PATH))
    xf.AddTranslateOp().Set(Gf.Vec3d(ELECTRODE_HEIGHT / 2.0, 0.0, 0.0))
    xf.AddRotateYOp().Set(90.0)

    # 봉 끝 (주황색 구)
    # WeldRod 로컬 Z축 = Cylinder 축 = RotateY 전 기준 → Tip은 WeldRod 로컬 +Z 끝에
    tip = UsdGeom.Sphere.Define(stage, ELECTRODE_TIP_PATH)
    tip.CreateRadiusAttr(ELECTRODE_RADIUS * 1.8)
    tip.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.5, 0.1)])
    UsdGeom.Xformable(stage.GetPrimAtPath(ELECTRODE_TIP_PATH)).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, ELECTRODE_HEIGHT / 2.0))  # WeldRod 로컬 +Z 끝 = 봉 끝

    print(f"[Weld] WeldElectrode spawned  parent={ELECTRODE_PRIM_PATH.rsplit('/', 1)[0]}")
    print(f"[Weld]   반지름={ELECTRODE_RADIUS*1000:.1f}mm  길이={ELECTRODE_HEIGHT*100:.1f}cm")


# ─── 용접 상태 머신 ──────────────────────────────────────────────────────────
class WeldState(Enum):
    IDLE     = auto()
    HEATING  = auto()
    WELDING  = auto()
    COOLDOWN = auto()


# ─── 용접 판 생성 ─────────────────────────────────────────────────────────────
def spawn_weld_plate():
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(PLATE_PATH).IsValid():
        print("[Weld] WeldPlate already exists, skipping.")
        return

    # 검은 배경 판
    geom = UsdGeom.Cube.Define(stage, PLATE_PATH)
    geom.CreateDisplayColorAttr([PLATE_COLOR])
    prim = stage.GetPrimAtPath(PLATE_PATH)
    xf   = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(*PLATE_POS.tolist()))
    xf.AddScaleOp().Set(Gf.Vec3f(PLATE_SIZE_X/2.0, PLATE_THICKNESS/2.0, PLATE_SIZE_Z/2.0))
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(PLATE_MASS)
    UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)

    # 초록 정사각형 마커 (판 앞면에 부착)
    marker = UsdGeom.Cube.Define(stage, MARKER_PATH)
    marker.CreateDisplayColorAttr([MARKER_COLOR])
    mxf = UsdGeom.Xformable(stage.GetPrimAtPath(MARKER_PATH))
    sx = (MARKER_SIZE / 2.0) / (PLATE_SIZE_X / 2.0)
    sz = (MARKER_SIZE / 2.0) / (PLATE_SIZE_Z / 2.0)
    sy = MARKER_THICKNESS / PLATE_THICKNESS
    mxf.AddTranslateOp().Set(Gf.Vec3d(0.0, -1.0 - sy, 0.0))  # 앞면(-y)에 부착
    mxf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))

    print(f"[Weld] WeldPlate spawned  pos={PLATE_POS}")
    print(f"[Weld] GreenMarker spawned  size={MARKER_SIZE}x{MARKER_SIZE}m")
    print(f"[Weld] PLATE_FRONT_Y={PLATE_FRONT_Y:.4f}")


# ─── 용접 시스템 ──────────────────────────────────────────────────────────────
class WeldingSystem:
    """
    UR10 ee_link 위치를 전극으로 사용하는 용접 시스템.
    ee_link가 PLATE_FRONT_Y에서 WELD_TRIGGER_DIST 이내로 접근하면 용접 시작.
    """

    def __init__(self, stage, ee_xform: XFormPrim):
        self._stage          = stage
        self._ee             = ee_xform        # UR10 ee_link XFormPrim
        self._state          = WeldState.IDLE
        self._heat_counter   = 0
        self._active_counter = 0
        self._cool_counter   = 0
        self._bead_idx       = 0
        self._fx_idx         = 0
        self._spark_prims    = []
        self._bubble_prims   = []

    def step(self, at_target: bool = False):
        self._tick_particles()
        tip_pos, elec_dir = self._get_tip_and_direction()
        if tip_pos is None:
            return
        dist = self._dist_to_plate(tip_pos)

        if self._state == WeldState.IDLE:
            if dist <= WELD_TRIGGER_DIST and at_target:
                self._state        = WeldState.HEATING
                self._heat_counter = 1
                print(f"[Weld] IDLE → HEATING  dist={dist*100:.1f}cm")

        elif self._state == WeldState.HEATING:
            if dist <= WELD_TRIGGER_DIST:
                self._heat_counter += 1
                if self._heat_counter >= WELD_HEAT_STEPS:
                    self._state          = WeldState.WELDING
                    self._active_counter = 0
                    print("[Weld] HEATING → WELDING")
            else:
                self._state = WeldState.IDLE
                print("[Weld] HEATING 취소 → IDLE")

        elif self._state == WeldState.WELDING:
            self._active_counter += 1
            if self._active_counter == 1:
                self._spawn_weld_bead(tip_pos, elec_dir)
            self._spawn_sparks(tip_pos)
            if self._active_counter >= WELD_ACTIVE_STEPS:
                self._state        = WeldState.COOLDOWN
                self._cool_counter = 0
                self._spawn_bubbles(tip_pos)
                print("[Weld] WELDING → COOLDOWN")

        elif self._state == WeldState.COOLDOWN:
            self._cool_counter += 1
            if self._cool_counter >= WELD_COOLDOWN_STEPS:
                self._state = WeldState.IDLE
                print("[Weld] COOLDOWN → IDLE")

    def _get_tip_and_direction(self):
        try:
            pos, quat = self._ee.get_world_pose()
            ee_pos   = np.array(pos[:3], dtype=float)
            R        = rotation_matrix_from_quaternion(np.array(quat))
            elec_dir = R @ np.array([1.0, 0.0, 0.0])   # ee_link local-x → world
            # 실제 용접봉 끝 = ee_link 중심 + 봉 길이만큼 앞
            tip_pos  = ee_pos + elec_dir * ELECTRODE_HEIGHT
            return tip_pos, elec_dir
        except Exception as e:
            print(f"[Weld] tip 계산 실패: {e}")
            return None, None

    def _dist_to_plate(self, tip_pos: np.ndarray) -> float:
        cx, cy, cz = PLATE_POS
        closest = np.array([
            np.clip(tip_pos[0], cx - PLATE_SIZE_X   / 2.0, cx + PLATE_SIZE_X   / 2.0),
            np.clip(tip_pos[1], cy - PLATE_THICKNESS / 2.0, cy + PLATE_THICKNESS / 2.0),
            np.clip(tip_pos[2], cz - PLATE_SIZE_Z   / 2.0, cz + PLATE_SIZE_Z   / 2.0),
        ])
        return float(np.linalg.norm(tip_pos - closest))

    def _spawn_weld_bead(self, tip_pos: np.ndarray, electrode_dir: np.ndarray):
        path = f"/World/WeldBeads/Bead_{self._bead_idx:04d}"
        self._bead_idx += 1
        bead_pos    = tip_pos.copy()
        bead_pos[1] = PLATE_FRONT_Y - BEAD_HEIGHT * 0.5
        plate_normal = np.array([0.0, 1.0, 0.0])
        cos_theta    = np.clip(np.dot(electrode_dir, plate_normal), -1.0, 1.0)
        angle_rad    = math.acos(abs(cos_theta))
        angle_ratio  = angle_rad / (math.pi / 2.0)
        dir_on_plate = electrode_dir - cos_theta * plate_normal
        dir_len      = np.linalg.norm(dir_on_plate)
        dist_ratio   = np.clip(self._dist_to_plate(tip_pos) / WELD_TRIGGER_DIST, 0.0, 1.0)
        dist_scale   = 0.3 + (1.2 - 0.3) * (1.0 - dist_ratio)
        sx = BEAD_POOL_RADIUS * dist_scale
        sy = BEAD_HEIGHT * dist_scale * (1.0 - 0.6 * angle_ratio)
        sz = BEAD_POOL_RADIUS * dist_scale
        stretch = 1.0 + 2.5 * angle_ratio
        if dir_len > 1e-4:
            d  = dir_on_plate / dir_len
            sx *= 1.0 + (stretch - 1.0) * abs(d[0])
            sz *= 1.0 + (stretch - 1.0) * abs(d[2])
        if not self._stage.GetPrimAtPath(path).IsValid():
            sphere = UsdGeom.Sphere.Define(self._stage, path)
            sphere.CreateRadiusAttr(1.0)
            sphere.CreateDisplayColorAttr([BEAD_COLOR])
            xf = UsdGeom.Xformable(self._stage.GetPrimAtPath(path))
            xf.AddTranslateOp().Set(Gf.Vec3d(*bead_pos.tolist()))
            xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
            print(f"[Weld] 비드 생성: {path}")

    def _spawn_sparks(self, origin: np.ndarray):
        plate_contact = origin.copy()
        plate_contact[1] = PLATE_FRONT_Y
        spark_origin = plate_contact + (origin - plate_contact) * SPARK_GAP_RATIO
        if len(self._spark_prims) >= SPARK_MAX_ALIVE:
            return
        spawn_n  = min(SPARK_BATCH_SIZE, SPARK_MAX_ALIVE - len(self._spark_prims))
        base_idx = self._fx_idx
        self._fx_idx += 1
        for i in range(spawn_n):
            path = f"/World/WeldFX/Spark_{base_idx:04d}_{i}"
            if self._stage.GetPrimAtPath(path).IsValid():
                continue
            jitter = np.array([
                random.uniform(-SPARK_JITTER, SPARK_JITTER),
                random.uniform(-SPARK_JITTER * 0.3, SPARK_JITTER * 0.3),
                random.uniform(-SPARK_JITTER, SPARK_JITTER),
            ])
            light = UsdLux.SphereLight.Define(self._stage, path)
            light.CreateIntensityAttr(SPARK_INTENSITY_PEAK)
            light.CreateColorAttr(SPARK_COLOR)
            light.CreateRadiusAttr(SPARK_RADIUS_LIGHT)
            UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(
                Gf.Vec3d(*(spark_origin + jitter).tolist()))
            self._spark_prims.append([path, SPARK_LIFETIME_STEPS])

    def _spawn_bubbles(self, origin: np.ndarray):
        base_idx = self._fx_idx
        self._fx_idx += 1
        for i in range(BUBBLE_COUNT):
            path = f"/World/WeldFX/Bubble_{base_idx:04d}_{i}"
            if self._stage.GetPrimAtPath(path).IsValid():
                continue
            geom = UsdGeom.Sphere.Define(self._stage, path)
            geom.CreateRadiusAttr(BUBBLE_RADIUS)
            geom.CreateDisplayColorAttr([BUBBLE_COLOR])
            offset = np.array([random.uniform(-0.04, 0.04),
                                random.uniform(-0.02, 0.02), 0.0])
            UsdGeom.Xformable(geom.GetPrim()).AddTranslateOp().Set(
                Gf.Vec3d(*(origin + offset).tolist()))
            rise_vel = np.array([
                random.uniform(-0.003, 0.003),
                random.uniform(-0.003, 0.003),
                BUBBLE_RISE_SPEED * random.uniform(0.8, 1.2),
            ])
            self._bubble_prims.append([path, BUBBLE_LIFETIME_STEPS, rise_vel])

    def _tick_particles(self):
        survivors = []
        for path, life in self._spark_prims:
            prim = self._stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            life -= 1
            if life <= 0:
                self._stage.RemovePrim(path)
                continue
            ratio = life / SPARK_LIFETIME_STEPS
            UsdLux.SphereLight(prim).GetIntensityAttr().Set(
                SPARK_INTENSITY_PEAK * (ratio ** 2))
            survivors.append([path, life])
        self._spark_prims = survivors

        b_survivors = []
        for path, life, vel in self._bubble_prims:
            prim = self._stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            life -= 1
            if life <= 0:
                self._stage.RemovePrim(path)
                continue
            xf   = UsdGeom.Xformable(prim)
            t_op = next((op for op in xf.GetOrderedXformOps()
                         if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
            if t_op is not None:
                t_op.Set(Gf.Vec3d(*(np.array(t_op.Get()) + vel).tolist()))
            b_survivors.append([path, life, vel])
        self._bubble_prims = b_survivors


# ─── Isaac Sim 메인 루프 ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Tool Target Controller Test")
    print("  /tool_target_pose → UR10 IK 제어")
    print("=" * 60)

    # USD 로드
    omni.usd.get_context().open_stage(USD_PATH)

    # 용접 판 + 마커 생성
    spawn_weld_plate()

    # 용접봉 생성 (ee_link 자식 prim)
    spawn_weld_electrode()

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

    # IK 제어기 생성 (DriveAPI 설정 포함)
    controller = ArmIKController(robot, ros_node)

    # 용접 시스템 생성 (UR10 ee_link를 전극으로 사용)
    stage    = omni.usd.get_context().get_stage()
    weld_sys = WeldingSystem(stage, controller._ee)
    print("[Main] ✓ WeldingSystem 생성 완료")
    print(f"[Main]   용접 트리거 거리: {WELD_TRIGGER_DIST*100:.1f}cm")
    print(f"[Main]   standoff = ELECTRODE_HEIGHT({ELECTRODE_HEIGHT*100:.0f}cm) + {WELD_TRIGGER_DIST*0.8*100:.1f}cm = {(ELECTRODE_HEIGHT+WELD_TRIGGER_DIST*0.8)*100:.1f}cm")

    # REACH 자세로 이동 (DriveAPI 경유 → 실제로 작동)
    reach_q = np.zeros(6)
    for i, jname in enumerate(UR10_JOINT_NAMES):
        reach_q[i] = np.deg2rad(REACH_POSE_DEG.get(jname, 0.0))
    controller._apply(reach_q)
    print("[Main] ✓ REACH_POSE DriveAPI 적용 (80프레임 안정화 대기 중...)")

    print("\n[Main] 시뮬레이션 실행 중. Ctrl+C로 종료\n")

    # 메인 루프
    prev_weld_state = WeldState.IDLE
    return_wait     = 0                  # REACH_POSE 복귀 대기 카운트다운

    try:
        while simulation_app.is_running():
            world.step(render=True)
            controller.update()
            weld_sys.step(at_target=controller._at_target)

            curr_weld_state = weld_sys._state

            # ── 복귀 대기 카운트다운 ────────────────────────────────────────
            if return_wait > 0:
                return_wait -= 1
                if return_wait == 0:
                    print("[Main] 복귀 완료 → weld_cycle_done 발행")
                    ros_node.publish_weld_cycle_done()
                    ros_node.corner_count = 0
                    controller._returning = False
                    ros_node.unlock()   # 다음 사이클 첫 코너 수신 허용

            # ── 용접 완료 감지 (COOLDOWN → IDLE) ────────────────────────────
            elif (prev_weld_state == WeldState.COOLDOWN
                    and curr_weld_state == WeldState.IDLE
                    and controller._at_target):
                controller._at_target = False
                ros_node.corner_count += 1
                ros_node.publish_corner_ack()
                print(f"[Main] corner {ros_node.corner_count}/4 완료 → corner_ack 발행")

                if ros_node.corner_count >= 4 * WELD_WAYPOINTS_PER_EDGE:
                    print(f"[Main] 전체 {4 * WELD_WAYPOINTS_PER_EDGE}점 완료 → REACH_POSE 복귀 시작")
                    controller._returning = True
                    controller._apply(reach_q)
                    return_wait = 120   # ~2초 대기
                else:
                    ros_node.unlock()   # 다음 코너 수신 허용

            prev_weld_state = curr_weld_state

    except KeyboardInterrupt:
        print("\n[Main] 종료 중...")
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
