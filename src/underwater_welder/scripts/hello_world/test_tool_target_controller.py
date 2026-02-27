#!/usr/bin/env python3
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
from pxr import UsdGeom, Gf, UsdPhysics, UsdLux, UsdVol, UsdShade, Sdf
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.prims import XFormPrim
from isaacsim.core.api.objects import DynamicCuboid

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped
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
WELD_WAYPOINTS_PER_EDGE = 50/2  # ← 변경 시 green_marker_pnp_node.py도 같이 수정
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

# ─── 주변 버블 상수 ───────────────────────────────────────────────────────────
AMBIENT_BUBBLE_COUNT    = 25
AMBIENT_BUBBLE_R_MIN    = 0.015
AMBIENT_BUBBLE_R_MAX    = 0.06
AMBIENT_BUBBLE_SPD_MIN  = 0.25   # m/s
AMBIENT_BUBBLE_SPD_MAX  = 0.8    # m/s
AMBIENT_BUBBLE_TOP_Z    = 8.0    # 수면 위 → 리셋
AMBIENT_BUBBLE_BOT_Z    = -0.5   # 바닥

# ─── 수중 물리 상수 ────────────────────────────────────────────────────────────
WATER_DENSITY    = 1000.0   # [kg/m³]
WATER_VISCOSITY  = 0.001    # [Pa·s]
GRAVITY_ACC      = 9.81     # [m/s²]
WATER_SURFACE_Z  = 3.0      # [m] 수면 높이

DRAG_Cd          = 1.05
RE_THRESHOLD     = 1000.0
SURFACE_GRID_RES = 3

WAVE_F_SURFACE   = 20.0     # 수평 파도력 [N]
WAVE_F_VERTICAL  = 4.0      # 수직 파도력 [N]
WAVE_Z_DECAY     = 0.2
WAVE_UPDATE_SEC  = 5.0      # 파도 방향 전환 주기 [s]
WAVE_AMP         = 0.15     # 파고 [m]
WAVE_LEN         = 6.0      # 파장 [m]
WAVE_SPEED       = 0.6      # 파속 [m/s]

# ─── 수중 시각 효과 플래그 ──────────────────────────────────────────────────────
ENABLE_WATER_VISUAL  = False      # True: VDB 볼륨 렌더링 활성화 / False: 물리만
ENABLE_DEMO_CUBES    = True       # True: 부력·파도 시각화용 데모 큐브 6개 생성

# 데모 큐브 상수
DEMO_CUBE_MASSES  = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]   # 각 큐브 질량 [kg]
DEMO_CUBE_RADIUS  = 3.5           # 링 배치 반경 [m]
DEMO_CUBE_Z       = WATER_SURFACE_Z - 0.3              # 초기 Z [m]
DEMO_CUBE_SIZE    = 0.25          # 큐브 한 변 길이 [m]
DEMO_CUBE_COLOR   = np.array([0.3, 0.5, 0.9])          # 파란색

VOL_SCATTERING       = 0.85
VOL_ABSORPTION       = 0.85
VOL_ANISOTROPY       = -0.75
VOL_ALBEDO           = (0.04, 0.28, 0.45)  # 수중 청록색
VOL_CENTER_Z         = 1.0       # 볼륨 중심 높이 [m]

# 부력 등록 대상 (prim_path: 실제 USD 경로 확인 후 주석 해제)
BUOYANCY_PRIM_CONFIGS = [
    {"prim_path": "/jackal/base_link", "shape": "cube", "char_length": 0.5, "size": 0.5},
]

PLATE_PATH       = "/World/WeldPlate"
PLATE_SIZE_X     = 3.5    # 3개 마커 × 1.0m 간격 + 양쪽 여유
PLATE_SIZE_Z     = 1.0    # 마커 Z 변화량(0.4m) + 여유
PLATE_THICKNESS  = 0.025
PLATE_MASS       = 60.0
PLATE_COLOR      = Gf.Vec3f(0.0, 0.0, 0.0)

# 마커 3개: X 1.0m 간격, Z 0.2m씩 높이 차이
# 월드 기준: marker0=(-0.21, ~1.0, 1.14)  marker1=(0.79, ~1.0, 1.34)  marker2=(1.79, ~1.0, 1.54)
MARKER_X_OFFSETS   = [-1.0, 0.0, 1.0]  # 판 중심 기준 X 오프셋 [m]
JACKAL_MOVE_DIST     = 0.98               # 마커 간 이동 거리 [m]
JACKAL_MOVE_FRAMES   = 90               # 이동 후 안정화 대기 프레임
JACKAL_TURN_RATE     = math.radians(2.0) # kinematic 회전 속도 [rad/frame] (~45f per 90°)
JACKAL_DRIVE_RATE    = 0.015            # kinematic 직진 속도 [m/frame] (~67f per 1m)
JACKAL_CORRECT_RATE  = 0.003           # 보정 속도 [m/frame] (Phase2보다 느림)
JACKAL_WHEEL_SPEED   = 60.0            # 휠 시각 효과용 DriveAPI 속도 [rad/s]
JACKAL_MOVE_TOL      = 0.02            # Phase2 → correcting 전환 허용 오차 [m]
JACKAL_CORRECT_TOL   = 0.001           # 보정 완료 허용 오차 [m] (1mm)
JACKAL_YAW_TOL       = math.radians(1.0) # 목표 yaw 허용 오차 [rad] (~1°)

MARKER_Z_OFFSETS = [-0.1, 0.0, 0.1]    # 판 중심 기준 Z 오프셋 [m]
MARKER_PATHS     = [PLATE_PATH + f"/GreenMarker_{i}" for i in range(3)]
MARKER_PATH      = MARKER_PATHS[0]      # 하위 호환
MARKER_SIZE      = 0.1
MARKER_THICKNESS = 0.001
MARKER_COLOR     = Gf.Vec3f(0.0, 1.0, 0.0)

# 판 중심 = 3개 마커의 X/Z 중앙
PLATE_POS      = np.array([0.79, 1.0, 1.24])
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

# ─── 수중 시각 효과 함수 ─────────────────────────────────────────────────────────
def _setup_underwater_volume_vdb(stage):
    """VDB 볼륨으로 수중 산란 효과 생성 (ENABLE_WATER_VISUAL=True 시 호출)"""
    vol_path  = "/World/UnderwaterVolume"
    mat_path  = "/World/Looks/UnderwaterVolMat"
    if stage.GetPrimAtPath(vol_path).IsValid():
        print("[UnderwaterVDB] 이미 존재, 스킵.")
        return
    vol = UsdVol.Volume.Define(stage, vol_path)
    UsdGeom.Xformable(vol.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, VOL_CENTER_Z))
    field_path = vol_path + "/density"
    fp = stage.DefinePrim(field_path, "OpenVDBAsset")
    fp.CreateAttribute("fieldName", Sdf.ValueTypeNames.Token).Set("density")
    vol.CreateFieldRelationship("density", Sdf.Path(field_path))
    mat    = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("OmniVolume")
    shader.CreateInput("scattering_color",
                       Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*VOL_ALBEDO))
    shader.CreateInput("scattering_intensity",
                       Sdf.ValueTypeNames.Float).Set(VOL_SCATTERING)
    shader.CreateInput("absorption_color",
                       Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*VOL_ALBEDO))
    shader.CreateInput("absorption_intensity",
                       Sdf.ValueTypeNames.Float).Set(VOL_ABSORPTION)
    shader.CreateInput("anisotropy",
                       Sdf.ValueTypeNames.Float).Set(VOL_ANISOTROPY)
    shader.CreateInput("density_field",
                       Sdf.ValueTypeNames.Token).Set("density")
    shader.CreateInput("density_multiplier",
                       Sdf.ValueTypeNames.Float).Set(1.0)
    mat.CreateVolumeOutput().ConnectToSource(shader.ConnectableAPI(), "out")
    UsdShade.MaterialBindingAPI(vol.GetPrim()).Bind(
        UsdShade.Material(mat),
        UsdShade.Tokens.weakerThanDescendants,
        "volume")
    print("[UnderwaterVDB] 볼륨 생성 완료")

def _apply_blue_material(stage, prim_path: str):
    """데모 큐브에 파란색 OmniPBR 머티리얼 적용"""
    mat_path = "/World/Looks/DemoCubeMat"
    mat_sh   = mat_path + "/Shader"
    if not stage.GetPrimAtPath(mat_path).IsValid():
        mat    = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, mat_sh)
        shader.SetSourceAsset("OmniPBR.mdl", "mdl")
        shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        shader.CreateIdAttr("OmniPBR")
        shader.CreateInput("diffuse_color_constant",
                           Sdf.ValueTypeNames.Color3f).Set(
                               Gf.Vec3f(*DEMO_CUBE_COLOR.tolist()))
        shader.CreateInput("reflection_roughness_constant",
                           Sdf.ValueTypeNames.Float).Set(0.2)
        shader.CreateInput("metallic_constant",
                           Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        mat.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        mat.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        UsdShade.MaterialBindingAPI(prim).Bind(
            UsdShade.Material(stage.GetPrimAtPath(mat_path)))

def spawn_demo_cubes(world, water):
    """무게가 다른 6개의 큐브를 링 형태로 배치 → 부력·항력·파도력 등록"""
    if not ENABLE_DEMO_CUBES:
        return
    stage = omni.usd.get_context().get_stage()
    S = DEMO_CUBE_SIZE
    N = len(DEMO_CUBE_MASSES)
    for i, mass in enumerate(DEMO_CUBE_MASSES):
        angle = 2.0 * math.pi * i / N
        pos = np.array([
            DEMO_CUBE_RADIUS * math.cos(angle),
            DEMO_CUBE_RADIUS * math.sin(angle),
            DEMO_CUBE_Z,
        ])
        path = f"/World/DemoCube_{i}"
        world.scene.add(DynamicCuboid(
            prim_path=path,
            name=f"demo_cube_{i}",
            position=pos,
            scale=np.array([S, S, S]),
            color=DEMO_CUBE_COLOR,
            mass=mass,
        ))
        water.register(path, shape="cube", char_length=S, size=S)
        _apply_blue_material(stage, path)
        print(f"[DemoCube] {path}  mass={mass}kg  pos={np.round(pos,2)}")
    print(f"[DemoCube] {N}개 생성 완료")


# ─── 수중 물리 함수 ──────────────────────────────────────────────────────────────
def _get_wave_height(x, y, t):
    k = 2.0 * math.pi / WAVE_LEN
    return WAVE_AMP * math.sin(k * (x + y) - WAVE_SPEED * t)


def _calc_wave_force(z_local, wave_dir):
    decay   = math.exp(-max(z_local, 0.0) / WAVE_Z_DECAY)
    f_horiz = WAVE_F_SURFACE * decay
    f_vert  = WAVE_F_VERTICAL * decay
    return np.array([wave_dir[0]*f_horiz, wave_dir[1]*f_horiz, f_vert], dtype=np.float32)


def _generate_points_cube(size: float, grid_res: int = SURFACE_GRID_RES):
    half   = size / 2.0
    volume = size ** 3
    coords = np.linspace(-half, half, grid_res)
    points, normals = [], []
    for x in coords:
        for y in coords:
            for z in coords:
                pt   = np.array([x, y, z])
                norm = np.linalg.norm(pt)
                points.append(pt)
                normals.append(-pt / norm if norm > 1e-9 else None)
    n = len(points)
    return points, normals, volume / n, (size ** 2) / n


def _generate_points_sphere(radius: float):
    volume = (4.0 / 3.0) * np.pi * radius ** 3
    r      = radius * 0.8
    raw = [np.zeros(3),
           np.array([ r, 0., 0.]), np.array([-r, 0., 0.]),
           np.array([0.,  r, 0.]), np.array([0., -r, 0.]),
           np.array([0., 0.,  r]), np.array([0., 0., -r])]
    points, normals = [], []
    for pt in raw:
        norm = np.linalg.norm(pt)
        points.append(pt)
        normals.append(-pt / norm if norm > 1e-9 else None)
    n = len(points)
    return points, normals, volume / n, (np.pi * radius ** 2) / n


def generate_sample_points(shape: str, **kwargs):
    if shape == "cube":
        return _generate_points_cube(kwargs["size"])
    else:
        return _generate_points_sphere(kwargs.get("radius", kwargs.get("size", 0.5) / 2.0))


def compute_pressure_forces(local_points, local_normals, area_per_point,
                             world_pos, R, surface_z):
    force_list = []
    for pt_local, n_local in zip(local_points, local_normals):
        if n_local is None:
            continue
        pt_world = world_pos + R @ pt_local
        h_depth  = surface_z - pt_world[2]
        if h_depth <= 0.0:
            continue
        pressure = WATER_DENSITY * GRAVITY_ACC * h_depth
        force    = pressure * area_per_point * (R @ n_local)
        force_list.append((pt_world, force))
    return force_list


def compute_drag_forces(local_points, area_per_point, char_length,
                         world_pos, R, surface_z, lin_vel, ang_vel):
    force_list = []
    for pt_local in local_points:
        pt_world = world_pos + R @ pt_local
        if surface_z - pt_world[2] <= 0.0:
            continue
        v_point = lin_vel + np.cross(ang_vel, pt_world - world_pos)
        speed   = np.linalg.norm(v_point)
        if speed < 1e-6:
            continue
        Re = WATER_DENSITY * speed * char_length / WATER_VISCOSITY
        if Re < RE_THRESHOLD:
            drag = -3.0 * np.pi * WATER_VISCOSITY * char_length * v_point
        else:
            drag = -0.5 * WATER_DENSITY * DRAG_Cd * area_per_point * speed * v_point
        force_list.append((pt_world, drag))
    return force_list


class WaterPhysics:
    """수중 물리 시스템: 부력 + 항력 + 파도력을 등록된 RigidPrim에 매 프레임 적용"""

    def __init__(self):
        self._wave_timer      = 0.0
        self._wave_dir        = np.array([1.0, 0.0], dtype=np.float64)
        self._wave_dir_target = np.array([1.0, 0.0], dtype=np.float64)
        self._wave_dir_sign   = 1.0
        self._registry        = []
        self._rigid_prims     = {}   # prim_path → RigidPrim (setup() 이후)

    def register(self, prim_path: str, shape: str, char_length: float, **kwargs):
        pts, norms, _, area_per_pt = generate_sample_points(shape, **kwargs)
        self._registry.append({
            "prim_path":      prim_path,
            "char_length":    char_length,
            "local_points":   pts,
            "local_normals":  norms,
            "area_per_point": area_per_pt,
        })
        print(f"[Water] 등록: {prim_path}  shape={shape}  pts={len(pts)}")

    def setup(self, world):
        """world.reset() 이후에 호출. physics callback 등록."""
        from isaacsim.core.prims import RigidPrim as _RP
        for obj in self._registry:
            path = obj["prim_path"]
            try:
                rp = _RP(prim_paths_expr=path)
                self._rigid_prims[path] = rp
                print(f"[Water] ✓ RigidPrim 등록: {path}")
            except Exception as e:
                print(f"[Water] ⚠ RigidPrim 등록 실패 ({path}): {e}")
        world.add_physics_callback("water_physics_step", self.step)
        print(f"[Water] ✓ physics callback 등록 완료  총={len(self._rigid_prims)}개")

    def step(self, step_size: float):
        if not self._rigid_prims:
            return

        # 파도 방향 업데이트
        self._wave_timer += step_size
        if self._wave_timer >= WAVE_UPDATE_SEC:
            self._wave_timer = 0.0
            angle = random.uniform(0.0, 2.0 * math.pi)
            self._wave_dir_target = np.array([math.cos(angle), math.sin(angle)])
            self._wave_dir_sign   = 1.0
        if self._wave_timer >= WAVE_UPDATE_SEC * 0.5:
            self._wave_dir_sign = -1.0
        wave_dir_final  = self._wave_dir_target * self._wave_dir_sign
        self._wave_dir += (wave_dir_final - self._wave_dir) * 0.02
        self._wave_dir /= np.linalg.norm(self._wave_dir) + 1e-8
        wave_mult  = math.sin(2.0 * math.pi * self._wave_timer / WAVE_UPDATE_SEC)
        wave_mult *= 1.0 + 0.15 * math.sin(3.1 * self._wave_timer)

        for obj in self._registry:
            rp = self._rigid_prims.get(obj["prim_path"])
            if rp is None:
                continue
            try:
                positions, orientations = rp.get_world_poses()
                world_pos = np.array(positions[0], dtype=np.float64)
                lin_vel   = np.array(rp.get_linear_velocities()[0], dtype=np.float64)
                ang_vel   = np.array(rp.get_angular_velocities()[0], dtype=np.float64)
                R         = rotation_matrix_from_quaternion(np.array(orientations[0]))
            except Exception:
                continue

            wave_h = _get_wave_height(float(world_pos[0]), float(world_pos[1]), self._wave_timer)
            surf_z = WATER_SURFACE_Z + wave_h

            def _apply(pt_w, fv):
                try:
                    rp.apply_forces_and_torques_at_pos(
                        forces=fv.astype(np.float32).reshape(1, 3),
                        torques=np.zeros((1, 3), dtype=np.float32),
                        positions=pt_w.astype(np.float32).reshape(1, 3),
                        is_global=True)
                except Exception:
                    pass

            for pt_w, fv in compute_pressure_forces(
                    obj["local_points"], obj["local_normals"], obj["area_per_point"],
                    world_pos, R, surf_z):
                _apply(pt_w, fv)

            for pt_w, fv in compute_drag_forces(
                    obj["local_points"], obj["area_per_point"], obj["char_length"],
                    world_pos, R, surf_z, lin_vel, ang_vel):
                _apply(pt_w, fv)

            z_local = float(world_pos[2]) - surf_z
            f_wave  = _calc_wave_force(z_local, self._wave_dir) * wave_mult
            _apply(world_pos, f_wave)


# ─── ROS2 구독 노드 ──────────────────────────────────────────────────────────
class ToolTargetSubscriber(Node):
    def __init__(self):
        super().__init__('tool_target_controller')
        self._target_pos: np.ndarray | None = None
        self._locked: bool = False      # True이면 이동 중 → 새 목표 무시
        self._servo_mode: bool = False  # True이면 서보 중 → standoff 미적용
        self.corner_count: int = 0      # 완료된 코너 수 (0~4)
        self.create_subscription(PoseStamped, '/tool_target_pose', self._cb, 10)
        self._corner_ack_pub      = self.create_publisher(Empty, '/corner_ack',      10)
        self._weld_cycle_done_pub = self.create_publisher(Empty, '/weld_cycle_done', 10)
        self._ee_pose_pub         = self.create_publisher(PointStamped, '/ee_pose',    10)
        self._plate_pose_pub      = self.create_publisher(PointStamped, '/plate_pose', 10)
        self.create_subscription(PoseStamped, '/ee_servo_cmd', self._servo_cb, 10)
        self.get_logger().info('/tool_target_pose 구독 시작')

    def _cb(self, msg: PoseStamped):
        if self._locked:
            return  # 이동 중: 새 목표 무시
        self._servo_mode = False     # 용접 목표 → standoff 적용
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

    def _servo_cb(self, msg: PoseStamped):
        """시각 서보 명령: 잠금 없을 때만 목표 갱신 (WELDING 중 덮어쓰기 방지)"""
        if self._locked:
            return  # 용접 웨이포인트 수신 중 → 서보 잔류 메시지 무시
        self._servo_mode = True      # 서보 모드 → standoff 미적용
        self._target_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])

    def publish_ee_pose(self, pos: np.ndarray):
        msg = PointStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "World"
        msg.point.x = float(pos[0])
        msg.point.y = float(pos[1])
        msg.point.z = float(pos[2])
        self._ee_pose_pub.publish(msg)

    def publish_plate_pose(self):
        """판 위치를 /plate_pose 로 발행 (green_marker_pnp_node가 구독)
        point.x = PLATE_POS[0]  (판 중심 X)
        point.y = PLATE_FRONT_Y (용접면 Y = plate_front_y)
        point.z = PLATE_POS[2]  (판 중심 Z = plate_center_z)
        """
        msg = PointStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "World"
        msg.point.x = float(PLATE_POS[0])
        msg.point.y = float(PLATE_FRONT_Y)
        msg.point.z = float(PLATE_POS[2])
        self._plate_pose_pub.publish(msg)

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
        if self._ros._servo_mode:
            effective_target = target  # 서보: EE 위치 직접 목표 (standoff 미적용)
        else:
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

    # 초록 정사각형 마커 3개 (판 앞면에 부착, 위치/높이 각각 다름)
    sx = (MARKER_SIZE / 2.0) / (PLATE_SIZE_X / 2.0)
    sz = (MARKER_SIZE / 2.0) / (PLATE_SIZE_Z / 2.0)
    sy = MARKER_THICKNESS / PLATE_THICKNESS
    for i, (mx_off, mz_off) in enumerate(zip(MARKER_X_OFFSETS, MARKER_Z_OFFSETS)):
        local_x = mx_off / (PLATE_SIZE_X / 2.0)
        local_z = mz_off / (PLATE_SIZE_Z / 2.0)
        marker = UsdGeom.Cube.Define(stage, MARKER_PATHS[i])
        marker.CreateDisplayColorAttr([MARKER_COLOR])
        mxf = UsdGeom.Xformable(stage.GetPrimAtPath(MARKER_PATHS[i]))
        mxf.AddTranslateOp().Set(Gf.Vec3d(local_x, -1.0 - sy, local_z))
        mxf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
        world_x = PLATE_POS[0] + mx_off
        world_z = PLATE_POS[2] + mz_off
        print(f"[Weld] GreenMarker_{i} 생성  world=({world_x:.2f}, {PLATE_FRONT_Y:.3f}, {world_z:.2f})")

    print(f"[Weld] WeldPlate spawned  pos={PLATE_POS}  size=({PLATE_SIZE_X}×{PLATE_SIZE_Z}m)")
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


# ─── Jackal 휠 속도 제어 ──────────────────────────────────────────────────────
_WHEEL_JOINT_NAMES = [
    "front_left_wheel_joint",
    "front_right_wheel_joint",
    "rear_left_wheel_joint",
    "rear_right_wheel_joint",
]

def _setup_ambient_bubbles(stage) -> list:
    """주변 환경 버블 25개를 생성하고 상태 리스트 반환."""
    random.seed(80)
    bubbles = []
    for i in range(AMBIENT_BUBBLE_COUNT):
        path = f"/World/AmbientBubbles/bubble_{i:02d}"
        sph  = UsdGeom.Sphere.Define(stage, path)
        r    = random.uniform(AMBIENT_BUBBLE_R_MIN, AMBIENT_BUBBLE_R_MAX)
        sph.CreateRadiusAttr(r)
        x = random.uniform(-5.0, 5.0)
        y = random.uniform(-5.0, 5.0)
        z = random.uniform(AMBIENT_BUBBLE_BOT_Z, AMBIENT_BUBBLE_TOP_Z)
        UsdGeom.Xformable(sph).AddTranslateOp().Set(Gf.Vec3f(x, y, z))
        sph.CreateDisplayColorAttr([(0.65, 0.82, 1.0)])
        bubbles.append({"path": path, "x": x, "y": y, "z": z,
                         "speed": random.uniform(AMBIENT_BUBBLE_SPD_MIN, AMBIENT_BUBBLE_SPD_MAX)})
    print(f"[Bubble] 주변 버블 {len(bubbles)}개 생성 완료")
    return bubbles


def _step_ambient_bubbles(stage, bubbles: list, dt: float = 1.0 / 60.0):
    """메인 루프에서 매 프레임 호출 — 버블 위치 업데이트."""
    for b in bubbles:
        b["z"] += b["speed"] * dt
        if b["z"] > AMBIENT_BUBBLE_TOP_Z:
            b["z"] = AMBIENT_BUBBLE_BOT_Z
            b["x"] = random.uniform(-5.0, 5.0)
            b["y"] = random.uniform(-5.0, 5.0)
        prim = stage.GetPrimAtPath(b["path"])
        if prim.IsValid():
            attr = prim.GetAttribute("xformOp:translate")
            if attr:
                attr.Set(Gf.Vec3f(b["x"], b["y"], b["z"]))


def _setup_wheel_drives(stage) -> dict:
    """Stage 순회로 휠 조인트 DriveAPI 탐색·초기화 후 캐시 딕셔너리 반환."""
    drives = {}
    for prim in stage.Traverse():
        ppath = str(prim.GetPath())
        for wname in _WHEEL_JOINT_NAMES:
            if wname not in drives and ppath.endswith(wname):
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                if not drive:
                    drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.GetStiffnessAttr().Set(0.0)
                drive.GetDampingAttr().Set(10_000_000.0)
                drive.GetMaxForceAttr().Set(10_000_000.0)
                drive.GetTargetVelocityAttr().Set(0.0)
                drives[wname] = drive
                break
    found = list(drives.keys())
    print(f"[WheelDrive] 발견된 휠 조인트 {len(found)}/4: {found}")
    if len(found) < 4:
        print(f"[WheelDrive] ⚠ 일부 휠 조인트를 찾지 못함! USD 경로 확인 필요")
    return drives


def _set_wheel_velocities(drives: dict, fl: float, fr: float, rl: float = None, rr: float = None):
    """캐시된 DriveAPI로 Jackal 휠 속도 설정 [rad/s]. 양수=전진, 음수=후진."""
    if rl is None: rl = fl
    if rr is None: rr = fr
    vel_map = {
        "front_left_wheel_joint":  fl,
        "front_right_wheel_joint": fr,
        "rear_left_wheel_joint":   rl,
        "rear_right_wheel_joint":  rr,
    }
    for wname, vel in vel_map.items():
        if wname in drives:
            drives[wname].GetTargetVelocityAttr().Set(vel)


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

    # 수중 시각 효과 (VDB 볼륨)
    if ENABLE_WATER_VISUAL:
        _setup_underwater_volume_vdb(omni.usd.get_context().get_stage())

    # 용접봉 생성 (ee_link 자식 prim)
    spawn_weld_electrode()

    # World 생성
    world = World()
    world.scene.add_default_ground_plane()
    robot = Articulation(prim_path=ROBOT_PATH)

    # 수중 물리 시스템 생성 및 정적 객체 등록 (world.reset() 이전)
    water = WaterPhysics()
    for cfg in BUOYANCY_PRIM_CONFIGS:
        water.register(
            cfg["prim_path"], cfg["shape"], cfg["char_length"],
            **{k: v for k, v in cfg.items() if k not in ("prim_path", "shape", "char_length")}
        )
    # 데모 큐브 생성 + 부력 등록 (scene.add는 world.reset() 이전 필요)
    spawn_demo_cubes(world, water)

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
    stage         = omni.usd.get_context().get_stage()
    weld_sys      = WeldingSystem(stage, controller._ee)
    wheel_drives  = _setup_wheel_drives(stage)
    ambient_bubbles = _setup_ambient_bubbles(stage)

    print("[Main] ✓ WeldingSystem 생성 완료")

    # 수중 물리 RigidPrim 래핑 (world.reset() 이후 실행)
    water.setup(world)
    total_reg = len(BUOYANCY_PRIM_CONFIGS) + (len(DEMO_CUBE_MASSES) if ENABLE_DEMO_CUBES else 0)
    print(f"[Main] ✓ WaterPhysics 생성 완료  등록={total_reg}개")
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
    return_wait     = 0       # REACH_POSE 복귀 대기 카운트다운
    move_wait       = 0       # Jackal 도달 후 안정화 대기 카운트다운
    marker_idx      = 0       # 현재 용접 중인 마커 인덱스 (0~2)
    jackal_xf       = XFormPrim(prim_path="/jackal")
    # Jackal 이동 상태 머신: "idle" | "turning_x" | "driving_x"
    jackal_phase    = "idle"
    jackal_target_x = 0.0
    jackal_move_cnt = 0
    # 용접 중 위치 락 (밀림 방지)
    _lp, _lq        = jackal_xf.get_world_pose()
    jackal_lock_pos  = np.array(_lp, dtype=float)
    jackal_lock_quat = np.array(_lq, dtype=float)

    try:
        while simulation_app.is_running():
            world.step(render=True)
            _step_ambient_bubbles(stage, ambient_bubbles)
            controller.update()
            ros_node.publish_ee_pose(controller._ee_pos())
            ros_node.publish_plate_pose()
            weld_sys.step(at_target=controller._at_target)

            curr_weld_state = weld_sys._state

            # ── REACH_POSE 복귀 대기 → 완료 시 Jackal 이동 시작 ────────────
            if return_wait > 0:
                return_wait -= 1
                if return_wait == 0:
                    marker_idx += 1
                    if marker_idx < len(MARKER_X_OFFSETS):
                        pos, _ = jackal_xf.get_world_pose()
                        jackal_target_x = float(pos[0]) + JACKAL_MOVE_DIST
                        jackal_phase    = "turning_x"
                        jackal_move_cnt = 0
                        _set_wheel_velocities(wheel_drives, JACKAL_WHEEL_SPEED, -JACKAL_WHEEL_SPEED,
                                              JACKAL_WHEEL_SPEED, -JACKAL_WHEEL_SPEED)
                        print(f"[Main] 마커 {marker_idx-1} 완료 → Phase1 CW회전 시작 (목표X={jackal_target_x:.2f}m)")
                    else:
                        print("[Main] 전체 마커 용접 완료!")

            # ── Jackal 3단계 kinematic 이동 상태 머신 ───────────────────
            elif jackal_phase != "idle":
                pos, quat = jackal_xf.get_world_pose()
                pos  = np.array(pos,  dtype=float)
                quat = np.array(quat, dtype=float)
                qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
                cur_yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
                jackal_move_cnt += 1

                def _yaw_to_quat(yaw):
                    h = yaw / 2.0
                    return np.array([math.cos(h), 0.0, 0.0, math.sin(h)])

                if jackal_phase == "turning_x":
                    # kinematic CW 회전 → yaw 0°
                    # XYZ는 jackal_lock_pos에 고정, yaw만 변경 (회전 중 위치 흘림 방지)
                    yaw_err = cur_yaw - 0.0
                    if yaw_err >  math.pi: yaw_err -= 2*math.pi
                    if yaw_err < -math.pi: yaw_err += 2*math.pi
                    if jackal_move_cnt % 30 == 0:
                        print(f"[Main] Phase1 CW회전  yaw={math.degrees(cur_yaw):.1f}°  err={math.degrees(yaw_err):.1f}°")
                    if abs(yaw_err) <= JACKAL_YAW_TOL:
                        jackal_xf.set_world_pose(jackal_lock_pos, _yaw_to_quat(0.0))
                        jackal_phase = "driving_x"
                        jackal_move_cnt = 0
                        _set_wheel_velocities(wheel_drives, JACKAL_WHEEL_SPEED, JACKAL_WHEEL_SPEED,
                                              JACKAL_WHEEL_SPEED, JACKAL_WHEEL_SPEED)
                        print(f"[Main] Phase2: +X 직진 시작 (→{jackal_target_x:.2f}m)")
                    else:
                        step = min(JACKAL_TURN_RATE, abs(yaw_err)) * (-1.0 if yaw_err > 0 else 1.0)
                        jackal_xf.set_world_pose(jackal_lock_pos, _yaw_to_quat(cur_yaw + step))

                elif jackal_phase == "driving_x":
                    # X만 이동, Y/Z는 jackal_lock_pos 기준으로 고정
                    cur_x = pos[0]
                    dx = jackal_target_x - cur_x
                    if jackal_move_cnt % 30 == 0:
                        print(f"[Main] Phase2 직진  X={cur_x:.3f}→{jackal_target_x:.2f}  남은={dx:.3f}m")
                    if abs(dx) <= JACKAL_MOVE_TOL:
                        print(f"[Main] Phase2→보정  실제X={cur_x:.4f}  목표X={jackal_target_x:.4f}  오차={dx*100:.2f}cm")
                        _set_wheel_velocities(wheel_drives, JACKAL_CORRECT_RATE * 200,
                                              JACKAL_CORRECT_RATE * 200,
                                              JACKAL_CORRECT_RATE * 200,
                                              JACKAL_CORRECT_RATE * 200)
                        jackal_phase    = "correcting"
                        jackal_move_cnt = 0
                    else:
                        step = min(JACKAL_DRIVE_RATE, abs(dx)) * (1.0 if dx > 0 else -1.0)
                        moving = jackal_lock_pos.copy()
                        moving[0] = cur_x + step
                        jackal_xf.set_world_pose(moving, _yaw_to_quat(0.0))

                elif jackal_phase == "correcting":
                    # 느린 속도로 정확한 위치에 도달
                    cur_x = pos[0]
                    dx = jackal_target_x - cur_x
                    if jackal_move_cnt % 10 == 0:
                        print(f"[Main] 보정 중  X={cur_x:.4f}→{jackal_target_x:.4f}  오차={dx*100:.2f}cm")
                    if abs(dx) <= JACKAL_CORRECT_TOL:
                        print(f"[Main] 보정 완료  최종X={cur_x:.4f}  잔여오차={dx*1000:.1f}mm")
                        _set_wheel_velocities(wheel_drives, 0.0, 0.0, 0.0, 0.0)
                        final_pos = jackal_lock_pos.copy()
                        final_pos[0] = cur_x          # 실제 도착 위치 그대로 lock
                        jackal_xf.set_world_pose(final_pos, _yaw_to_quat(0.0))
                        jackal_lock_pos  = final_pos.copy()
                        jackal_lock_quat = _yaw_to_quat(0.0)
                        jackal_phase    = "idle"
                        jackal_move_cnt = 0
                        move_wait       = JACKAL_MOVE_FRAMES
                    else:
                        step = min(JACKAL_CORRECT_RATE, abs(dx)) * (1.0 if dx > 0 else -1.0)
                        moving = jackal_lock_pos.copy()
                        moving[0] = cur_x + step
                        jackal_xf.set_world_pose(moving, _yaw_to_quat(0.0))

            # ── Jackal 안정화 대기 ────────────────────────────────────────
            elif move_wait > 0:
                move_wait -= 1
                if move_wait == 0:
                    print(f"[Main] Jackal 안정화 완료 → weld_cycle_done 발행 (마커 {marker_idx})")
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

            # ── 용접 중 위치 강제 고정 (매 프레임 lock 위치 유지) ────────────
            if (jackal_phase == "idle" and move_wait == 0 and return_wait == 0):
                jackal_xf.set_world_pose(jackal_lock_pos, jackal_lock_quat)

            prev_weld_state = curr_weld_state

    except KeyboardInterrupt:
        print("\n[Main] 종료 중...")
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
