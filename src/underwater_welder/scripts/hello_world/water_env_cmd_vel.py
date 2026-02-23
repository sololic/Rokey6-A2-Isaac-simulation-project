"""
water_env_cmd_vel.py
======================
water_env.py 복사본 — cmd_vel 통합 + 팔/용접 제거 버전

[원본 대비 변경 사항]
  추가: JackalCmdVelController import + 초기화 + 매 스텝 update()
  제거: 용접 블록 전체 (WeldState, WeldingSystem, spawn_weld_plate,
         spawn_weld_electrode, setup_welding)
  제거: 로봇-전극 결합 (_attach_electrode_to_robot)
  제거: __init__ / setup_scene / setup_post_load / _buoyancy_step 의
         용접 관련 호출 [E-W0~W3]

[사용 방법]
  water_env_extension.py 에서 아래처럼 교체:
    # 기존
    from .water_env import Water_Env
    # 변경
    from .water_env_cmd_vel import Water_Env

[cmd_vel 테스트]
  ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
      '{linear: {x: 1.0}, angular: {z: 0.0}}' -r 10
"""

from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import RigidPrim
from pxr import UsdPhysics, PhysxSchema, UsdGeom, UsdLux, UsdVol, UsdShade, Gf, Sdf
import omni.usd
import numpy as np
import math
import random

# ── [cmd_vel 블록 C-0] cmd_vel 제어기 import ──────────────────────────────
import rclpy
from test_cmd_vel_controller import JackalCmdVelController
# ── [cmd_vel 블록 C-0 끝] ──────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
#  1. 상수 설정
# ═══════════════════════════════════════════════════════════════════════════

WATER_DENSITY   = 1000.0
WATER_VISCOSITY = 0.001
GRAVITY         = 9.81
WATER_SURFACE_Z = 3.0

DRAG_Cd          = 1.05
RE_THRESHOLD     = 1000.0
SURFACE_GRID_RES = 3


# ═══════════════════════════════════════════════════════════════════════════
#  [파도 블록 A]
# ═══════════════════════════════════════════════════════════════════════════

WAVE_LAMBDA     = 8.0
WAVE_F_SURFACE  = 20.0
WAVE_F_VERTICAL = 4.0
WAVE_Z_DECAY    = 0.2
WAVE_UPDATE_SEC = 5.0
WAVE_AMP        = 0.15
WAVE_LEN        = 6.0
WAVE_SPEED      = 0.6

def _get_wave_height(x, y, t):
    k = 2.0 * math.pi / WAVE_LEN
    return WAVE_AMP * math.sin(k * (x + y) - WAVE_SPEED * t)

def _calc_wave_force(z_local, wave_dir):
    decay   = math.exp(-max(z_local, 0.0) / WAVE_Z_DECAY)
    f_horiz = WAVE_F_SURFACE * decay
    f_vert  = WAVE_F_VERTICAL * decay
    return np.array([wave_dir[0]*f_horiz, wave_dir[1]*f_horiz, f_vert],
                    dtype=np.float32)

# ═══════════════════════════════════════════════════════════════════════════
#  2. 포인트 및 법선 계산 함수
# ═══════════════════════════════════════════════════════════════════════════

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


def _generate_points_cylinder(radius: float, height: float,
                               grid_res: int = SURFACE_GRID_RES):
    volume      = np.pi * radius ** 2 * height
    points, normals = [], []
    r_steps     = np.linspace(0.0, radius, grid_res)
    theta_steps = np.linspace(0.0, 2 * np.pi, grid_res * 4, endpoint=False)
    z_steps     = np.linspace(-height / 2.0, height / 2.0, grid_res)
    for r in r_steps:
        for theta in theta_steps:
            for z in z_steps:
                pt   = np.array([r * np.cos(theta), r * np.sin(theta), z])
                norm = np.linalg.norm(pt)
                points.append(pt)
                normals.append(-pt / norm if norm > 1e-9 else None)
    n = len(points)
    return points, normals, volume / n, (np.pi * radius ** 2) / n


def _generate_points_cone(radius: float, height: float,
                           grid_res: int = SURFACE_GRID_RES):
    volume      = (1.0 / 3.0) * np.pi * radius ** 2 * height
    points, normals = [], []
    z_steps     = np.linspace(-height / 2.0, height / 2.0, grid_res)
    theta_steps = np.linspace(0.0, 2 * np.pi, grid_res * 4, endpoint=False)
    for z in z_steps:
        current_r = max(radius * (1.0 - (z + height / 2.0) / height), 1e-4)
        for r in np.linspace(0.0, current_r, grid_res):
            for theta in theta_steps:
                pt   = np.array([r * np.cos(theta), r * np.sin(theta), z])
                norm = np.linalg.norm(pt)
                points.append(pt)
                normals.append(-pt / norm if norm > 1e-9 else None)
    n = len(points)
    return points, normals, volume / n, (np.pi * radius ** 2) / n


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
    elif shape == "cylinder":
        return _generate_points_cylinder(kwargs["radius"], kwargs["height"])
    elif shape == "cone":
        return _generate_points_cone(kwargs["radius"], kwargs["height"])
    else:
        return _generate_points_sphere(kwargs.get("radius", kwargs.get("size", 0.5) / 2.0))


def rotation_matrix_from_quaternion(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


# ═══════════════════════════════════════════════════════════════════════════
#  3. 수압 계산 함수
# ═══════════════════════════════════════════════════════════════════════════

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
        pressure = WATER_DENSITY * GRAVITY * h_depth
        force    = pressure * area_per_point * (R @ n_local)
        force_list.append((pt_world, force))
    return force_list


# ═══════════════════════════════════════════════════════════════════════════
#  4. 항력 계산 함수
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
#  5. 물체 레지스트리
# ═══════════════════════════════════════════════════════════════════════════

_buoyancy_registry = []


def register_buoyancy_object(prim_path: str, shape: str, char_length: float, **kwargs):
    pts, norms, vol_per_pt, area_per_pt = generate_sample_points(shape, **kwargs)
    _buoyancy_registry.append({
        "prim_path":      prim_path,
        "shape":          shape,
        "char_length":    char_length,
        "local_points":   pts,
        "local_normals":  norms,
        "vol_per_point":  vol_per_pt,
        "area_per_point": area_per_pt,
    })
    print(f"[Buoyancy] 등록: {prim_path}  shape={shape}  pts={len(pts)}")


# ═══════════════════════════════════════════════════════════════════════════
#  6. 오브젝트 스폰 블록 A
# ═══════════════════════════════════════════════════════════════════════════

SPAWN_MASSES  = [20.0, 25.0, 30.0]
SPAWN_RADIUS  = 4.0
SPAWN_Z       = WATER_SURFACE_Z - 0.3
OBJECT_SIZE   = 0.5


def _ring_positions(n: int, radius: float, z: float, angle_offset: float = 0.0):
    positions = []
    for i in range(n):
        angle = angle_offset + 2.0 * math.pi * i / n
        positions.append(np.array([radius * math.cos(angle),
                                    radius * math.sin(angle), z]))
    return positions


def spawn_test_objects(world):
    N = len(SPAWN_MASSES)
    S = OBJECT_SIZE
    R = S / 2.0
    configs = [
        ("cube",     "Cube",     np.array([1.0, 0.4, 0.1]), S,   {"size": S},                  0),
        ("cylinder", "Cylinder", np.array([0.2, 0.8, 0.3]), R*2, {"radius": R, "height": S},  90),
        ("cone",     "Cone",     np.array([0.3, 0.5, 0.9]), R*2, {"radius": R, "height": S}, 180),
        ("sphere",   "Sphere",   np.array([0.8, 0.2, 0.8]), R*2, {"radius": R},              270),
    ]
    for shape, label, color, char_len, kw, angle_deg in configs:
        positions = _ring_positions(N, SPAWN_RADIUS, SPAWN_Z,
                                    angle_offset=math.radians(angle_deg))
        for i, (pos, mass) in enumerate(zip(positions, SPAWN_MASSES)):
            path = f"/World/{label}_{i}"
            if shape == "cube":
                world.scene.add(DynamicCuboid(
                    prim_path=path, name=f"{label.lower()}_{i}",
                    position=pos, scale=np.array([S, S, S]),
                    color=color, mass=mass,
                ))
            else:
                _spawn_dynamic_shape(world, shape, path, label, i, pos, color, S, mass)
            register_buoyancy_object(path, shape=shape, char_length=char_len, **kw)


def _spawn_dynamic_shape(world, shape, path, label, idx, pos, color, size, mass):
    stage = omni.usd.get_context().get_stage()
    R     = size / 2.0
    if shape == "cylinder":
        geom = UsdGeom.Cylinder.Define(stage, path)
        geom.CreateRadiusAttr(R); geom.CreateHeightAttr(size)
    elif shape == "cone":
        geom = UsdGeom.Cone.Define(stage, path)
        geom.CreateRadiusAttr(R); geom.CreateHeightAttr(size)
    elif shape == "sphere":
        geom = UsdGeom.Sphere.Define(stage, path)
        geom.CreateRadiusAttr(R)
    prim = stage.GetPrimAtPath(path)
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*pos.tolist()))
    geom.CreateDisplayColorAttr([Gf.Vec3f(*color.tolist())])
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    print(f"[Spawn] {shape} {path} at {pos}  mass={mass} kg")


# ═══════════════════════════════════════════════════════════════════════════
#  7. 메인 클래스
# ═══════════════════════════════════════════════════════════════════════════

class Water_Env(BaseSample):

    def __init__(self):
        super().__init__()
        self._rigid_prim_cache = {}

        # ── [파도 블록 B-1]
        self._wave_dir        = np.array([1.0, 0.0])
        self._wave_dir_target = self._wave_dir.copy()
        self._wave_timer      = 0.0
        self._wave_dir_sign   = 1.0
        # ── [파도 블록 B-1 끝]

        # ── [cmd_vel 블록 C-1]
        self._cmd_vel_ctrl = None   # JackalCmdVelController
        # ── [cmd_vel 블록 C-1 끝]

    # ── 7-1. setup_scene ──────────────────────────────────────────────────
    def setup_scene(self):
        world = self.get_world()
        stage = omni.usd.get_context().get_stage()

        _buoyancy_registry.clear()
        world.scene.add_default_ground_plane()

        physics_scene_prim = None
        for prim in stage.Traverse():
            if prim.GetTypeName() == "PhysicsScene":
                physics_scene_prim = prim
                break
        if physics_scene_prim is None:
            print("[Error] PhysicsScene not found.")
            return

        physics_scene = UsdPhysics.Scene(physics_scene_prim)
        physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(GRAVITY)

        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
        physx_scene.CreateEnableGPUDynamicsAttr(False)
        physx_scene.CreateEnableCCDAttr(False)
        physx_scene.CreateBroadphaseTypeAttr("MBP")
        physx_scene.CreateSolverTypeAttr("TGS")
        print("[OK] PhysicsScene configured")

        light_path = "/World/DistantLight"
        if not stage.GetPrimAtPath(light_path).IsValid():
            dl = UsdLux.DistantLight.Define(stage, Sdf.Path(light_path))
            dl.CreateIntensityAttr(1000)
            dl.CreateAngleAttr(0.53)
            UsdGeom.Xformable(dl.GetPrim()).AddRotateXYZOp().Set(
                Gf.Vec3f(-45.0, 0.0, 0.0))
            print("[OK] DistantLight added")

        spawn_test_objects(world)

        # ── [로봇 로드 블록 R-1] Jackal 로드만 (전극 결합 없음)
        _setup_jackal_robot(stage, world)

        # ── [수중 환경 블록 호출 E-U1]
        _setup_underwater_volume_vdb(stage)
        global EFFECTOR_PATH
        EFFECTOR_PATH = "/jackal/ur10/ee_link"
        _setup_camera_light_rig(stage)
        # ── [수중 환경 블록 호출 E-U1 끝]

    # ── 7-2. setup_post_load ──────────────────────────────────────────────
    async def setup_post_load(self):
        self._world = self.get_world()
        self._rigid_prim_cache.clear()

        for obj in _buoyancy_registry:
            path = obj["prim_path"]
            try:
                rp = RigidPrim(prim_paths_expr=path)
                self._rigid_prim_cache[path] = rp
            except Exception as e:
                print(f"[Warning] RigidPrim 초기화 실패: {path}  ({e})")

        self._world.add_physics_callback("buoyancy_step", self._buoyancy_step)
        print(f"[OK] buoyancy_step registered  ({len(_buoyancy_registry)} objects)")

        # ── [cmd_vel 블록 C-2] ROS2 + 제어기 초기화 ──────────────────────
        try:
            rclpy.init()
        except RuntimeError:
            pass  # 이미 초기화된 경우 무시
        self._cmd_vel_ctrl = JackalCmdVelController()
        print("[cmd_vel] ✓ 초기화 완료  →  /cmd_vel 수신 대기 중")
        # ── [cmd_vel 블록 C-2 끝] ─────────────────────────────────────────

    # ── 7-3. _buoyancy_step ───────────────────────────────────────────────
    def _buoyancy_step(self, step_size):

        # ── [파도 블록 B-2]
        self._wave_timer += step_size
        if self._wave_timer >= WAVE_UPDATE_SEC:
            self._wave_timer = 0.0
            angle = random.uniform(0.0, 2.0 * math.pi)
            self._wave_dir_target = np.array([math.cos(angle), math.sin(angle)])
            self._wave_dir_sign   = 1.0
            print(f"[Wave] 방향: {math.degrees(angle):.1f}°")
        if self._wave_timer >= WAVE_UPDATE_SEC * 0.5:
            self._wave_dir_sign = -1.0
        wave_dir_final  = self._wave_dir_target * self._wave_dir_sign
        self._wave_dir += (wave_dir_final - self._wave_dir) * 0.02
        self._wave_dir /= np.linalg.norm(self._wave_dir) + 1e-8
        wave_multiplier  = math.sin(2.0 * math.pi * self._wave_timer / WAVE_UPDATE_SEC)
        wave_multiplier *= (1.0 + 0.15 * math.sin(3.1 * self._wave_timer + random.uniform(0, 1)))
        # ── [파도 블록 B-2 끝]

        for obj in _buoyancy_registry:
            path = obj["prim_path"]
            rp   = self._rigid_prim_cache.get(path)
            if rp is None:
                continue

            positions, orientations = rp.get_world_poses()
            world_pos = positions[0]
            lin_vel   = rp.get_linear_velocities()[0]
            ang_vel   = rp.get_angular_velocities()[0]
            R         = rotation_matrix_from_quaternion(orientations[0])

            # ── [파도 블록 B-3]
            wave_h = _get_wave_height(float(world_pos[0]), float(world_pos[1]), self._wave_timer)
            surf_z = WATER_SURFACE_Z + wave_h
            # ── [파도 블록 B-3 끝]

            for pt_world, force_vec in compute_pressure_forces(
                    obj["local_points"], obj["local_normals"], obj["area_per_point"],
                    world_pos, R, surf_z):
                rp.apply_forces_and_torques_at_pos(
                    forces=force_vec.astype(np.float32).reshape(1, 3),
                    torques=np.zeros((1, 3), dtype=np.float32),
                    positions=pt_world.astype(np.float32).reshape(1, 3),
                    is_global=True)

            for pt_world, force_vec in compute_drag_forces(
                    obj["local_points"], obj["area_per_point"], obj["char_length"],
                    world_pos, R, surf_z, lin_vel, ang_vel):
                rp.apply_forces_and_torques_at_pos(
                    forces=force_vec.astype(np.float32).reshape(1, 3),
                    torques=np.zeros((1, 3), dtype=np.float32),
                    positions=pt_world.astype(np.float32).reshape(1, 3),
                    is_global=True)

            # ── [파도 블록 B-4]
            z_local = float(world_pos[2]) - surf_z
            f_wave  = _calc_wave_force(z_local, self._wave_dir) * wave_multiplier
            rp.apply_forces_and_torques_at_pos(
                forces=f_wave.reshape(1, 3).astype(np.float32),
                torques=np.zeros((1, 3), dtype=np.float32),
                positions=world_pos.astype(np.float32).reshape(1, 3),
                is_global=True)
            # ── [파도 블록 B-4 끝]

        # ── [cmd_vel 블록 C-3] 매 스텝 바퀴 제어 ────────────────────────
        if self._cmd_vel_ctrl is not None:
            self._cmd_vel_ctrl.update()
        # ── [cmd_vel 블록 C-3 끝] ─────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
#  [로봇 로드 블록] Jackal 차량만 (전극 결합 제거됨)
# ═══════════════════════════════════════════════════════════════════════════

def _setup_jackal_robot(stage, world):
    """
    Jackal + UR10 USD 로드 및 초기 위치 설정.
    ★ 원본과 달리 ee_link_path 반환값을 사용하지 않음 (전극 결합 제거).
    """
    robot_usd       = "/home/rokey/isaacsim/extension_examples/hello_world/jackal_and_ur10.usd"
    robot_prim_path = "/jackal"

    if not stage.GetPrimAtPath(robot_prim_path).IsValid():
        prim = stage.DefinePrim(robot_prim_path, "Xform")
        prim.GetReferences().AddReference(robot_usd)
        print(f"[Robot] Loaded USD: {robot_usd}")

    spawn_pos = Gf.Vec3d(0.0, -0.8, 0.2)
    spawn_rot = Gf.Vec3f(0.0, 0.0, 90.0)

    prim = stage.GetPrimAtPath(robot_prim_path)
    xf   = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(spawn_pos)
    xf.AddRotateXYZOp().Set(spawn_rot)
    print(f"[Robot] Spawn: {spawn_pos}")

    base_link = f"{robot_prim_path}/base_link"
    base_prim = stage.GetPrimAtPath(base_link)
    if base_prim.IsValid():
        mass_api = UsdPhysics.MassAPI.Apply(base_prim)
        mass_api.CreateMassAttr(200.0)
        print("[Robot] Base mass set to 200kg")

    print("[Robot] ✓ 로봇 로드 완료 (전극 결합 없음 — cmd_vel 제어 모드)")


# ═══════════════════════════════════════════════════════════════════════════
#  [수중 환경 블록]
# ═══════════════════════════════════════════════════════════════════════════

VOL_HALF_X      = 14.0
VOL_HALF_Y      = 14.0
VOL_HALF_Z      =  6.0
VOL_CENTER_Z    =  1.0
VOL_SCATTERING  = 0.85
VOL_ABSORPTION  = 0.85
VOL_ANISOTROPY  = -0.75
VOL_ALBEDO      = (0.04, 0.28, 0.45)

EFFECTOR_PATH     = "/World/EndEffector"
EFFECTOR_TEST_POS = (0.0, -0.5, 1.5)
EFFECTOR_TEST_ROT = (90.0, 0.0, 0.0)

CAM_LOCAL_POS     = (0.0,  0.05, 0.0)
CAM_LOCAL_ROT_XYZ = (0.0,  0.0,  0.0)
CAM_FOCAL_MM      = 18.0

LIGHT_LOCAL_POS   = (0.03, 0.05, 0.0)
LIGHT_INTENSITY   = 5000.0
LIGHT_COLOR       = (0.85, 0.95, 1.0)


def _setup_underwater_volume_vdb(stage):
    vol_path = "/World/UnderwaterVolume"
    mat_path = "/World/Looks/UnderwaterVolMat"

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

    print(f"[UnderwaterVDB] 볼륨 생성")


def _setup_camera_light_rig(stage):
    rig_path   = EFFECTOR_PATH + "/CameraRig"
    cam_path   = rig_path + "/Camera"
    light_path = rig_path + "/HeadLight"

    if not stage.GetPrimAtPath(EFFECTOR_PATH).IsValid():
        xf = UsdGeom.Xform.Define(stage, EFFECTOR_PATH)
        xformable = UsdGeom.Xformable(xf.GetPrim())
        xformable.AddTranslateOp().Set(Gf.Vec3d(*EFFECTOR_TEST_POS))
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(*EFFECTOR_TEST_ROT))
        print(f"[CameraRig] 더미 EndEffector 생성: {EFFECTOR_PATH}")

    if not stage.GetPrimAtPath(rig_path).IsValid():
        UsdGeom.Xform.Define(stage, rig_path)

    if not stage.GetPrimAtPath(cam_path).IsValid():
        cam = UsdGeom.Camera.Define(stage, cam_path)
        cam.CreateFocalLengthAttr(CAM_FOCAL_MM)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        xf = UsdGeom.Xformable(cam.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*CAM_LOCAL_POS))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(*CAM_LOCAL_ROT_XYZ))
        print(f"[CameraRig] Camera: {cam_path}")

    if not stage.GetPrimAtPath(light_path).IsValid():
        sl = UsdLux.SphereLight.Define(stage, light_path)
        sl.CreateIntensityAttr(LIGHT_INTENSITY)
        sl.CreateColorAttr(Gf.Vec3f(*LIGHT_COLOR))
        sl.CreateRadiusAttr(0.015)
        UsdGeom.Xformable(sl.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(*LIGHT_LOCAL_POS))
        print(f"[CameraRig] HeadLight: {light_path}")

    print(f"[CameraRig] 완료. 뷰포트 카메라: '{cam_path}'")
