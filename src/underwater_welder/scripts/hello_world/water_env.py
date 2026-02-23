"""
water_env.py
======================
Isaac Sim 5.0.0 / hello_world 예제 구조

구조
----
1. 상수 설정
        파도 블록 A
2. 포인트 및 법선 계산 함수
3. 수압 계산 함수
4. 항력 계산 함수
5. 물체 레지스트리
6. 오브젝트 스폰 블록 A
7. 메인 클래스
        파도 블록 B-1
    7-1. setup_scene
        조명 블록
        오브젝트 스폰 블록 B
        [용접 블록 호출 E-W1]
        [수중 환경 블록 호출 E-U1]  ← 추가
    7-2. setup_post_load
        [용접 블록 호출 E-W2]
    7-3. _buoyancy_step (스텝 콜백)
            파도 블록 B-2
            파도 블록 B-3
        7-3-1. 수압
        7-3-2. 항력
            파도 블록 B-4
        [용접 블록 호출 E-W3]
[용접 블록]
    W-1. 상수
    W-2. 용접 상태 머신
    W-3. spawn_weld_plate
    W-4. spawn_weld_electrode
    W-5. register_electrode_buoyancy
    W-6. WeldingSystem
    W-7. setup_welding (헬퍼)
[수중 환경 블록]              ← 추가
    U-1. 상수
    U-2. _setup_underwater_volume_vdb
    U-3. _setup_camera_light_rig
"""

from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import RigidPrim
# ↓ UsdVol, UsdShade 추가 — 수중 환경 블록에 필요
from pxr import UsdPhysics, PhysxSchema, UsdGeom, UsdLux, UsdVol, UsdShade, Gf, Sdf
import omni.usd
import numpy as np
import math
import random


# ═══════════════════════════════════════════════════════════════════════════
#  1. 상수 설정
# ═══════════════════════════════════════════════════════════════════════════

WATER_DENSITY   = 1000.0
WATER_VISCOSITY = 0.001
GRAVITY         = 9.8
WATER_SURFACE_Z = 3.0

DRAG_Cd          = 1.05
RE_THRESHOLD     = 1000.0
SURFACE_GRID_RES = 3


# ═══════════════════════════════════════════════════════════════════════════
#  [파도 블록 A]  ← 파도 비활성화 시 이 블록 전체 주석 처리
# ═══════════════════════════════════════════════════════════════════════════

WAVE_LAMBDA     = 8.0
WAVE_F_SURFACE  = 400.0
WAVE_F_VERTICAL = 80.0
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
#  [파도 블록 A 끝]
# ═══════════════════════════════════════════════════════════════════════════


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
    world.scene._scene_registry
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

        # ── [용접 블록 호출 E-W0]
        self._welding_system = None
        # ── [용접 블록 호출 E-W0 끝]

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
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

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

        # ── [용접 블록 호출 E-W1]
        setup_welding(world, stage)
        self._welding_system = WeldingSystem(stage)
        # ── [용접 블록 호출 E-W1 끝]

        # ── [수중 환경 블록 호출 E-U1]  ← 비활성화 시 이 두 줄 주석 처리
        _setup_underwater_volume_vdb(stage)
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

        # ── [용접 블록 호출 E-W2]
        if self._welding_system is not None:
            self._welding_system.init_rigid_prims()
        # ── [용접 블록 호출 E-W2 끝]

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
            # surf_z = WATER_SURFACE_Z   # ← 파도 비활성화 시 이 줄만 남김
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

        # ── [용접 블록 호출 E-W3]
        if self._welding_system is not None:
            self._welding_system.step(step_size)
        # ── [용접 블록 호출 E-W3 끝]


# ═══════════════════════════════════════════════════════════════════════════
#  [용접 블록]
# ═══════════════════════════════════════════════════════════════════════════

from enum import Enum, auto

# ── W-1. 상수 ──────────────────────────────────────────────────────────────

WELD_TRIGGER_DIST    = 0.035
WELD_HEAT_STEPS      = 30
WELD_ACTIVE_STEPS    = 20
WELD_COOLDOWN_STEPS  = 15

BEAD_POOL_RADIUS     = 0.015
BEAD_HEIGHT          = 0.003
BEAD_COLOR           = Gf.Vec3f(0.78, 0.78, 0.82)

SPARK_BATCH_SIZE      = 4
SPARK_MAX_ALIVE       = 6
SPARK_INTENSITY_PEAK  = 50000000.0
SPARK_RADIUS_LIGHT    = 0.003
SPARK_COLOR           = Gf.Vec3f(0.4, 0.6, 1.0)
SPARK_LIFETIME_STEPS  = 8
SPARK_JITTER          = 0.008
SPARK_GAP_RATIO       = 0.95

BUBBLE_COUNT          = 3
BUBBLE_RADIUS         = 0.008
BUBBLE_COLOR          = Gf.Vec3f(0.82, 0.92, 1.0)
BUBBLE_RISE_SPEED     = 0.05
BUBBLE_LIFETIME_STEPS = 60

PLATE_PATH            = "/World/WeldPlate"
PLATE_SIZE_X          = 1.2
PLATE_SIZE_Z          = 0.8
PLATE_THICKNESS       = 0.012
PLATE_MASS            = 60.0
PLATE_COLOR           = Gf.Vec3f(0.35, 0.35, 0.40)
PLATE_POS             = np.array([0.0, 0.5, WATER_SURFACE_Z - 1.5])
PLATE_FRONT_Y         = float(PLATE_POS[1]) - PLATE_THICKNESS / 2.0

ELECTRODE_KINEMATIC   = True
# ELECTRODE_KINEMATIC = False

ELECTRODE_PATH        = "/World/WeldElectrode"
ELECTRODE_RADIUS      = 0.006
ELECTRODE_HEIGHT      = 0.35
ELECTRODE_MASS        = 0.18
ELECTRODE_COLOR       = Gf.Vec3f(0.55, 0.50, 0.12)
ELECTRODE_TIP_COLOR   = Gf.Vec3f(1.0, 0.08, 0.08)
ELECTRODE_TIP_RADIUS  = ELECTRODE_RADIUS * 1.6
ELECTRODE_TIP_PATH    = ELECTRODE_PATH + "/TipMarker"
ELECTRODE_POS         = np.array([
    0.0,
    PLATE_FRONT_Y - ELECTRODE_HEIGHT / 2.0 - 0.05,
    float(PLATE_POS[2]),
])
ELECTRODE_ROT_X_DEG   = -90.0


# ── W-2. 용접 상태 머신 ────────────────────────────────────────────────────

class WeldState(Enum):
    IDLE     = auto()
    HEATING  = auto()
    WELDING  = auto()
    COOLDOWN = auto()


# ── W-3. spawn_weld_plate ──────────────────────────────────────────────────

def spawn_weld_plate(world, stage):
    if stage.GetPrimAtPath(PLATE_PATH).IsValid():
        print("[Weld] WeldPlate already exists, skipping.")
        return
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
    print(f"[Weld] WeldPlate spawned  pos={PLATE_POS}  [kinematic]")


# ── W-4. spawn_weld_electrode ──────────────────────────────────────────────

def spawn_weld_electrode(world, stage):
    if stage.GetPrimAtPath(ELECTRODE_PATH).IsValid():
        print("[Weld] WeldElectrode already exists, skipping.")
        return
    geom = UsdGeom.Cylinder.Define(stage, ELECTRODE_PATH)
    geom.CreateRadiusAttr(ELECTRODE_RADIUS)
    geom.CreateHeightAttr(ELECTRODE_HEIGHT)
    geom.CreateDisplayColorAttr([ELECTRODE_COLOR])
    prim = stage.GetPrimAtPath(ELECTRODE_PATH)
    xf   = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(*ELECTRODE_POS.tolist()))
    xf.AddRotateXOp().Set(ELECTRODE_ROT_X_DEG)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(ELECTRODE_MASS)
    if ELECTRODE_KINEMATIC:
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
        print("[Weld] Electrode → kinematic")
    else:
        print("[Weld] Electrode → dynamic")
    tip = UsdGeom.Sphere.Define(stage, ELECTRODE_TIP_PATH)
    tip.CreateRadiusAttr(ELECTRODE_TIP_RADIUS)
    tip.CreateDisplayColorAttr([ELECTRODE_TIP_COLOR])
    UsdGeom.Xformable(stage.GetPrimAtPath(ELECTRODE_TIP_PATH)).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, ELECTRODE_HEIGHT / 2.0))
    print(f"[Weld] WeldElectrode spawned  pos={ELECTRODE_POS}")


# ── W-5. register_electrode_buoyancy ──────────────────────────────────────

def register_electrode_buoyancy():
    register_buoyancy_object(
        prim_path=ELECTRODE_PATH, shape="cylinder",
        char_length=ELECTRODE_RADIUS * 2.0,
        radius=ELECTRODE_RADIUS, height=ELECTRODE_HEIGHT)


# ── W-6. WeldingSystem ────────────────────────────────────────────────────

class WeldingSystem:

    def __init__(self, stage):
        self._stage          = stage
        self._state          = WeldState.IDLE
        self._heat_counter   = 0
        self._active_counter = 0
        self._cool_counter   = 0
        self._bead_idx       = 0
        self._fx_idx         = 0
        self._spark_prims    = []
        self._bubble_prims   = []
        self._electrode_rp: RigidPrim | None = None

    def init_rigid_prims(self):
        try:
            self._electrode_rp = RigidPrim(prim_paths_expr=ELECTRODE_PATH)
            print("[Weld] Electrode RigidPrim OK")
        except Exception as e:
            print(f"[Weld][Warning] Electrode RigidPrim 실패: {e}")

    def step(self, step_size: float):
        self._tick_particles()
        if self._electrode_rp is None:
            return
        tip_pos, electrode_dir = self._get_tip_and_direction()
        if tip_pos is None:
            return
        dist = self._dist_to_plate(tip_pos)

        if self._state == WeldState.IDLE:
            if dist <= WELD_TRIGGER_DIST:
                self._state = WeldState.HEATING
                self._heat_counter = 1
                print(f"[Weld] IDLE → HEATING  dist={dist*100:.1f} cm")

        elif self._state == WeldState.HEATING:
            if dist <= WELD_TRIGGER_DIST:
                self._heat_counter += 1
                if self._heat_counter >= WELD_HEAT_STEPS:
                    self._state = WeldState.WELDING
                    self._active_counter = 0
                    print(f"[Weld] HEATING → WELDING")
            else:
                print(f"[Weld] HEATING 취소 → IDLE")
                self._state = WeldState.IDLE

        elif self._state == WeldState.WELDING:
            self._active_counter += 1
            if self._active_counter == 1:
                self._spawn_weld_bead(tip_pos, electrode_dir)
            self._spawn_sparks(tip_pos)
            if self._active_counter >= WELD_ACTIVE_STEPS:
                self._state = WeldState.COOLDOWN
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
            positions, orientations = self._electrode_rp.get_world_poses()
            center   = np.array(positions[0])
            R        = rotation_matrix_from_quaternion(np.array(orientations[0]))
            tip_pos  = center + R @ np.array([0.0, 0.0, ELECTRODE_HEIGHT / 2.0])
            elec_dir = R @ np.array([0.0, 0.0, 1.0])
            return tip_pos, elec_dir
        except Exception as e:
            print(f"[Weld][Warning] tip 계산 실패: {e}")
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
            d = dir_on_plate / dir_len
            sx *= 1.0 + (stretch - 1.0) * abs(d[0])
            sz *= 1.0 + (stretch - 1.0) * abs(d[2])
        if not self._stage.GetPrimAtPath(path).IsValid():
            sphere = UsdGeom.Sphere.Define(self._stage, path)
            sphere.CreateRadiusAttr(1.0)
            sphere.CreateDisplayColorAttr([BEAD_COLOR])
            xf = UsdGeom.Xformable(self._stage.GetPrimAtPath(path))
            xf.AddTranslateOp().Set(Gf.Vec3d(*bead_pos.tolist()))
            xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
            print(f"[Weld] 비드: {path}")

    def _spawn_sparks(self, origin: np.ndarray):
        plate_contact = origin.copy()
        plate_contact[1] = PLATE_FRONT_Y
        spark_origin = plate_contact + (origin - plate_contact) * SPARK_GAP_RATIO
        alive = len(self._spark_prims)
        if alive >= SPARK_MAX_ALIVE:
            return
        spawn_n  = min(SPARK_BATCH_SIZE, SPARK_MAX_ALIVE - alive)
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
            rise_vel = np.array([random.uniform(-0.003, 0.003),
                                  random.uniform(-0.003, 0.003),
                                  BUBBLE_RISE_SPEED * random.uniform(0.8, 1.2)])
            self._bubble_prims.append([path, BUBBLE_LIFETIME_STEPS, rise_vel])

    def _tick_particles(self):
        spark_survivors = []
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
            spark_survivors.append([path, life])
        self._spark_prims = spark_survivors

        bubble_survivors = []
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
            bubble_survivors.append([path, life, vel])
        self._bubble_prims = bubble_survivors


# ── W-7. setup_welding ────────────────────────────────────────────────────

def setup_welding(world, stage):
    spawn_weld_plate(world, stage)
    spawn_weld_electrode(world, stage)
    register_electrode_buoyancy()
    print("[Weld] setup_welding 완료")

# ═══════════════════════════════════════════════════════════════════════════
#  [용접 블록 끝]
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
#  [수중 환경 블록]
#  비활성화: 이 블록 전체 + setup_scene() 내 E-U1 두 줄 주석 처리.
#
#  ★ 이 블록의 두 함수는 반드시 클래스 밖 모듈 레벨에 위치해야 합니다.
#    클래스 안에 들여쓰면 self 없는 호출이 불가능해집니다.
# ═══════════════════════════════════════════════════════════════════════════

# ── U-1. 상수 ──────────────────────────────────────────────────────────────

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


# ── U-2. _setup_underwater_volume_vdb ─────────────────────────────────────

def _setup_underwater_volume_vdb(stage):
    """
    씬 전체를 덮는 OmniVolume 수중 볼류메트릭 산란 박스.
    RTX 렌더 모드 필수.
    재실행 안전 (이미 존재하면 스킵).
    """
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

    print(f"[UnderwaterVDB] 볼륨 생성  scatter={VOL_SCATTERING}  "
          f"absorb={VOL_ABSORPTION}  anisotropy={VOL_ANISOTROPY}")


# ── U-3. _setup_camera_light_rig ──────────────────────────────────────────

def _setup_camera_light_rig(stage):
    """
    EFFECTOR_PATH 자식으로 카메라 + SphereLight 리그 생성.
    EFFECTOR_PATH 가 없으면 더미 Xform 자동 생성.

    Stage 트리 확인 경로:
        /World/EndEffector/CameraRig/Camera
        /World/EndEffector/CameraRig/HeadLight

    카메라 시점 전환:
        뷰포트 카메라 선택 드롭다운 →
        "/World/EndEffector/CameraRig/Camera" 선택
    """
    rig_path   = EFFECTOR_PATH + "/CameraRig"
    cam_path   = rig_path + "/Camera"
    light_path = rig_path + "/HeadLight"

    # 엔드 이펙터 없으면 더미 생성
    if not stage.GetPrimAtPath(EFFECTOR_PATH).IsValid():
        xf = UsdGeom.Xform.Define(stage, EFFECTOR_PATH)
        xformable = UsdGeom.Xformable(xf.GetPrim())
        xformable.AddTranslateOp().Set(Gf.Vec3d(*EFFECTOR_TEST_POS))
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(*EFFECTOR_TEST_ROT)) # 회전(orient) 연산 추가
        print(f"[CameraRig] 더미 EndEffector 생성: {EFFECTOR_PATH}  "
              f"pos={EFFECTOR_TEST_POS}")

    if not stage.GetPrimAtPath(rig_path).IsValid():
        UsdGeom.Xform.Define(stage, rig_path)

    if not stage.GetPrimAtPath(cam_path).IsValid():
        cam = UsdGeom.Camera.Define(stage, cam_path)
        cam.CreateFocalLengthAttr(CAM_FOCAL_MM)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        xf = UsdGeom.Xformable(cam.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*CAM_LOCAL_POS))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(*CAM_LOCAL_ROT_XYZ))
        print(f"[CameraRig] Camera 생성: {cam_path}")
    else:
        print(f"[CameraRig] Camera 이미 존재: {cam_path}")

    if not stage.GetPrimAtPath(light_path).IsValid():
        sl = UsdLux.SphereLight.Define(stage, light_path)
        sl.CreateIntensityAttr(LIGHT_INTENSITY)
        sl.CreateColorAttr(Gf.Vec3f(*LIGHT_COLOR))
        sl.CreateRadiusAttr(0.015)
        UsdGeom.Xformable(sl.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(*LIGHT_LOCAL_POS))
        print(f"[CameraRig] HeadLight 생성: {light_path}")

    print(f"[CameraRig] 완료. 뷰포트 카메라 드롭다운에서 '{cam_path}' 선택하세요.")

# ═══════════════════════════════════════════════════════════════════════════
#  [수중 환경 블록 끝]
# ═══════════════════════════════════════════════════════════════════════════
