#!/usr/bin/env python3
"""
수중 환경 모듈 (jackal_ur10_demo.py 에서 분리)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UnderwaterEnvironment 클래스:
  - 조명/해저면/용접 타겟
  - 버블/잔해 애니메이션
  - 부력/파도 물리 (gravity 스케일링 + PhysX per-step force)
  - 바닥/바퀴 마찰 재질

사용 예:
  from underwater_env import UnderwaterEnvironment
  env = UnderwaterEnvironment(world, robot_body_path="/jackal/base_link")
  env.create(stage)
  world.add_physics_callback("wave", env.physics_step_callback)
  asyncio.ensure_future(env.animate_bubbles())
  asyncio.ensure_future(env.animate_debris())
"""

import math
import random
import asyncio

import carb
from omni.physx import get_physx_simulation_interface
from pxr import (
    UsdPhysics, PhysxSchema, UsdGeom, UsdLux,
    UsdUtils, PhysicsSchemaTools, UsdShade, Gf
)
import numpy as np

# ── 수중 물리 상수 ────────────────────────────────────────────────────────
WATER_DENSITY   = 1025.0
GRAVITY_CONST   = 9.81
WAVE_F_ROBOT    = 150.0
WAVE_UPDATE_SEC = 6.0

JACKAL_WIDTH   = 0.43
JACKAL_LENGTH  = 0.55
JACKAL_HEIGHT  = 0.32
JACKAL_MASS_KG = 200.0
JACKAL_VOLUME  = 0.155   # 부력용 체적 (eff_g ≈ 2 m/s²)

_C_d = 1.2
_A_frontal = JACKAL_WIDTH * JACKAL_HEIGHT
JACKAL_LINEAR_DAMPING  = 0.5 * _C_d * WATER_DENSITY * _A_frontal
JACKAL_ANGULAR_DAMPING = 30.0


class UnderwaterEnvironment:
    """수중 환경 일체 관리 클래스"""

    def __init__(self, body_prim_path: str = "/jackal/base_link"):
        self.body_prim_path = body_prim_path

        # 물리 캐시
        self._physx_iface   = None
        self._stage_id      = None
        self._body_path_int = None
        self._physics_ready = False

        # 파도 파라미터
        self._wave_dir        = np.array([1.0, 0.0])
        self._wave_dir_target = self._wave_dir.copy()
        self._wave_timer      = 0.0
        self._wave_dir_sign   = 1.0

        # 버블/잔해
        self.bubbles = []
        self.debris  = []

    # ── 환경 생성 ──────────────────────────────────────────────────────────
    def create(self, stage):
        """전체 수중 환경 생성 (World 생성 전에 호출)"""
        self._cleanup(stage)
        self._create_dome_light(stage)
        self._create_seafloor(stage)
        self._create_weld_target(stage)
        self._create_bubbles(stage)
        self._create_debris(stage)
        self._setup_physics_scene(stage)
        self._setup_underwater_physics(stage)
        self._set_jackal_mass(stage)
        self._setup_wheel_friction(stage)
        print("  ✓ 수중 환경 생성 완료")

    def _cleanup(self, stage):
        for path in ["/World/UnderwaterDome", "/World/Seafloor",
                     "/World/WeldTarget", "/World/Bubbles", "/World/Debris"]:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                stage.RemovePrim(path)
        self.bubbles.clear()
        self.debris.clear()

    def _create_dome_light(self, stage):
        dome = UsdLux.DomeLight.Define(stage, "/World/UnderwaterDome")
        dome.CreateColorAttr(Gf.Vec3f(0.04, 0.15, 0.35))
        dome.CreateIntensityAttr(700.0)

    def _create_seafloor(self, stage):
        floor = UsdGeom.Cube.Define(stage, "/World/Seafloor")
        floor.CreateSizeAttr(1.0)
        UsdGeom.Xformable(floor).AddScaleOp().Set(Gf.Vec3f(30.0, 30.0, 0.05))
        UsdGeom.Xformable(floor).AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, -0.03))
        floor.CreateDisplayColorAttr([(0.20, 0.17, 0.12)])

    def _create_weld_target(self, stage):
        pipe = UsdGeom.Cylinder.Define(stage, "/World/WeldTarget")
        pipe.CreateRadiusAttr(0.08)
        pipe.CreateHeightAttr(1.0)
        pipe.CreateAxisAttr("Z")
        UsdGeom.Xformable(pipe).AddTranslateOp().Set(Gf.Vec3f(3.0, 2.0, 0.3))
        pipe.CreateDisplayColorAttr([(0.55, 0.55, 0.60)])
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/World/WeldTarget"))

    def _create_bubbles(self, stage):
        random.seed(42)
        for i in range(25):
            path = f"/World/Bubbles/bubble_{i:02d}"
            sph = UsdGeom.Sphere.Define(stage, path)
            r = random.uniform(0.015, 0.06)
            sph.CreateRadiusAttr(r)
            x = random.uniform(-5.0, 5.0)
            y = random.uniform(-5.0, 5.0)
            z = random.uniform(-0.5, 5.0)
            UsdGeom.Xformable(sph).AddTranslateOp().Set(Gf.Vec3f(x, y, z))
            sph.CreateDisplayColorAttr([(0.65, 0.82, 1.0)])
            self.bubbles.append({"path": path, "x": x, "y": y, "z": z,
                                  "speed": random.uniform(0.25, 0.8), "r": r})

    def _create_debris(self, stage):
        random.seed(77)
        configs = [
            ("float",   4, +0.30, (0.80, 0.60, 0.25), 0.10),
            ("neutral", 3, +0.01, (0.50, 0.62, 0.65), 0.08),
            ("sink",    4, -0.22, (0.40, 0.40, 0.45), 0.09),
        ]
        for dtype, count, net_buoy, color, size in configs:
            for i in range(count):
                path = f"/World/Debris/{dtype}_{i:02d}"
                cube = UsdGeom.Cube.Define(stage, path)
                cube.CreateSizeAttr(size)
                x = random.uniform(-5.0, 5.0)
                y = random.uniform(-5.0, 5.0)
                z = random.uniform(0.5, 4.5)
                UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3f(x, y, z))
                cube.CreateDisplayColorAttr([color])
                self.debris.append({"path": path, "net_buoy": net_buoy,
                                    "x": x, "y": y, "z": z,
                                    "vx": 0.0, "vy": 0.0, "vz": 0.0})

    def _setup_physics_scene(self, stage):
        buoyancy = WATER_DENSITY * JACKAL_VOLUME * GRAVITY_CONST
        weight   = JACKAL_MASS_KG * GRAVITY_CONST
        eff_g    = max(0.5, GRAVITY_CONST * (1.0 - buoyancy / weight))

        for candidate in ["/Environment/physicsScene", "/physicsScene", "/World/physicsScene"]:
            prim = stage.GetPrimAtPath(candidate)
            if prim.IsValid():
                UsdPhysics.Scene(prim).CreateGravityMagnitudeAttr().Set(eff_g)
                print(f"  ✓ eff_gravity={eff_g:.2f} m/s² ({candidate})")
                return

        scene_path = "/physicsScene"
        scene = UsdPhysics.Scene.Define(stage, scene_path)
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(eff_g)
        sc_prim = stage.GetPrimAtPath(scene_path)
        ps = PhysxSchema.PhysxSceneAPI.Apply(sc_prim)
        ps.CreateEnableCCDAttr(True)
        ps.CreateEnableStabilizationAttr(True)
        print(f"  ✓ eff_gravity={eff_g:.2f} m/s² (new scene)")

        # 바닥 마찰
        mat_prim = UsdShade.Material.Define(stage, "/World/GroundFrictionMat").GetPrim()
        pm = UsdPhysics.MaterialAPI.Apply(mat_prim)
        pm.CreateStaticFrictionAttr().Set(1.5)
        pm.CreateDynamicFrictionAttr().Set(1.2)
        pm.CreateRestitutionAttr().Set(0.0)
        for gnd_path in ["/Environment/GroundPlane/CollisionPlane",
                          "/World/defaultGroundPlane"]:
            gnd = stage.GetPrimAtPath(gnd_path)
            if gnd.IsValid():
                UsdShade.MaterialBindingAPI.Apply(gnd).Bind(
                    UsdShade.Material(mat_prim),
                    UsdShade.Tokens.weakerThanDescendants, "physics")
                break

    def _setup_underwater_physics(self, stage):
        body_prim = stage.GetPrimAtPath(self.body_prim_path)
        if not body_prim.IsValid():
            return
        try:
            physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(body_prim)
            physx_rb.CreateLinearDampingAttr().Set(JACKAL_LINEAR_DAMPING)
            physx_rb.CreateAngularDampingAttr().Set(JACKAL_ANGULAR_DAMPING + 70.0)
        except Exception as e:
            print(f"  ⚠ Damping error: {e}")
        try:
            import omni.usd
            self._physx_iface   = get_physx_simulation_interface()
            self._stage_id      = UsdUtils.StageCache.Get().GetId(
                omni.usd.get_context().get_stage()).ToLongInt()
            self._body_path_int = PhysicsSchemaTools.sdfPathToInt(body_prim.GetPath())
        except Exception as e:
            print(f"  ⚠ PhysX interface: {e}")

    def _set_jackal_mass(self, stage):
        body_prim = stage.GetPrimAtPath(self.body_prim_path)
        if not body_prim.IsValid():
            return
        mass_api = UsdPhysics.MassAPI.Apply(body_prim)
        mass_api.GetMassAttr().Set(JACKAL_MASS_KG)
        mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, -JACKAL_HEIGHT * 0.3))

    def _setup_wheel_friction(self, stage):
        wheel_mat = stage.GetPrimAtPath("/jackal/PhysicsMaterials/wheels")
        if wheel_mat.IsValid():
            pm = UsdPhysics.MaterialAPI.Apply(wheel_mat)
            pm.CreateStaticFrictionAttr().Set(2.0)
            pm.CreateDynamicFrictionAttr().Set(1.5)

    # ── 물리 콜백 (파도력) ─────────────────────────────────────────────────
    def activate(self):
        """물리 콜백 활성화 (World 초기화 후 호출)"""
        self._physics_ready = True

    def physics_step_callback(self, step_size):
        if not self._physics_ready:
            return
        self._wave_timer += step_size
        if self._wave_timer >= WAVE_UPDATE_SEC:
            self._wave_timer = 0.0
            angle = random.uniform(0.0, 2.0 * math.pi)
            self._wave_dir_target = np.array([math.cos(angle), math.sin(angle)])
            self._wave_dir_sign = 1.0
        if self._wave_timer >= WAVE_UPDATE_SEC * 0.5:
            self._wave_dir_sign = -1.0

        wave_dir_final = self._wave_dir_target * self._wave_dir_sign
        self._wave_dir += (wave_dir_final - self._wave_dir) * 0.015
        norm = np.linalg.norm(self._wave_dir)
        if norm > 1e-8:
            self._wave_dir /= norm

        if self._physx_iface is not None:
            t = self._wave_timer
            wave_mul = (math.sin(2.0 * math.pi * t / WAVE_UPDATE_SEC)
                        * (1.0 + 0.12 * math.sin(2.9 * t)))
            fx = float(self._wave_dir[0] * WAVE_F_ROBOT * wave_mul)
            fy = float(self._wave_dir[1] * WAVE_F_ROBOT * wave_mul)
            if math.isfinite(fx) and math.isfinite(fy):
                try:
                    self._physx_iface.apply_force_at_pos(
                        self._stage_id, self._body_path_int,
                        carb.Float3(fx, fy, 0.0),
                        carb.Float3(0.0, 0.0, 0.0), "Force"
                    )
                except Exception:
                    pass

    # ── 애니메이션 ─────────────────────────────────────────────────────────
    async def animate_bubbles(self):
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        dt = 0.033
        while True:
            for b in self.bubbles:
                b["z"] += b["speed"] * dt
                if b["z"] > 8.0:
                    b["z"] = -0.5
                    b["x"] = random.uniform(-5.0, 5.0)
                    b["y"] = random.uniform(-5.0, 5.0)
                prim = stage.GetPrimAtPath(b["path"])
                if prim.IsValid():
                    attr = prim.GetAttribute("xformOp:translate")
                    if attr:
                        attr.Set(Gf.Vec3f(b["x"], b["y"], b["z"]))
            await asyncio.sleep(dt)

    async def animate_debris(self):
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        dt, DRAG = 0.033, 1.8
        while True:
            t = self._wave_timer
            wave_mul = math.sin(2.0 * math.pi * t / WAVE_UPDATE_SEC)
            wave_mul *= 1.0 + 0.12 * math.sin(2.9 * t)
            WAVE_A = 0.15
            cur_x = self._wave_dir[0] * WAVE_A * wave_mul
            cur_y = self._wave_dir[1] * WAVE_A * wave_mul
            for d in self.debris:
                d["vz"] += (d["net_buoy"] - DRAG * d["vz"]) * dt
                d["vx"] += (cur_x - DRAG * 0.2 * d["vx"]) * dt
                d["vy"] += (cur_y - DRAG * 0.2 * d["vy"]) * dt
                d["z"]  += d["vz"] * dt
                d["x"]  = max(-6.0, min(d["x"] + d["vx"]*dt, 6.0))
                d["y"]  = max(-6.0, min(d["y"] + d["vy"]*dt, 6.0))
                d["z"]  = max(-0.5, min(d["z"], 7.0))
                prim = stage.GetPrimAtPath(d["path"])
                if prim.IsValid():
                    attr = prim.GetAttribute("xformOp:translate")
                    if attr:
                        attr.Set(Gf.Vec3f(d["x"], d["y"], d["z"]))
            await asyncio.sleep(dt)
