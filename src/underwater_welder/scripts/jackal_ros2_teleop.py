#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════════════════
Jackal + UR10 ROS2 Teleop 데모
══════════════════════════════════════════════════════════════════════════

jackal_ur10_demo.py 의 자동 시퀀스 대신 ROS2 /cmd_vel 으로 수동 조종.
팀원의 Action Graph (Camera + JointStates) 와 함께 동작.

실행:
  /home/rokey/isaacsim/python.sh jackal_ros2_teleop.py

조종:
  ros2 topic pub /cmd_vel geometry_msgs/Twist \
    "{linear: {x: 0.5}, angular: {z: 0.0}}" --once

  또는 teleop_twist_keyboard:
  ros2 run teleop_twist_keyboard teleop_twist_keyboard
══════════════════════════════════════════════════════════════════════════
"""

import sys
import math
import random

from isaacsim import SimulationApp

CONFIG = {
    "headless": False,
    "width": 1920,
    "height": 1080,
}
simulation_app = SimulationApp(CONFIG)

import omni.usd
import carb
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.physx import get_physx_simulation_interface
from pxr import (
    UsdPhysics, PhysxSchema, Gf, UsdGeom, UsdLux,
    UsdUtils, PhysicsSchemaTools, Sdf, UsdShade
)
import numpy as np

# ── UDP 소켓 브릿지 (rclpy 불필요, Python 버전 충돌 없음) ────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vehicle_control'))
from ros2_vehicle_bridge import init_bridge, spin_once, shutdown_bridge

# ═════════════════════════════════════════════════════════════════════════
#  설정 상수 (jackal_ur10_demo.py 와 동일)
# ═════════════════════════════════════════════════════════════════════════
USD_PATH = "/home/rokey/Downloads/jackal_and_ur10.usd"

WATER_DENSITY   = 1025.0
GRAVITY_CONST   = 9.81
WAVE_F_ROBOT    = 100.0    # 수동 조종 시 파도력 약하게
WAVE_UPDATE_SEC = 6.0

JACKAL_WIDTH    = 0.43 * 3.0   # 1.29 m: 실제 치수 × 3.0× 스케일
JACKAL_HEIGHT   = 0.32 * 3.0   # 0.96 m
JACKAL_MASS_KG  = 500.0        # 500 kg 유지
JACKAL_VOLUME   = 0.155        # 부력 체적 유지 (eff_g ≈ 6.7 m/s² 유지)

_C_d           = 1.2
_A_frontal     = JACKAL_WIDTH * JACKAL_HEIGHT
JACKAL_LINEAR_DAMPING  = 0.5 * _C_d * WATER_DENSITY * _A_frontal
JACKAL_ANGULAR_DAMPING = 100.0

BODY_PRIM_PATH = "/jackal/base_link"

WHEEL_JOINT_PATHS = [
    "/jackal/front_left_wheel_joint",
    "/jackal/front_right_wheel_joint",
    "/jackal/rear_left_wheel_joint",
    "/jackal/rear_right_wheel_joint",
]

UR10_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# ── UR10 초기 자세 (도 단위) ──────────────────────────────────────────────
# 여기서 값을 바꾸면 스폰 시 팔 자세가 변경됨
SPAWN_ARM_POSE_DEG = {
    "shoulder_pan_joint":   0.0,
    "shoulder_lift_joint":  0.0,   # 0 = 앞으로 수평 (안전 스폰), -90 = 위로 세움
    "elbow_joint":          0.0,
    "wrist_1_joint":        0.0,
    "wrist_2_joint":        0.0,
    "wrist_3_joint":        0.0,
}


# ═════════════════════════════════════════════════════════════════════════
class JackalROS2Teleop:
    def __init__(self):
        self.world          = None
        self.robot          = None
        self.bridge         = None          # ROS2 브릿지
        self._physx_iface   = None
        self._stage_id      = None
        self._body_path_int = None
        self._physics_ready = False
        self._wave_timer    = 0.0
        self._wave_dir      = np.array([1.0, 0.0])
        self._wave_dir_target = np.array([1.0, 0.0])
        self._wave_dir_sign = 1.0

    # ── 물리 씬 설정 ──────────────────────────────────────────────────────
    def setup_physics(self):
        stage = omni.usd.get_context().get_stage()

        # 중력 (부력 보상)
        buoyancy = WATER_DENSITY * JACKAL_VOLUME * GRAVITY_CONST
        weight   = JACKAL_MASS_KG * GRAVITY_CONST
        eff_g    = max(0.5, GRAVITY_CONST * (1.0 - buoyancy / weight))

        for candidate in ["/Environment/physicsScene", "/physicsScene"]:
            sp = stage.GetPrimAtPath(candidate)
            if sp.IsValid():
                scene = UsdPhysics.Scene(sp)
                scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
                scene.CreateGravityMagnitudeAttr().Set(eff_g)
                print(f"  ✓ gravity={eff_g:.2f} m/s²  ({candidate})")
                break

        # damping
        body_prim = stage.GetPrimAtPath(BODY_PRIM_PATH)
        if body_prim.IsValid():
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(body_prim)
            rb.CreateLinearDampingAttr().Set(JACKAL_LINEAR_DAMPING)
            rb.CreateAngularDampingAttr().Set(JACKAL_ANGULAR_DAMPING)
            print(f"  ✓ linear_damping={JACKAL_LINEAR_DAMPING:.1f}")

        # 질량
        if body_prim.IsValid():
            ma = UsdPhysics.MassAPI.Apply(body_prim)
            ma.GetMassAttr().Set(JACKAL_MASS_KG)
            ma.GetCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, -JACKAL_HEIGHT * 0.3))
            print(f"  ✓ mass={JACKAL_MASS_KG} kg")

        # 바퀴 마찰
        wmat = stage.GetPrimAtPath("/jackal/PhysicsMaterials/wheels")
        if wmat.IsValid():
            pm = UsdPhysics.MaterialAPI.Apply(wmat)
            pm.CreateStaticFrictionAttr().Set(2.0)
            pm.CreateDynamicFrictionAttr().Set(1.5)

        # 바닥 마찰
        mat_prim = UsdShade.Material.Define(stage, "/World/GroundFrictionMat").GetPrim()
        pm2 = UsdPhysics.MaterialAPI.Apply(mat_prim)
        pm2.CreateStaticFrictionAttr().Set(1.5)
        pm2.CreateDynamicFrictionAttr().Set(1.2)
        pm2.CreateRestitutionAttr().Set(0.0)
        for gnd in ["/Environment/GroundPlane/CollisionPlane",
                    "/World/defaultGroundPlane/GroundPlane/CollisionPlane"]:
            gp = stage.GetPrimAtPath(gnd)
            if gp.IsValid():
                UsdShade.MaterialBindingAPI.Apply(gp).Bind(
                    UsdShade.Material(mat_prim),
                    UsdShade.Tokens.weakerThanDescendants, "physics")
                print(f"  ✓ ground friction applied ({gnd})")
                break

        # PhysX force 인터페이스 캐시
        try:
            self._physx_iface   = get_physx_simulation_interface()
            self._stage_id      = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
            self._body_path_int = PhysicsSchemaTools.sdfPathToInt(
                body_prim.GetPath())
        except Exception as e:
            print(f"  ⚠ PhysX interface: {e}")

    # ── 파도력 콜백 ───────────────────────────────────────────────────────
    def physics_step_callback(self, step_size):
        if not self._physics_ready:
            return
        self._wave_timer += step_size
        if self._wave_timer >= WAVE_UPDATE_SEC:
            self._wave_timer = 0.0
            a = random.uniform(0.0, 2.0 * math.pi)
            self._wave_dir_target = np.array([math.cos(a), math.sin(a)])
            self._wave_dir_sign   = 1.0
        if self._wave_timer >= WAVE_UPDATE_SEC * 0.5:
            self._wave_dir_sign = -1.0

        self._wave_dir += (self._wave_dir_target * self._wave_dir_sign
                           - self._wave_dir) * 0.015
        n = np.linalg.norm(self._wave_dir)
        if n > 1e-8:
            self._wave_dir /= n

        if self._physx_iface and self._body_path_int:
            t   = self._wave_timer
            mul = (math.sin(2.0 * math.pi * t / WAVE_UPDATE_SEC)
                   * (1.0 + 0.12 * math.sin(2.9 * t)))
            fx  = float(self._wave_dir[0] * WAVE_F_ROBOT * mul)
            fy  = float(self._wave_dir[1] * WAVE_F_ROBOT * mul)
            if math.isfinite(fx) and math.isfinite(fy):
                try:
                    self._physx_iface.apply_force_at_pos(
                        self._stage_id, self._body_path_int,
                        carb.Float3(fx, fy, 0.0),
                        carb.Float3(0.0, 0.0, 0.0), "Force")
                except Exception:
                    pass

    # ── 초기화 ────────────────────────────────────────────────────────────
    def setup(self):
        print("=" * 70)
        print("Jackal ROS2 Teleop")
        print("=" * 70)

        # USD 로드
        omni.usd.get_context().open_stage(USD_PATH)
        simulation_app.update()
        stage = omni.usd.get_context().get_stage()

        # /jackal/FixedJoint 는 UR10-Jackal 연결 조인트 → 제거 금지 (DOF 10개 유지)

        # orient 리셋 + 스케일 2×
        jp = stage.GetPrimAtPath("/jackal")
        if jp.IsValid():
            has_scale = False
            for op in UsdGeom.Xformable(jp).GetOrderedXformOps():
                if "orient" in op.GetOpName():
                    op.Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
                elif "translate" in op.GetOpName():
                    c = op.Get()
                    op.Set(Gf.Vec3d(float(c[0]), float(c[1]), 0.15))
                elif "scale" in op.GetOpName():
                    try:
                        c = op.Get()
                        # 현재 스케일(1.5×)의 2배 = 3.0×
                        op.Set(Gf.Vec3d(c[0] * 2.0, c[1] * 2.0, c[2] * 2.0))
                        print(f"  ✓ Jackal 스케일 2× 적용 ({c[0]:.2f}→{c[0]*2:.2f})")
                        has_scale = True
                    except Exception as e:
                        print(f"  ⚠ 스케일 설정 실패: {e}")
            if not has_scale:
                UsdGeom.Xformable(jp).AddScaleOp().Set(Gf.Vec3d(3.0, 3.0, 3.0))
                print("  ✓ Jackal 스케일 3.0× 추가")

        # UR10 스케일 0.33×
        ur10_prim = stage.GetPrimAtPath("/jackal/ur10")
        if ur10_prim.IsValid():
            ur10_xform = UsdGeom.Xformable(ur10_prim)
            has_ur10_scale = False
            for op in ur10_xform.GetOrderedXformOps():
                if "scale" in op.GetOpName():
                    op.Set(Gf.Vec3d(0.33, 0.33, 0.33))
                    print("  ✓ UR10 스케일 0.33× 적용")
                    has_ur10_scale = True
                    break
            if not has_ur10_scale:
                ur10_xform.AddScaleOp().Set(Gf.Vec3d(0.33, 0.33, 0.33))
                print("  ✓ UR10 스케일 0.33× 추가")
        else:
            print("  ⚠ /jackal/ur10 프림을 찾을 수 없음")

        # 물리 설정
        print("\n[Physics Setup]")
        self.setup_physics()
        self._setup_ur10_drives(stage)

        # 환경 생성
        print("\n[Environment]")
        self._create_environment(stage)

        # World
        self.world = World()
        self.world.scene.add_default_ground_plane()
        self.world.add_physics_callback("wave", self.physics_step_callback)

        print("\n[ROS2 Bridge]")
        self.bridge = init_bridge()
        print("  ✓ /cmd_vel 브릿지 시작")

        print("\n✓ Setup complete")

    def _setup_ur10_drives(self, stage):
        """UR10 관절 DriveAPI: 위치 제어 + maxForce 제한 (차량 반력 최소화)"""
        count = 0
        for prim in stage.Traverse():
            path_str = str(prim.GetPath())
            for jname in UR10_JOINT_NAMES:
                if path_str.endswith(jname):
                    drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                    drive.GetStiffnessAttr().Set(500.0)   # 낮춤: 반력 차량 전달 최소화
                    drive.GetDampingAttr().Set(100.0)
                    drive.GetMaxForceAttr().Set(150.0)    # 150 Nm: 차량 뒤집힘 방지
                    drive.GetTargetPositionAttr().Set(0.0)  # 초기 target = 0
                    count += 1
                    break
        print(f"  ✓ UR10 DriveAPI 설정 ({count}/6, stiffness=500, maxForce=150Nm)")

    def _create_environment(self, stage):
        # 이전 잔재 정리
        for p in ["/World/UnderwaterDome", "/World/Seafloor"]:
            prim = stage.GetPrimAtPath(p)
            if prim.IsValid():
                stage.RemovePrim(p)

        dome = UsdLux.DomeLight.Define(stage, "/World/UnderwaterDome")
        dome.CreateColorAttr(Gf.Vec3f(0.04, 0.15, 0.35))
        dome.CreateIntensityAttr(700.0)

        floor = UsdGeom.Cube.Define(stage, "/World/Seafloor")
        floor.CreateSizeAttr(1.0)
        UsdGeom.Xformable(floor).AddScaleOp().Set(Gf.Vec3f(30.0, 30.0, 0.05))
        UsdGeom.Xformable(floor).AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, -0.03))
        floor.CreateDisplayColorAttr([(0.20, 0.17, 0.12)])
        print("  ✓ Underwater environment")

    # ── 보조 추진력 (wheel slip 보완) ────────────────────────────────────
    def _apply_drive_force(self):
        """cmd_vel linear_x 기반 body force - 바퀴 슬립 보완용"""
        if not (self._physx_iface and self._body_path_int and self.bridge):
            return
        linear_x = self.bridge._linear_x
        if linear_x == 0.0:
            return

        # 로봇 현재 방향 벡터 추출
        stage = omni.usd.get_context().get_stage()
        prim  = stage.GetPrimAtPath(BODY_PRIM_PATH)
        if not prim.IsValid():
            return
        xform_cache = UsdGeom.XformCache()
        world_tf    = xform_cache.GetLocalToWorldTransform(prim)
        # Jackal 로컬 X축 = 전진 방향
        fwd = world_tf.TransformDir(Gf.Vec3d(1, 0, 0))
        n   = math.sqrt(fwd[0]**2 + fwd[1]**2)
        if n < 1e-8:
            return
        fx_n, fy_n = fwd[0] / n, fwd[1] / n

        # 힘 = 댐핑 × 속도 × 배율 (반발력 극복 + 추진)
        force = JACKAL_LINEAR_DAMPING * linear_x * 3.0
        try:
            self._physx_iface.apply_force_at_pos(
                self._stage_id, self._body_path_int,
                carb.Float3(float(fx_n * force), float(fy_n * force), 0.0),
                carb.Float3(0.0, 0.0, 0.0), "Force")
        except Exception:
            pass

    # ── PhysX 경로 재캐시 (world.reset() 후 stage_id 변경 대응) ──────────
    def _recache_physx(self):
        stage = omni.usd.get_context().get_stage()
        body_prim = stage.GetPrimAtPath(BODY_PRIM_PATH)
        if not body_prim.IsValid():
            print("  ⚠ body_prim 없음")
            return
        try:
            self._physx_iface   = get_physx_simulation_interface()
            self._stage_id      = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
            self._body_path_int = PhysicsSchemaTools.sdfPathToInt(body_prim.GetPath())
            print(f"  ✓ PhysX 재캐시 (stage_id={self._stage_id})")
        except Exception as e:
            print(f"  ⚠ PhysX recache: {e}")

    # ── 메인 루프 ─────────────────────────────────────────────────────────
    def run(self):
        self.world.reset()
        self._recache_physx()   # reset 후 stage_id 재취득

        # 로봇 Articulation 초기화
        self.robot = Articulation("/jackal")
        self.robot.initialize()
        print(f"✓ Robot ready (DOF: {self.robot.num_dof})")

        # UR10 초기 자세 설정 (이름 기반) + DriveAPI target 동기화
        try:
            dof_names = self.robot.dof_names
            arm_map   = {n: i for i, n in enumerate(dof_names) if n in UR10_JOINT_NAMES}
            init_pos  = self.robot.get_joint_positions().copy()
            for jname, deg in SPAWN_ARM_POSE_DEG.items():
                idx = arm_map.get(jname)
                if idx is not None:
                    init_pos[idx] = np.deg2rad(deg)
            self.robot.set_joint_positions(init_pos)

            # DriveAPI target도 동일 각도로 설정 → 드라이브 반력 없음
            stage = omni.usd.get_context().get_stage()
            for prim in stage.Traverse():
                path_str = str(prim.GetPath())
                for jname in UR10_JOINT_NAMES:
                    if path_str.endswith(jname):
                        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                        if drive:
                            target_rad = np.deg2rad(SPAWN_ARM_POSE_DEG.get(jname, 0.0))
                            drive.GetTargetPositionAttr().Set(float(np.rad2deg(target_rad)))
                        break
            print(f"✓ UR10 초기 자세 + DriveAPI target 동기화: {arm_map}")
        except Exception as e:
            print(f"  ⚠ UR10 자세 설정 실패: {e}")

        # 안정화 대기 (60 스텝)
        for _ in range(60):
            self.world.step(render=True)

        self._physics_ready = True
        print("\n✓ 파도 물리 활성화")
        print("\n[ROS2 Teleop 시작]")
        print("  ros2 topic pub /cmd_vel geometry_msgs/Twist \\")
        print('    "{linear: {x: 0.5}, angular: {z: 0.0}}" --once')
        print("  또는: ros2 run teleop_twist_keyboard teleop_twist_keyboard")
        print("  Ctrl+C 로 종료\n")

        # 메인 루프: ROS2 + 시뮬레이션
        try:
            while simulation_app.is_running():
                spin_once(self.bridge)     # UDP 수신 + 바퀴 적용
                self._apply_drive_force()  # 보조 추진력
                self.world.step(render=True)
        except KeyboardInterrupt:
            pass
        finally:
            print("\n종료 중...")
            shutdown_bridge(self.bridge)


# ═════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo = JackalROS2Teleop()
    demo.setup()
    demo.run()
    simulation_app.close()
