#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════════════════════
UNDERWATER WELDER - Jackal + UR10 데모
══════════════════════════════════════════════════════════════════════════════

[개발 히스토리]
  Step 1. standalone_demo.py  - Franka 팔 + 수중 기본 물리 (중력 스케일링)
  Step 2. standalone_demo.py  - 파도력(PhysX per-step force API) + debris/bubble 추가
           → 백업: standalone_demo_wave_force_backup.py
  Step 3. ur10_demo.py        - ur_new_car.usd 기반 UR10 + 토치(실린더) 장착
           → 바퀴 DriveAPI 미설정 문제: joint_efforts 토크 방식으로 우회했으나 불안정
           → OmniGraph 간섭으로 바퀴 제어 신뢰성 낮음
  Step 4. (현재) jackal_ur10_demo.py
           → /home/rokey/Downloads/jackal_and_ur10.usd 사용
           → Jackal 차량 (DriveAPI 사전 설정 완료): targetVelocity 직접 설정
           → UR10 arm (6DOF) + 용접 토치(실린더) 코드 장착
           → 수중 물리: gravity 스케일링 + damping + wave force 유지

[USD 구조 요약 - jackal_and_ur10.usd]
  /jackal                             (defaultPrim, Articulation root)
  /jackal/base_link                   (메인 바디)
  /jackal/front_left_wheel_joint      (DriveAPI.angular: stiffness=0, damping=10M, vel=0)
  /jackal/front_right_wheel_joint
  /jackal/rear_left_wheel_joint
  /jackal/rear_right_wheel_joint
  /jackal/ur10                        (UR10 서브트리, 0.5× 스케일)
  /jackal/ur10/joints/shoulder_pan_joint
  /jackal/ur10/joints/shoulder_lift_joint
  /jackal/ur10/joints/elbow_joint
  /jackal/ur10/joints/wrist_1_joint
  /jackal/ur10/joints/wrist_2_joint
  /jackal/ur10/joints/wrist_3_joint
  /jackal/ur10/ee_link                (end-effector)
  /jackal/base_link/FixedJoint        (Jackal↔UR10 고정 연결)
  ※ Jackal 1.5×, UR10 0.5× 스케일 적용 (균형 유지)

[핵심 개선 사항 vs ur10_demo.py]
  - 바퀴: DriveAPI 사전 설정 → targetVelocity 직접 설정 (안정적)
  - 팔: DriveAPI Apply + stiffness/damping (위치 제어 모드)
  - OmniGraph 간섭 없음 (Jackal USD가 clean)
══════════════════════════════════════════════════════════════════════════════
"""

import sys
import asyncio
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
from omni.isaac.core.utils.types import ArticulationAction
from omni.physx import get_physx_simulation_interface
from pxr import (
    UsdPhysics, PhysxSchema, Gf, UsdGeom, UsdLux,
    UsdUtils, PhysicsSchemaTools, Sdf, UsdShade
)
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
#  설정 상수
# ═══════════════════════════════════════════════════════════════════════════
USD_PATH = "/home/rokey/Downloads/jackal_and_ur10.usd"

# 수중 물리
WATER_DENSITY   = 1025.0   # 해수 밀도 (kg/m³)
GRAVITY_CONST   = 9.81
WAVE_F_ROBOT    = 150.0    # 로봇 최대 수평 파도력 (N)
WAVE_UPDATE_SEC = 6.0

# ── Jackal 물리 파라미터 (실제 치수 기반) ────────────────────────────────
# Jackal 실제 치수: 508×430×270 mm (휠 포함 약간 더 큼)
# UR10 (0.5x 스케일) 포함한 전체 체적으로 부력 계산
JACKAL_WIDTH    = 0.43   # m
JACKAL_LENGTH   = 0.55   # m (UR10 베이스 포함)
JACKAL_HEIGHT   = 0.32   # m (UR10 마운트 포함)
JACKAL_MASS_KG  = 200.0  # kg: 부력 계산으로 eff_g ≈ 2 m/s² (수중 착저형)
# 부력용 체적: 실제 외형보다 크게 (내부 공기 공간 포함, 실제 수중 로봇과 동일)
# eff_g = g × (1 - ρ_water×V_buoy / mass) ≈ 2 m/s² 목표
# → V_buoy = mass × (1 - eff_g/g) / ρ_water = 200×(1-2/9.81)/1025 ≈ 0.155 m³
JACKAL_VOLUME   = 0.155  # m³ (부력용: 내부 공기 포함, eff_g ≈ 2 m/s²)

# ── 유체역학 항력 계수 계산 ───────────────────────────────────────────────
# F_drag = 0.5 × C_d × ρ × A_frontal × v²  (실제 이차 항력)
# PhysX linear_damping = 0.5 × C_d × ρ × A_frontal  (v=1 m/s 기준 선형 근사)
_C_d            = 1.2    # 박스형 수중 로봇 항력계수 (1.0~1.5)
_A_frontal      = JACKAL_WIDTH * JACKAL_HEIGHT  # 전면 투영 면적 (m²)
JACKAL_LINEAR_DAMPING  = 0.5 * _C_d * WATER_DENSITY * _A_frontal  # ≈ 100 Ns/m
JACKAL_ANGULAR_DAMPING = 30.0   # 회전 저항 (전복 방지용 별도 설정)

# ── 바퀴 속도 계산 ──────────────────────────────────────────────────────
# Jackal 바퀴 반경: 0.098 m (실제값)
# 목표 선속도: eff_g=2 m/s² 로 법선력이 줄어 슬립 발생 → 여유 5x 적용
# ω = v/r = 1.0/0.098 ≈ 10.2 rad/s, 5x 여유 → 50 rad/s
_WHEEL_RADIUS   = 0.098                          # m (Jackal 실제 바퀴 반경)
_TARGET_LINEAR  = 2.0                            # m/s (목표 선속도)
WHEEL_VELOCITY  = _TARGET_LINEAR / _WHEEL_RADIUS * 5.0  # ≈ 51 rad/s (수중 슬립 여유 5x)

# Jackal 바퀴 경로 (DriveAPI 사전 설정 확인됨)
WHEEL_JOINT_PATHS = [
    "/jackal/front_left_wheel_joint",
    "/jackal/front_right_wheel_joint",
    "/jackal/rear_left_wheel_joint",
    "/jackal/rear_right_wheel_joint",
]
# Jackal 좌/우 바퀴 구분 (차동구동 - 반대 부호로 전진)
# 왼쪽: +velocity, 오른쪽: -velocity → 전방 직진
LEFT_WHEELS  = {"front_left", "rear_left"}
RIGHT_WHEELS = {"front_right", "rear_right"}

# Jackal body (force 적용 대상)
BODY_PRIM_PATH = "/jackal/base_link"

# UR10 end-effector
EE_PRIM_PATH = "/jackal/ur10/ee_link"

# UR10 joint 경로 접두사 (스캔용)
UR10_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# ── UR10 초기 자세 (joint 이름 → 도 단위, 슬라이스 방식 금지) ────────────
# 실제 확인된 DOF 순서: [0]lift [1]elbow [2]pan [3]w1 [4]w2 [5~8]바퀴 [9]w3
# shoulder_lift = -90 → 팔이 위를 향함 (테스트로 확인됨)
SPAWN_ARM_POSE_DEG = {
    "shoulder_pan_joint":   0.0,
    "shoulder_lift_joint":  0.0,    # 낙하 중 전복 방지: 팔 수평(0°) → 무게중심 낮춤
    "elbow_joint":          0.0,    # 착지 후 HOME_POSE_DEG로 텔레포트
    "wrist_1_joint":        0.0,
    "wrist_2_joint":        0.0,
    "wrist_3_joint":        0.0,
}
HOME_POSE_DEG = {
    "shoulder_pan_joint":   0.0,
    "shoulder_lift_joint": -90.0,   # 위로 세움 (착지 후 안전하게 텔레포트)
    "elbow_joint":          0.0,
    "wrist_1_joint":        0.0,
    "wrist_2_joint":        0.0,
    "wrist_3_joint":        0.0,
}


class JackalUR10UnderwaterDemo:
    """Jackal + UR10 수중 용접 로봇 데모"""

    def __init__(self):
        self.world  = None
        self.robot  = None

        # physx per-step force 캐시
        self._physx_iface   = None
        self._stage_id      = None
        self._body_path_int = None

        # 물리 콜백 활성화
        self._physics_ready = False

        # 파도 파라미터
        self._wave_dir        = np.array([1.0, 0.0])
        self._wave_dir_target = self._wave_dir.copy()
        self._wave_timer      = 0.0
        self._wave_dir_sign   = 1.0

        # 버블 / 잔해 데이터
        self.bubbles = []
        self.debris  = []

        # 토치 경로
        self._torch_path = None

    # ─────────────────────────────────────────────────────────────────────
    #  유틸: prim 트리 출력
    # ─────────────────────────────────────────────────────────────────────
    def print_prim_tree(self, max_depth=6):
        print("\n── USD Prim Tree ─────────────────────────────────────────")
        stage = omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            path  = str(prim.GetPath())
            depth = len(path.split("/")) - 2
            if depth > max_depth:
                continue
            indent = "  " * depth
            print(f"{indent}{path}  [{prim.GetTypeName()}]")
        print("──────────────────────────────────────────────────────────\n")

    # ─────────────────────────────────────────────────────────────────────
    #  용접 토치 장착 (실린더, ee_link 하위)
    # ─────────────────────────────────────────────────────────────────────
    def attach_welding_torch(self):
        print(f"\nAttaching welding torch to {EE_PRIM_PATH}...")
        stage     = omni.usd.get_context().get_stage()
        ee_prim   = stage.GetPrimAtPath(EE_PRIM_PATH)

        if not ee_prim.IsValid():
            print(f"  ✗ EE prim not found: {EE_PRIM_PATH}")
            # 스캔 fallback
            for prim in stage.Traverse():
                p = str(prim.GetPath())
                if "ee_link" in p.lower() or "tool0" in p.lower():
                    print(f"  → fallback EE: {p}")
                    EE_PRIM_PATH_actual = p
                    break
            else:
                print("  ✗ EE 완전히 못 찾음")
                return
        else:
            EE_PRIM_PATH_actual = EE_PRIM_PATH

        torch_path = EE_PRIM_PATH_actual + "/WeldingTorch"

        # 이전 실행 잔재 정리
        existing = stage.GetPrimAtPath(torch_path)
        if existing.IsValid():
            stage.RemovePrim(torch_path)

        # 실린더 토치 본체 (X축 방향)
        cylinder = UsdGeom.Cylinder.Define(stage, torch_path)
        cylinder.CreateRadiusAttr(0.025)   # 2.5 cm
        cylinder.CreateHeightAttr(0.15)    # 15 cm
        cylinder.CreateAxisAttr("X")       # X축 방향으로 뻗음
        xform = UsdGeom.Xformable(cylinder)
        xform.AddTranslateOp().Set(Gf.Vec3f(0.075, 0.0, 0.0))
        cylinder.CreateDisplayColorAttr([(0.95, 0.45, 0.10)])

        # 토치 팁 (아크 점)
        tip_path = torch_path + "/Tip"
        tip = UsdGeom.Sphere.Define(stage, tip_path)
        tip.CreateRadiusAttr(0.015)
        UsdGeom.Xformable(tip).AddTranslateOp().Set(Gf.Vec3f(0.15, 0.0, 0.0))
        tip.CreateDisplayColorAttr([(1.0, 0.95, 0.2)])

        self._torch_path = torch_path
        print(f"  ✓ 토치 생성: {torch_path}")
        print(f"    radius=2.5cm, height=15cm, axis=X, color=orange")
        print(f"  ✓ 아크 팁: {tip_path}")

    # ─────────────────────────────────────────────────────────────────────
    #  수중 환경
    # ─────────────────────────────────────────────────────────────────────
    def create_underwater_environment(self):
        print("\nCreating underwater environment...")
        stage = omni.usd.get_context().get_stage()

        # 이전 실행 잔재 정리 (XformOp 중복 추가 크래시 방지)
        for path in ["/World/UnderwaterDome", "/World/Seafloor",
                     "/World/WeldTarget", "/World/Bubbles", "/World/Debris"]:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                stage.RemovePrim(path)
        self.bubbles.clear()

        # 파란 돔 라이트
        dome = UsdLux.DomeLight.Define(stage, "/World/UnderwaterDome")
        dome.CreateColorAttr(Gf.Vec3f(0.04, 0.15, 0.35))
        dome.CreateIntensityAttr(700.0)
        print("  ✓ Underwater dome light (deep blue)")

        # 해저면 (시각용 - collision은 default_ground_plane이 담당)
        floor = UsdGeom.Cube.Define(stage, "/World/Seafloor")
        floor.CreateSizeAttr(1.0)
        UsdGeom.Xformable(floor).AddScaleOp().Set(Gf.Vec3f(30.0, 30.0, 0.05))
        UsdGeom.Xformable(floor).AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, -0.03))
        floor.CreateDisplayColorAttr([(0.20, 0.17, 0.12)])
        # CollisionAPI 없음 - default_ground_plane(z=0)이 실제 물리 바닥
        print("  ✓ Seafloor visual (no collision, ground plane handles physics)")

        # 용접 목표물 (차량 진행 방향을 막지 않도록 측면에 배치)
        pipe = UsdGeom.Cylinder.Define(stage, "/World/WeldTarget")
        pipe.CreateRadiusAttr(0.08)
        pipe.CreateHeightAttr(1.0)
        pipe.CreateAxisAttr("Z")
        UsdGeom.Xformable(pipe).AddTranslateOp().Set(Gf.Vec3f(3.0, 2.0, 0.3))  # 측면 배치
        pipe.CreateDisplayColorAttr([(0.55, 0.55, 0.60)])
        pipe_prim = stage.GetPrimAtPath("/World/WeldTarget")
        UsdPhysics.CollisionAPI.Apply(pipe_prim)
        print("  ✓ Weld target pipe created (side position, 3.0, 2.0)")

        # 버블 (시각 효과)
        random.seed(42)
        for i in range(25):
            path = f"/World/Bubbles/bubble_{i:02d}"
            sphere = UsdGeom.Sphere.Define(stage, path)
            r = random.uniform(0.015, 0.06)
            sphere.CreateRadiusAttr(r)
            x = random.uniform(-5.0, 5.0)
            y = random.uniform(-5.0, 5.0)
            z = random.uniform(-0.5, 5.0)
            UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3f(x, y, z))
            sphere.CreateDisplayColorAttr([(0.65, 0.82, 1.0)])
            self.bubbles.append({
                "path": path, "x": x, "y": y, "z": z,
                "speed": random.uniform(0.25, 0.8), "r": r,
            })
        print(f"  ✓ {len(self.bubbles)} bubbles created")

    def create_debris_cubes(self):
        """수중 잔해 (부유/중립/가라앉음)"""
        print("\nCreating debris cubes...")
        stage = omni.usd.get_context().get_stage()
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
                self.debris.append({
                    "path": path, "net_buoy": net_buoy,
                    "x": x, "y": y, "z": z,
                    "vx": 0.0, "vy": 0.0, "vz": 0.0,
                })
        print(f"  ✓ {len(self.debris)} debris cubes created")

    # ─────────────────────────────────────────────────────────────────────
    #  물리 씬 설정
    # ─────────────────────────────────────────────────────────────────────
    def setup_physics_scene(self):
        """부력 보상: effective_gravity = g × (1 - buoyancy/weight)"""
        print("\nSetting up physics scene...")
        buoyancy  = WATER_DENSITY * JACKAL_VOLUME * GRAVITY_CONST
        weight    = JACKAL_MASS_KG * GRAVITY_CONST
        eff_g     = max(0.5, GRAVITY_CONST * (1.0 - buoyancy / weight))

        stage      = omni.usd.get_context().get_stage()
        # 실제 physicsScene 경로 탐색 (USD 로드 후 /Environment/physicsScene)
        for candidate in ["/Environment/physicsScene", "/physicsScene", "/World/physicsScene"]:
            scene_prim = stage.GetPrimAtPath(candidate)
            if scene_prim.IsValid():
                scene_path = candidate
                break
        else:
            scene_prim = None

        try:
            if scene_prim is not None and scene_prim.IsValid():
                scene = UsdPhysics.Scene(scene_prim)
                scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
                scene.CreateGravityMagnitudeAttr().Set(eff_g)
            else:
                scene_path = "/physicsScene"
                scene = UsdPhysics.Scene.Define(stage, scene_path)
                scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
                scene.CreateGravityMagnitudeAttr().Set(eff_g)
                scene_prim = stage.GetPrimAtPath(scene_path)
                physx_sc = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
                physx_sc.CreateEnableCCDAttr(True)
                physx_sc.CreateEnableStabilizationAttr(True)
                physx_sc.CreateEnableGPUDynamicsAttr(False)
                physx_sc.CreateBroadphaseTypeAttr("MBP")
                physx_sc.CreateSolverTypeAttr("TGS")
            print(f"  ✓ Effective gravity: {eff_g:.2f} m/s²"
                  f"  (scene={scene_path}, buoyancy={buoyancy:.0f}N, weight={weight:.0f}N)")
        except Exception as e:
            print(f"  ⚠ Physics scene error: {e}")

        # 바닥 마찰 재질 설정 (eff_g가 낮아 법선력 감소 → 마찰 보완)
        try:
            mat_path = "/World/GroundFrictionMat"
            mat_prim = stage.GetPrimAtPath(mat_path)
            if not mat_prim.IsValid():
                mat_prim = UsdShade.Material.Define(stage, mat_path).GetPrim()
            phys_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
            phys_mat.CreateStaticFrictionAttr().Set(1.5)   # 높은 마찰 (eff_g 보완)
            phys_mat.CreateDynamicFrictionAttr().Set(1.2)
            phys_mat.CreateRestitutionAttr().Set(0.0)

            # 기본 바닥면에 마찰 재질 바인딩 (prim tree에서 확인된 실제 경로)
            gnd_prim = None
            for gnd_candidate in [
                "/Environment/GroundPlane/CollisionPlane",
                "/Environment/GroundPlane/CollisionMesh",
                "/Environment/GroundPlane",
                "/World/defaultGroundPlane/GroundPlane/CollisionPlane",
                "/World/defaultGroundPlane",
            ]:
                p = stage.GetPrimAtPath(gnd_candidate)
                if p.IsValid():
                    gnd_prim = p
                    break
            if gnd_prim is not None:
                bind_api = UsdShade.MaterialBindingAPI.Apply(gnd_prim)
                bind_api.Bind(UsdShade.Material(mat_prim),
                              UsdShade.Tokens.weakerThanDescendants,
                              "physics")
                print(f"  ✓ Ground friction: static=1.5, dynamic=1.2 ({gnd_prim.GetPath()})")
            else:
                print("  ⚠ Ground prim not found (friction not applied)")
        except Exception as e:
            print(f"  ⚠ Friction material error: {e}")

    def setup_underwater_physics(self):
        """base_link에 damping 적용 + physx force 인터페이스 캐시"""
        print("\nSetting up underwater physics for Jackal body...")
        stage     = omni.usd.get_context().get_stage()
        body_prim = stage.GetPrimAtPath(BODY_PRIM_PATH)

        if not body_prim.IsValid():
            print(f"  ✗ {BODY_PRIM_PATH} not found")
            return

        # 수중 damping (PhysxRigidBodyAPI)
        # linear=60: 200kg 바디가 수중에서 천천히 가라앉도록 (4.0은 너무 약해서 자유낙하 수준)
        # angular=30: 파도에 의한 회전 진동 억제
        try:
            physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(body_prim)
            physx_rb.CreateLinearDampingAttr().Set(JACKAL_LINEAR_DAMPING)
            physx_rb.CreateAngularDampingAttr().Set(JACKAL_ANGULAR_DAMPING + 70.0)  # +70: 팔 반력 흡수
            print(f"  ✓ RigidBody damping: linear={JACKAL_LINEAR_DAMPING:.1f} Ns/m"
                  f"  (C_d={_C_d}, A={_A_frontal:.3f}m²), angular={JACKAL_ANGULAR_DAMPING+70:.0f}")
        except Exception as e:
            print(f"  ⚠ Damping error: {e}")

        # per-step force 인터페이스 캐시 (파도력용)
        try:
            self._physx_iface   = get_physx_simulation_interface()
            self._stage_id      = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
            self._body_path_int = PhysicsSchemaTools.sdfPathToInt(body_prim.GetPath())
            print("  ✓ PhysX force interface cached")
        except Exception as e:
            print(f"  ⚠ PhysX interface cache error: {e}")

    def set_jackal_mass(self):
        """Jackal body 질량 설정 + 안정성 강화"""
        stage     = omni.usd.get_context().get_stage()
        body_prim = stage.GetPrimAtPath(BODY_PRIM_PATH)
        if not body_prim.IsValid():
            print(f"  ⚠ Mass 설정 실패: {BODY_PRIM_PATH} 없음")
            return
        try:
            mass_api = UsdPhysics.MassAPI.Apply(body_prim)
            mass_api.GetMassAttr().Set(JACKAL_MASS_KG)
            # CoM을 최대한 낮게 → 강한 복원 모멘트 (ballast 효과)
            mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, -JACKAL_HEIGHT * 0.3))
            print(f"  ✓ Jackal mass: {JACKAL_MASS_KG} kg, CoM z={-JACKAL_HEIGHT*0.3:.3f} m (저중심 30%)")
        except Exception as e:
            print(f"  ⚠ Mass error: {e}")

    def setup_wheel_friction(self):
        """바퀴 마찰 재질 강화 + customGeometry 제거 (구식 속성으로 접촉 오류 유발)"""
        print("\nSetting up wheel friction...")
        stage = omni.usd.get_context().get_stage()

        # USD 기존 바퀴 재질에 마찰값 덮어쓰기
        wheel_mat_prim = stage.GetPrimAtPath("/jackal/PhysicsMaterials/wheels")
        if wheel_mat_prim.IsValid():
            phys_mat = UsdPhysics.MaterialAPI.Apply(wheel_mat_prim)
            phys_mat.CreateStaticFrictionAttr().Set(2.0)
            phys_mat.CreateDynamicFrictionAttr().Set(1.5)
            phys_mat.CreateRestitutionAttr().Set(0.0)
            print("  ✓ Wheel material override: static=2.0, dynamic=1.5")

        # 각 바퀴 collision에 customGeometry 제거 + 마찰 재질 명시 바인딩
        wheel_col_paths = [
            "/jackal/front_left_wheel_link/collisions",
            "/jackal/front_right_wheel_link/collisions",
            "/jackal/rear_left_wheel_link/collisions",
            "/jackal/rear_right_wheel_link/collisions",
        ]
        fixed = 0
        for wpath in wheel_col_paths:
            wprim = stage.GetPrimAtPath(wpath)
            if not wprim.IsValid():
                continue
            # obsolete customGeometry 속성 제거
            for attr_name in ["customGeometry", "physxCollision:customGeometry"]:
                if wprim.HasAttribute(attr_name):
                    wprim.RemoveProperty(attr_name)
            # 마찰 재질 명시 바인딩
            if wheel_mat_prim.IsValid():
                bind_api = UsdShade.MaterialBindingAPI.Apply(wprim)
                bind_api.Bind(UsdShade.Material(wheel_mat_prim),
                              UsdShade.Tokens.weakerThanDescendants, "physics")
            fixed += 1
        print(f"  ✓ Wheel friction applied to {fixed} collisions (customGeometry removed)")

    def setup_ur10_joint_drives(self):
        """UR10 관절 DriveAPI 설정 (위치 제어 모드)
        ★ maxForce: 실제 UR10 최대 토크 ~330 Nm 기준으로 설정
           500000 Nm은 차량을 뒤집을 만한 반력 발생 → 300으로 제한
        """
        print("\nSetting up UR10 joint drives...")
        stage = omni.usd.get_context().get_stage()
        count = 0
        for prim in stage.Traverse():
            path_str = str(prim.GetPath())
            for jname in UR10_JOINT_NAMES:
                if path_str.endswith(jname):
                    drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                    drive.GetStiffnessAttr().Set(1000.0)    # 10000 → 1000 (10x 감소)
                    drive.GetDampingAttr().Set(200.0)       # 1000 → 200
                    drive.GetMaxForceAttr().Set(300.0)      # 500000 → 300 Nm (실제 UR10 기준)
                    count += 1
                    break
        print(f"  ✓ UR10 DriveAPI applied ({count}/6 joints, stiffness=1000, maxForce=300Nm)")

    # ─────────────────────────────────────────────────────────────────────
    #  Jackal 바퀴 제어 (DriveAPI targetVelocity)
    # ─────────────────────────────────────────────────────────────────────
    def set_wheel_velocity(self, forward: float, turn: float = 0.0):
        """Jackal 차동구동 속도 설정
        forward > 0 = 전진, turn > 0 = 좌회전
        좌/우 바퀴 모두 +velocity = 전진 (축 방향 동일, 테스트로 확인됨)
        """
        # 보조 추진력 계산: 목표속도×damping (바퀴 슬립 시 body force로 보완)
        # target_v = _TARGET_LINEAR × (forward/WHEEL_VELOCITY) → _TARGET_LINEAR로 속도 제어
        target_v = (forward / WHEEL_VELOCITY) * _TARGET_LINEAR if WHEEL_VELOCITY > 0 else 0.0
        self._propulsion_fx = JACKAL_LINEAR_DAMPING * target_v * 2.5   # +X 방향 추진력
        self._propulsion_fy = JACKAL_LINEAR_DAMPING * (turn / WHEEL_VELOCITY) * _TARGET_LINEAR * 0.5 if WHEEL_VELOCITY > 0 else 0.0

        stage = omni.usd.get_context().get_stage()
        for jpath in WHEEL_JOINT_PATHS:
            prim = stage.GetPrimAtPath(jpath)
            if not prim.IsValid():
                print(f"  ⚠ wheel joint not found: {jpath}")
                continue

            # 좌/우 모두 같은 부호 (테스트로 확인: 둘 다 + = 전진)
            if any(side in jpath for side in LEFT_WHEELS):
                vel = forward + turn
            else:  # right wheels
                vel = forward - turn

            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if drive:
                drive.GetTargetVelocityAttr().Set(vel)
            else:
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.GetStiffnessAttr().Set(0.0)
                drive.GetDampingAttr().Set(10000000.0)
                drive.GetMaxForceAttr().Set(10000000.0)
                drive.GetTargetVelocityAttr().Set(vel)

    def stop_wheels(self):
        self._propulsion_fx = 0.0
        self._propulsion_fy = 0.0
        self.set_wheel_velocity(0.0, 0.0)

    # ─────────────────────────────────────────────────────────────────────
    #  물리 콜백 (파도력)
    # ─────────────────────────────────────────────────────────────────────
    def physics_step_callback(self, step_size):
        if not self._physics_ready:
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

        wave_dir_final = self._wave_dir_target * self._wave_dir_sign
        self._wave_dir += (wave_dir_final - self._wave_dir) * 0.015
        norm = np.linalg.norm(self._wave_dir)
        if norm > 1e-8:
            self._wave_dir /= norm

        # 파도력 + 보조 추진력 → Jackal base_link
        if self._physx_iface is not None and self._body_path_int is not None:
            try:
                t       = self._wave_timer
                wave_mul = (math.sin(2.0 * math.pi * t / WAVE_UPDATE_SEC)
                            * (1.0 + 0.12 * math.sin(2.9 * t)))
                fx = float(self._wave_dir[0] * WAVE_F_ROBOT * wave_mul)
                fy = float(self._wave_dir[1] * WAVE_F_ROBOT * wave_mul)
                # 보조 추진력 합산 (바퀴 슬립 보완: 항력의 2.5배로 가속 확보)
                fx += getattr(self, "_propulsion_fx", 0.0)
                fy += getattr(self, "_propulsion_fy", 0.0)
                if math.isfinite(fx) and math.isfinite(fy):
                    self._physx_iface.apply_force_at_pos(
                        self._stage_id, self._body_path_int,
                        carb.Float3(fx, fy, 0.0),
                        carb.Float3(0.0, 0.0, 0.0), "Force"
                    )
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────
    #  애니메이션 (버블 / 잔해)
    # ─────────────────────────────────────────────────────────────────────
    async def animate_bubbles(self):
        stage = omni.usd.get_context().get_stage()
        dt    = 0.033
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
        stage = omni.usd.get_context().get_stage()
        dt    = 0.033
        DRAG  = 1.8
        while True:
            t        = self._wave_timer
            wave_mul = math.sin(2.0 * math.pi * t / WAVE_UPDATE_SEC)
            wave_mul *= 1.0 + 0.12 * math.sin(2.9 * t)
            WAVE_A   = 0.15
            cur_x    = self._wave_dir[0] * WAVE_A * wave_mul
            cur_y    = self._wave_dir[1] * WAVE_A * wave_mul
            for d in self.debris:
                d["vz"] += (d["net_buoy"] - DRAG * d["vz"]) * dt
                d["vx"] += (cur_x - DRAG * 0.2 * d["vx"]) * dt
                d["vy"] += (cur_y - DRAG * 0.2 * d["vy"]) * dt
                d["z"]  += d["vz"] * dt
                d["x"]  += d["vx"] * dt
                d["y"]  += d["vy"] * dt
                if d["z"] > 7.0:
                    d["z"] = 0.3; d["vz"] = 0.0
                elif d["z"] < -0.5:
                    d["z"] = -0.5; d["vz"] = 0.0
                d["x"] = max(-6.0, min(d["x"], 6.0))
                d["y"] = max(-6.0, min(d["y"], 6.0))
                prim = stage.GetPrimAtPath(d["path"])
                if prim.IsValid():
                    attr = prim.GetAttribute("xformOp:translate")
                    if attr:
                        attr.Set(Gf.Vec3f(d["x"], d["y"], d["z"]))
            await asyncio.sleep(dt)

    # ─────────────────────────────────────────────────────────────────────
    #  셋업
    # ─────────────────────────────────────────────────────────────────────
    def setup(self):
        print("=" * 70)
        print("UNDERWATER WELDER  -  Jackal + UR10 Demo")
        print("=" * 70)

        print(f"\nLoading USD: {USD_PATH}")
        omni.usd.get_context().open_stage(USD_PATH)

        # 깨진 FixedJoint 제거 (Franka panda_link0 참조 → 존재하지 않는 prim → 에러 발생)
        try:
            stage = omni.usd.get_context().get_stage()
            broken_joint = stage.GetPrimAtPath("/jackal/FixedJoint")
            if broken_joint.IsValid():
                stage.RemovePrim("/jackal/FixedJoint")
                print("  ✓ /jackal/FixedJoint 제거 (Franka 잔재, 불필요한 에러 원인)")
        except Exception as e:
            print(f"  ⚠ FixedJoint 제거 실패: {e}")

        # Jackal 스폰 위치/방향 초기화: z=0.3으로 올리고 회전은 identity로 리셋
        try:
            stage = omni.usd.get_context().get_stage()
            jackal_prim = stage.GetPrimAtPath("/jackal")
            if jackal_prim.IsValid():
                xformable = UsdGeom.Xformable(jackal_prim)
                ops = xformable.GetOrderedXformOps()
                translate_set = False
                for op in ops:
                    op_name = op.GetOpName()
                    if "translate" in op_name:
                        cur = op.Get()
                        op.Set(Gf.Vec3d(float(cur[0]), float(cur[1]), 0.1))
                        print(f"  ✓ Jackal 스폰 z: {float(cur[2]):.3f} → 0.05 (낙하 최소화)")
                        translate_set = True
                    elif "orient" in op_name:
                        # 쿼터니언 → identity (1, 0, 0, 0) = 직립
                        op.Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
                        print(f"  ✓ Jackal orient 리셋 → identity (직립)")
                    elif "rotate" in op_name:
                        # rotateXYZ / rotateX 등 → 0
                        try:
                            op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
                        except Exception:
                            try:
                                op.Set(0.0)
                            except Exception:
                                pass
                        print(f"  ✓ Jackal rotate 리셋 → 0: {op_name}")
                if not translate_set:
                    xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.05))
                    print("  ✓ Jackal translate 추가: z=0.3")
        except Exception as e:
            print(f"  ⚠ Jackal 위치/방향 설정 실패: {e}")

        # prim 트리 출력 (구조 확인)
        self.print_prim_tree()

        # 환경 생성
        self.create_underwater_environment()
        self.create_debris_cubes()

        # 물리 설정
        self.setup_physics_scene()
        self.setup_underwater_physics()
        self.set_jackal_mass()
        self.setup_wheel_friction()

        # UR10 DriveAPI 설정
        self.setup_ur10_joint_drives()

        # 토치 장착
        self.attach_welding_torch()

        # World 생성 + 확실한 physics 바닥 + callback
        print("\nCreating World...")
        self.world = World()
        # default_ground_plane: z=0에 안정적인 physics 바닥 생성 (수동 cube보다 신뢰성 높음)
        self.world.scene.add_default_ground_plane()
        print("  ✓ Default ground plane added (z=0)")
        self.world.add_physics_callback("wave_callback",
                                        callback_fn=self.physics_step_callback)

        # Articulation 등록
        print("Getting Jackal+UR10 articulation...")
        self.robot = Articulation(prim_path="/jackal")

        print("\n✓ Setup complete")

    # ─────────────────────────────────────────────────────────────────────
    #  메인 루프
    # ─────────────────────────────────────────────────────────────────────
    async def run_async(self):
        print("\nResetting world...")
        await self.world.reset_async()
        # 3초 대기 제거: 대기 중 팔이 앞으로 뻗은 채 로봇이 쏠리므로 즉시 초기화

        print("Initializing robot articulation...")
        self.robot.initialize()
        n_dof = self.robot.num_dof
        print(f"✓ Robot ready (DOF: {n_dof})")

        # DOF 이름 출력
        try:
            dof_names = self.robot.dof_names
            print("  DOF 목록:")
            for i, name in enumerate(dof_names):
                print(f"    [{i:2d}] {name}")
        except Exception:
            pass

        # ── joint 이름 → DOF 인덱스 매핑 빌드 ─────────────────────────
        dof_names = self.robot.dof_names
        arm_joint_map = {
            name: i for i, name in enumerate(dof_names)
            if name in UR10_JOINT_NAMES
        }
        print(f"  UR10 joint map: {arm_joint_map}")

        # ── UR10 팔 즉시 스폰 자세로 텔레포트 (이름 기반, 슬라이스 금지) ──
        try:
            init_pos = self.robot.get_joint_positions().copy()
            for jname, deg in SPAWN_ARM_POSE_DEG.items():
                idx = arm_joint_map.get(jname)
                if idx is not None:
                    init_pos[idx] = np.deg2rad(deg)
            self.robot.set_joint_positions(init_pos)
            print(f"✓ UR10 스폰 자세 설정 (이름 기반)")
        except Exception as e:
            print(f"  ⚠ UR10 초기 포지션 설정 실패: {e}")

        # 착지 안정화 대기 (z=0.05, g=eff 높음 → 거의 즉시 착지, 1.5s면 충분)
        await asyncio.sleep(1.5)
        print("✓ 착지 안정화 완료")

        self._physics_ready = True
        print("✓ Wave physics ACTIVE")

        # 버블 / 잔해 애니메이션 시작
        asyncio.ensure_future(self.animate_bubbles())
        asyncio.ensure_future(self.animate_debris())
        print("✓ Animation started")

        # ── 데모 시퀀스 ──────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("DEMO SEQUENCE - Jackal 주행 + UR10 수중 용접")
        print("=" * 70)

        # UR10 포즈 정의 (joint 이름 → 도 단위)
        # shoulder_lift: -90 = 위, 0 = 수평
        ur10_poses = {
            "home": {
                "shoulder_pan_joint":   0.0,
                "shoulder_lift_joint": -90.0,   # 위로 세움
                "elbow_joint":          0.0,
                "wrist_1_joint":        0.0,
                "wrist_2_joint":        0.0,
                "wrist_3_joint":        0.0,
            },
            "reach": {
                "shoulder_pan_joint":   0.0,
                "shoulder_lift_joint": -45.0,   # 앞으로 45°
                "elbow_joint":         -60.0,
                "wrist_1_joint":       -75.0,
                "wrist_2_joint":        90.0,
                "wrist_3_joint":        0.0,
            },
            "weld_pos": {
                "shoulder_pan_joint":  35.0,    # 오른쪽 스윙
                "shoulder_lift_joint": -50.0,
                "elbow_joint":         -70.0,
                "wrist_1_joint":       -60.0,
                "wrist_2_joint":        90.0,
                "wrist_3_joint":        0.0,
            },
            "weld_neg": {
                "shoulder_pan_joint": -35.0,    # 왼쪽 스윙
                "shoulder_lift_joint": -50.0,
                "elbow_joint":         -70.0,
                "wrist_1_joint":       -60.0,
                "wrist_2_joint":        90.0,
                "wrist_3_joint":        0.0,
            },
            "retract": {
                "shoulder_pan_joint":   0.0,
                "shoulder_lift_joint": -70.0,
                "elbow_joint":          50.0,
                "wrist_1_joint":       -80.0,
                "wrist_2_joint":        0.0,
                "wrist_3_joint":        0.0,
            },
        }

        def make_full_target(pose_dict: dict) -> np.ndarray:
            """pose dict → 전체 DOF 배열 (바퀴 등 나머지는 현재값 유지)"""
            arr = self.robot.get_joint_positions().copy()
            for jname, deg in pose_dict.items():
                idx = arm_joint_map.get(jname)
                if idx is not None:
                    arr[idx] = np.deg2rad(deg)
            return arr

        async def move_arm(pose_name: str, duration: float = 5.0):
            """팔을 지정 포즈로 보간 이동 (이름 기반, 바퀴 DOF 건드리지 않음)"""
            current     = self.robot.get_joint_positions().copy()
            full_target = make_full_target(ur10_poses[pose_name])

            steps = max(30, int(duration / 0.05))
            dt    = duration / steps
            for s in range(steps):
                t      = (s + 1) / steps
                alpha  = t * t * (3.0 - 2.0 * t)   # smoothstep
                interp = current + alpha * (full_target - current)
                self.robot.apply_action(ArticulationAction(joint_positions=interp))
                await asyncio.sleep(dt)
            print(f"    → {pose_name} 완료")

        # ── Phase 1: 홈 포지션 이동 (착지 후 팔을 위로 천천히 올림) ─────
        # stiffness=1000, maxForce=300 Nm → 반력 작음, angular_damping=100 → 안정
        print("\n[Phase 1] 홈 포지션으로 이동 (shoulder_lift 0°→-90°, 6초)")
        await move_arm("home", 6.0)

        # ── Phase 2: Jackal 전진 + 팔 뻗기 ─────────────────────────
        print("\n[Phase 2] Jackal 전진 + 팔 도달")
        self.set_wheel_velocity(WHEEL_VELOCITY)
        print(f"  ✓ 바퀴 속도: {WHEEL_VELOCITY} rad/s (좌+{WHEEL_VELOCITY}, 우+{WHEEL_VELOCITY})")
        await move_arm("reach", 5.0)
        await asyncio.sleep(3.0)

        # ── Phase 3: 용접 작업 (좌우 스윙) ─────────────────────────
        print("\n[Phase 3] 수중 용접 시작 (좌우 반복)")
        self.stop_wheels()
        for cycle in range(3):
            print(f"  [용접 사이클 {cycle+1}/3]")
            await move_arm("weld_pos", 3.0)
            await asyncio.sleep(0.5)
            await move_arm("weld_neg", 3.0)
            await asyncio.sleep(0.5)

        # ── Phase 4: 팔 복귀 + 후진 ────────────────────────────────
        print("\n[Phase 4] 팔 복귀 + 후진")
        await move_arm("retract", 4.0)
        self.set_wheel_velocity(-WHEEL_VELOCITY * 0.7)  # 후진
        await asyncio.sleep(3.0)
        self.stop_wheels()

        # ── Phase 5: 홈 복귀 ────────────────────────────────────────
        print("\n[Phase 5] 홈 포지션 복귀")
        await move_arm("home", 4.0)
        await asyncio.sleep(2.0)

        print("\n" + "=" * 70)
        print("DEMO COMPLETE!")
        print("  - Jackal 주행 완료")
        print("  - UR10 수중 용접 시퀀스 완료")
        print("  - 파도력 / 버블 / 잔해 시뮬레이션 유지 중")
        print("=" * 70)

    def run(self):
        self.setup()
        asyncio.ensure_future(self.run_async())
        print("\nSimulation running... (Ctrl+C to stop)\n")
        while simulation_app.is_running():
            self.world.step(render=True)


def main():
    demo = JackalUR10UnderwaterDemo()
    try:
        demo.run()
    except KeyboardInterrupt:
        print("\nStopping simulation...")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
