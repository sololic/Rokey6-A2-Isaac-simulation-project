#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════════
JACKAL + UR10  수중 데모  (underwater_env.py 모듈 사용)
══════════════════════════════════════════════════════════════════
jackal_ur10_demo.py 의 모듈화 버전
- 수중 환경: underwater_env.UnderwaterEnvironment
- 로봇 제어: 이 파일 (팔 제어 + 균열 추종)

실행:
  /home/rokey/isaacsim/python.sh jackal_ur10_underwater_demo.py

파이프라인 (터미널 분리):
  python3 camera/crack_detection_node.py
  python3 camera/crack_to_socket.py
  python3 camera/test_image_publisher.py
══════════════════════════════════════════════════════════════════
"""

import asyncio
import socket

from isaacsim import SimulationApp

CONFIG = {
    "headless": False,
    "width": 1920,
    "height": 1080,
    "extra_args": ["--enable", "omni.isaac.ros2_bridge"]
}
simulation_app = SimulationApp(CONFIG)

import omni.usd
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from pxr import UsdPhysics, UsdGeom, Gf
import numpy as np

# 수중 환경 모듈
from underwater_env import UnderwaterEnvironment

# ═══════════════════════════════════════════════════════════════
#  설정 상수
# ═══════════════════════════════════════════════════════════════
USD_PATH = "/home/rokey/Downloads/jackal_and_ur10.usd"

UR10_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

HOME_POSE_DEG = {
    "shoulder_pan_joint":   0.0,
    "shoulder_lift_joint": -90.0,
    "elbow_joint":          0.0,
    "wrist_1_joint":        0.0,
    "wrist_2_joint":        0.0,
    "wrist_3_joint":        0.0,
}

REACH_POSE_DEG = {
    "shoulder_pan_joint":   90.0,
    "shoulder_lift_joint":  -90.0,
    "elbow_joint":         90.0,
    "wrist_1_joint":       0.0,
    "wrist_2_joint":        90.0,
    "wrist_3_joint":         0.0,
}

# 구역별 팔 자세 (reach 베이스 + pan ±30°)
ZONE_POSES = [
    {"shoulder_pan_joint": -30.0, "shoulder_lift_joint": -45.0,   # Zone 0: 좌상
     "elbow_joint": -60.0, "wrist_1_joint": -75.0, "wrist_2_joint": 90.0, "wrist_3_joint": 0.0},
    {"shoulder_pan_joint":  30.0, "shoulder_lift_joint": -45.0,   # Zone 1: 우상
     "elbow_joint": -60.0, "wrist_1_joint": -75.0, "wrist_2_joint": 90.0, "wrist_3_joint": 0.0},
    {"shoulder_pan_joint":  30.0, "shoulder_lift_joint": -45.0,   # Zone 2: 우하
     "elbow_joint": -60.0, "wrist_1_joint": -75.0, "wrist_2_joint": 90.0, "wrist_3_joint": 0.0},
    {"shoulder_pan_joint": -30.0, "shoulder_lift_joint": -45.0,   # Zone 3: 좌하
     "elbow_joint": -60.0, "wrist_1_joint": -75.0, "wrist_2_joint": 90.0, "wrist_3_joint": 0.0},
]

# MARKER_XY     = [(-0.8, 2.0), (0.8, 2.0), (0.8, 3.0), (-0.8, 3.0)]
MARKER_XY     = [(-0.2, 1.0), (-0.2, 1.0),(-0.2, 1.0),(-0.2, 1.0)]
ZONE_BOUNDARY = 0.08
ZONE_CONFIRM  = 3
ZONE_TIMEOUT  = 3.0

WHEEL_JOINT_PATHS = [
    "/jackal/front_left_wheel_joint",
    "/jackal/front_right_wheel_joint",
    "/jackal/rear_left_wheel_joint",
    "/jackal/rear_right_wheel_joint",
]


class JackalUR10UnderwaterDemo:
    """Jackal + UR10 수중 데모 (underwater_env 모듈 사용)"""

    def __init__(self):
        self.world           = None
        self.robot           = None
        self._env            = UnderwaterEnvironment(body_prim_path="/jackal/base_link")
        self._arm_joint_map  = {}
        self._crack_sock     = None
        self._crack_x        = 0.0
        self._crack_y        = 0.0
        self._crack_detected = False

    # ─────────────────────────────────────────────────────────
    #  셋업
    # ─────────────────────────────────────────────────────────
    def setup(self):
        print("=" * 60)
        print("JACKAL + UR10  Underwater Demo  (모듈 버전)")
        print("=" * 60)

        omni.usd.get_context().open_stage(USD_PATH)
        stage = omni.usd.get_context().get_stage()

        # Jackal 스폰 높이 초기화
        jackal_prim = stage.GetPrimAtPath("/jackal")
        if jackal_prim.IsValid():
            for op in UsdGeom.Xformable(jackal_prim).GetOrderedXformOps():
                if "translate" in op.GetOpName():
                    cur = op.Get()
                    op.Set(Gf.Vec3d(float(cur[0]), float(cur[1]), 0.1))
                elif "orient" in op.GetOpName():
                    op.Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

        # ── 수중 환경 생성 (모듈) ─────────────────────────────
        self._env.create(stage)

        # ── UR10 DriveAPI (위치 제어) ──────────────────────────
        count = 0
        for prim in stage.Traverse():
            for jname in UR10_JOINT_NAMES:
                if str(prim.GetPath()).endswith(jname):
                    d = UsdPhysics.DriveAPI.Apply(prim, "angular")
                    d.GetStiffnessAttr().Set(1000.0)
                    d.GetDampingAttr().Set(200.0)
                    d.GetMaxForceAttr().Set(300.0)
                    count += 1
                    break
        print(f"  ✓ UR10 DriveAPI ({count}/6 joints)")

        # ── 용접 토치 ─────────────────────────────────────────
        self._attach_torch(stage)

        # ── 균열 마커 + 경로 시각화 ───────────────────────────
        self._create_crack_markers(stage)
        self._create_weld_path(stage)

        # ── World + 물리 콜백 ─────────────────────────────────
        self.world = World()
        self.world.scene.add_default_ground_plane()
        self.world.add_physics_callback("wave_callback",
                                        callback_fn=self._env.physics_step_callback)

        # ── Articulation + UDP ────────────────────────────────
        self.robot = Articulation(prim_path="/jackal")
        self._setup_crack_receiver()

        print("\n✓ Setup complete (underwater mode)")

    def _attach_torch(self, stage):
        ee = stage.GetPrimAtPath("/jackal/ur10/ee_link")
        if not ee.IsValid():
            return
        tp = "/jackal/ur10/ee_link/WeldingTorch"
        if stage.GetPrimAtPath(tp).IsValid():
            stage.RemovePrim(tp)
        cyl = UsdGeom.Cylinder.Define(stage, tp)
        cyl.CreateRadiusAttr(0.025)
        cyl.CreateHeightAttr(0.15)
        cyl.CreateAxisAttr("X")
        UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3f(0.075, 0.0, 0.0))
        cyl.CreateDisplayColorAttr([(0.95, 0.45, 0.10)])
        print("  ✓ 용접 토치 장착")

    def _create_crack_markers(self, stage):
        S, T, CZ = 0.2, 0.025, 1.24
        color = Gf.Vec3f(0.0, 1.0, 0.0)
        # color = Gf.Vec3f(1.0, 0.15, 0.15)
        black_color = Gf.Vec3f(0.0, 0.0, 0.0)
        margin = 0.3

        # bars = [(0, S/2, S, T), (0, -S/2, S, T), (-S/2, 0, T, S), (S/2, 0, T, S)]
        # for i, (cx, cy) in enumerate(MARKER_XY):
        #     for j, (dx, dz, sx, sz) in enumerate(bars):
        #         path = f'/World/CrackMarker_{i}_bar{j}'
        #         box = UsdGeom.Cube.Define(stage, path)
        #         box.GetSizeAttr().Set(1.0)
        #         xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        #         xf.AddTranslateOp().Set(Gf.Vec3d(cx+dx, cy, CZ+dz))
        #         xf.AddScaleOp().Set(Gf.Vec3d(sx, T, sz))
        #         box.GetDisplayColorAttr().Set([color])

        for i, (cx, cy) in enumerate(MARKER_XY):
            # margin = 0.03
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
            xf.AddScaleOp().Set(Gf.Vec3d(S, T, S))   # x=S, y=얇음, z=S
            box.GetDisplayColorAttr().Set([color])
    
        print("  ✓ 균열 마커 4개 (빨간 프레임)")

    def _create_weld_path(self, stage):
        CZ, T = 0.25, 0.018
        seg_colors = [
            Gf.Vec3f(1.0, 0.85, 0.0), Gf.Vec3f(1.0, 0.85, 0.0),
            Gf.Vec3f(1.0, 0.85, 0.0), Gf.Vec3f(0.55, 0.55, 0.55),
        ]
        for i in range(4):
            x1, y1 = MARKER_XY[i]
            x2, y2 = MARKER_XY[(i+1) % 4]
            p = f'/World/WeldPath/seg_{i}'
            box = UsdGeom.Cube.Define(stage, p)
            box.GetSizeAttr().Set(1.0)
            xf = UsdGeom.Xformable(stage.GetPrimAtPath(p))
            xf.AddTranslateOp().Set(Gf.Vec3d((x1+x2)/2, (y1+y2)/2, CZ))
            xf.AddScaleOp().Set(Gf.Vec3d(max(abs(x2-x1), T), max(abs(y2-y1), T), T))
            box.GetDisplayColorAttr().Set([seg_colors[i]])

        order_colors = [
            Gf.Vec3f(0.2, 1.0, 0.2), Gf.Vec3f(1.0, 1.0, 0.0),
            Gf.Vec3f(1.0, 0.5, 0.0), Gf.Vec3f(1.0, 0.2, 0.2),
        ]
        for i, (mx, my) in enumerate(MARKER_XY):
            p = f'/World/WeldPath/num_{i}'
            sph = UsdGeom.Sphere.Define(stage, p)
            sph.GetRadiusAttr().Set(0.055)
            UsdGeom.Xformable(stage.GetPrimAtPath(p)).AddTranslateOp().Set(
                Gf.Vec3d(mx, my, CZ+0.28))
            sph.GetDisplayColorAttr().Set([order_colors[i]])

        ind = UsdGeom.Sphere.Define(stage, '/World/WeldPath/indicator')
        ind.GetRadiusAttr().Set(0.08)
        UsdGeom.Xformable(stage.GetPrimAtPath('/World/WeldPath/indicator')).AddTranslateOp().Set(
            Gf.Vec3d(MARKER_XY[0][0], MARKER_XY[0][1], CZ))
        ind.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 1.0, 1.0)])
        print("  ✓ 용접 경로 시각화")

    def _setup_crack_receiver(self):
        self._crack_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._crack_sock.bind(('127.0.0.1', 9998))
        self._crack_sock.setblocking(False)
        print("  ✓ Crack UDP:9998 준비")

    # ─────────────────────────────────────────────────────────
    #  균열 추종
    # ─────────────────────────────────────────────────────────
    def _recv_crack(self):
        if self._crack_sock is None:
            return
        try:
            data, _ = self._crack_sock.recvfrom(64)
            msg = data.decode().strip()
            if msg == 'none':
                self._crack_detected = False
            else:
                x, y = map(float, msg.split(','))
                self._crack_x = x
                self._crack_y = y
                self._crack_detected = True
        except BlockingIOError:
            pass

    def _classify_zone(self, nx, ny):
        if abs(nx) < ZONE_BOUNDARY or abs(ny) < ZONE_BOUNDARY:
            return -1
        if nx < 0 and ny > 0: return 0
        if nx > 0 and ny > 0: return 1
        if nx > 0 and ny < 0: return 2
        return 3

    def _move_weld_indicator(self, zone):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath('/World/WeldPath/indicator')
        if not prim.IsValid():
            return
        pos = (Gf.Vec3f(0.0, 0.0, -2.0) if zone < 0
               else Gf.Vec3f(MARKER_XY[zone][0], MARKER_XY[zone][1], 0.25))
        attr = prim.GetAttribute("xformOp:translate")
        if attr and attr.IsValid():
            attr.Set(pos)

    async def _move_arm(self, pose_deg: dict, duration=3.0):
        current = self.robot.get_joint_positions().copy()
        target  = current.copy()
        for jname, deg in pose_deg.items():
            idx = self._arm_joint_map.get(jname)
            if idx is not None:
                target[idx] = np.deg2rad(deg)
        steps = max(1, int(duration / 0.05))
        for s in range(steps):
            t = (s+1) / steps
            alpha = t * t * (3.0 - 2.0 * t)
            self.robot.apply_action(
                ArticulationAction(joint_positions=current + alpha*(target-current)))
            await asyncio.sleep(0.05)

    async def crack_follow_loop(self):
        print("\n[Crack Follow] 구역 기반 UR10 추종 시작 (수중 모드)")
        dt = 0.05
        last_zone = -1; candidate = -1; confirm_count = 0; no_detect_t = 0.0
        while True:
            self._recv_crack()
            if self._crack_detected:
                no_detect_t = 0.0
                zone = self._classify_zone(self._crack_x, self._crack_y)
                if zone == -1:
                    zone = last_zone
                if zone == candidate:
                    confirm_count += 1
                else:
                    candidate = zone; confirm_count = 1
                if confirm_count >= ZONE_CONFIRM and zone != last_zone and zone >= 0:
                    last_zone = zone
                    print(f"  → Zone {zone} 확정: CrackMarker_{zone}")
                    self._move_weld_indicator(zone)
                    await self._move_arm(ZONE_POSES[zone])
            else:
                no_detect_t += dt
                if no_detect_t >= ZONE_TIMEOUT and last_zone >= 0:
                    print("  ⚠ HOME 복귀")
                    self._move_weld_indicator(-1)
                    await self._move_arm(HOME_POSE_DEG)
                    last_zone = -1; candidate = -1; confirm_count = 0; no_detect_t = 0.0
            await asyncio.sleep(dt)

    # ─────────────────────────────────────────────────────────
    #  메인 루프
    # ─────────────────────────────────────────────────────────
    async def run_async(self):
        await self.world.reset_async()
        self.robot.initialize()

        dof_names = self.robot.dof_names
        self._arm_joint_map = {n: i for i, n in enumerate(dof_names)
                                if n in UR10_JOINT_NAMES}
        print(f"✓ DOF:{self.robot.num_dof}, UR10:{self._arm_joint_map}")

        # 스폰 자세 (팔 수평)
        init_pos = self.robot.get_joint_positions().copy()
        for jname in UR10_JOINT_NAMES:
            idx = self._arm_joint_map.get(jname)
            if idx is not None:
                init_pos[idx] = 0.0
        self.robot.set_joint_positions(init_pos)

        # 착지 대기
        await asyncio.sleep(1.5)

        # 수중 물리 활성화 + 애니메이션 시작
        self._env.activate()
        asyncio.ensure_future(self._env.animate_bubbles())
        asyncio.ensure_future(self._env.animate_debris())
        print("✓ 수중 물리 + 버블/잔해 애니메이션 시작")

        # Phase 1: HOME
        print("\n[Phase 1] HOME 포지션")
        await self._move_arm(HOME_POSE_DEG, 5.0)

        # Phase 2: Reach → 균열 추종
        print("\n[Phase 2] Reach 자세 → 균열 추종 시작")
        await self._move_arm(REACH_POSE_DEG, 4.0)
        print("✓ 균열 추종 루프 시작 (Ctrl+C 로 종료)")
        await self.crack_follow_loop()

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
        print("\nStopping...")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
