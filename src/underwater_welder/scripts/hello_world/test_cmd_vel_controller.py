#!/usr/bin/env python3
"""
test_cmd_vel_controller.py
/cmd_vel → Jackal 바퀴 DriveAPI 제어

[두 가지 사용법]
  1) standalone 테스트 (Isaac Sim Python으로 직접 실행):
       ~/.local/share/ov/pkg/isaac_sim-*/python.sh test_cmd_vel_controller.py

  2) water_env_cmd_vel.py 에서 import 해서 사용:
       from test_cmd_vel_controller import JackalCmdVelController

[테스트 명령 (다른 터미널)]
  # 전진
  ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
      '{linear: {x: 1.0}, angular: {z: 0.0}}' -r 10

  # 좌회전
  ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
      '{linear: {x: 0.5}, angular: {z: 1.0}}' -r 10

  # 정지
  ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
      '{linear: {x: 0.0}, angular: {z: 0.0}}' --once

[차동구동 공식]
  v_left  = (linear_x - angular_z * TRACK/2) / RADIUS * SLIP
  v_right = (linear_x + angular_z * TRACK/2) / RADIUS * SLIP
  angular_z > 0  →  좌회전 (왼쪽 바퀴 느림)
  angular_z < 0  →  우회전 (오른쪽 바퀴 느림)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import omni.usd
from pxr import UsdPhysics

# ─── Jackal 바퀴 파라미터 ───────────────────────────────────────────────────
WHEEL_RADIUS  = 0.098   # m (Jackal 실제 바퀴 반경)
WHEEL_TRACK   = 0.430   # m (좌우 바퀴 중심 간격)
SLIP_FACTOR   = 5.0     # 수중 슬립 보상 계수 (법선력 감소 → 여유 5x)
MAX_WHEEL_VEL = 100.0   # rad/s 클램핑 상한
MAX_LINEAR    = 2.0     # m/s  cmd_vel 안전 클램핑
MAX_ANGULAR   = 2.0     # rad/s cmd_vel 안전 클램핑

# ─── 바퀴 joint USD 경로 ───────────────────────────────────────────────────
WHEEL_JOINT_PATHS = [
    "/jackal/front_left_wheel_joint",
    "/jackal/front_right_wheel_joint",
    "/jackal/rear_left_wheel_joint",
    "/jackal/rear_right_wheel_joint",
]
LEFT_WHEELS  = {"front_left", "rear_left"}
RIGHT_WHEELS = {"front_right", "rear_right"}


# ─── ROS2 구독 노드 (내부용) ─────────────────────────────────────────────────
class _CmdVelNode(Node):
    """
    /cmd_vel (Twist) 구독자.
    JackalCmdVelController 가 내부적으로 사용 — 직접 사용하지 않음.
    """

    def __init__(self):
        super().__init__('jackal_cmd_vel_driver')
        self._linear_x  = 0.0
        self._angular_z = 0.0
        self.create_subscription(Twist, '/cmd_vel', self._cb, 10)
        self.get_logger().info('/cmd_vel 구독 시작')

    def _cb(self, msg: Twist):
        self._linear_x  = float(msg.linear.x)
        self._angular_z = float(msg.angular.z)
        self.get_logger().info(
            f'cmd_vel 수신: linear_x={self._linear_x:+.2f} m/s  '
            f'angular_z={self._angular_z:+.2f} rad/s'
        )

    def get_cmd_vel(self) -> tuple[float, float]:
        """(linear_x [m/s], angular_z [rad/s]) 반환"""
        return self._linear_x, self._angular_z


# ─── 메인 제어기 ─────────────────────────────────────────────────────────────
class JackalCmdVelController:
    """
    /cmd_vel → Jackal DriveAPI 바퀴 제어기

    초기화:
        rclpy.init() 이후에 생성해야 함.
        ctrl = JackalCmdVelController()

    매 시뮬레이션 스텝:
        ctrl.update()    ← _buoyancy_step() 안에 넣기

    종료:
        ctrl.destroy()
    """

    def __init__(self):
        self._node = _CmdVelNode()
        print("[CmdVel] ✓ JackalCmdVelController 초기화 완료")
        print(f"[CmdVel]   WHEEL_RADIUS={WHEEL_RADIUS}m  "
              f"WHEEL_TRACK={WHEEL_TRACK}m  "
              f"SLIP_FACTOR={SLIP_FACTOR}x")

    # ── 메인 업데이트 ────────────────────────────────────────────────────────
    def update(self):
        """
        매 시뮬레이션 스텝 호출.
        ROS2 메시지 처리 → 차동구동 변환 → DriveAPI 적용.
        """
        # 1. ROS2 메시지 처리 (논블로킹: 대기 중인 메시지만 처리)
        rclpy.spin_once(self._node, timeout_sec=0)

        # 2. cmd_vel 읽기 + 안전 클램핑
        v, w = self._node.get_cmd_vel()
        v = max(-MAX_LINEAR,  min(MAX_LINEAR,  v))
        w = max(-MAX_ANGULAR, min(MAX_ANGULAR, w))

        # 3. 차동구동 변환: 선속도 → 바퀴 각속도
        #    v_left  = (v - w * TRACK/2) / r * SLIP
        #    v_right = (v + w * TRACK/2) / r * SLIP
        #    w > 0 → 좌회전 (left < right)
        v_left  = (v - w * WHEEL_TRACK / 2.0) / WHEEL_RADIUS * SLIP_FACTOR
        v_right = (v + w * WHEEL_TRACK / 2.0) / WHEEL_RADIUS * SLIP_FACTOR

        # 안전 클램핑
        v_left  = max(-MAX_WHEEL_VEL, min(MAX_WHEEL_VEL, v_left))
        v_right = max(-MAX_WHEEL_VEL, min(MAX_WHEEL_VEL, v_right))

        # 4. DriveAPI 적용
        self._apply_wheels(v_left, v_right)

    def stop(self):
        """바퀴 긴급 정지"""
        self._apply_wheels(0.0, 0.0)

    def destroy(self):
        """정리 (프로그램 종료 시 호출)"""
        self.stop()
        self._node.destroy_node()

    # ── DriveAPI 적용 ────────────────────────────────────────────────────────
    def _apply_wheels(self, left: float, right: float):
        """
        USD DriveAPI를 통해 바퀴 각속도 설정.

        jackal_and_ur10.usd 는 DriveAPI가 사전 설정되어 있으므로
        Get() 으로 가져와서 targetVelocity만 변경.
        없는 경우(fallback) Apply()로 생성.
        """
        stage = omni.usd.get_context().get_stage()

        for jpath in WHEEL_JOINT_PATHS:
            prim = stage.GetPrimAtPath(jpath)
            if not prim.IsValid():
                continue

            # 좌/우 판별
            vel = left if any(s in jpath for s in LEFT_WHEELS) else right

            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if drive:
                drive.GetTargetVelocityAttr().Set(vel)
            else:
                # DriveAPI 미설정 USD용 안전 처리 (fallback)
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.GetStiffnessAttr().Set(0.0)
                drive.GetDampingAttr().Set(10_000_000.0)
                drive.GetMaxForceAttr().Set(10_000_000.0)
                drive.GetTargetVelocityAttr().Set(vel)


# ─── standalone 실행 (water_env 없이 단독 테스트) ────────────────────────────
def main():
    """
    Isaac Sim 단독 실행 모드.
    BaseSample 없이 간단히 cmd_vel → 바퀴 제어를 확인.

    실행:
      ~/.local/share/ov/pkg/isaac_sim-*/python.sh test_cmd_vel_controller.py
    """
    # ── Isaac Sim 먼저 초기화 ──────────────────────────────────────────────
    from isaacsim import SimulationApp
    sim_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

    import omni.usd as _omni_usd
    from omni.isaac.core import World

    USD_PATH = "/home/rokey/isaacsim/extension_examples/hello_world/jackal_and_ur10.usd"

    print("=" * 60)
    print("  cmd_vel Controller Standalone Test")
    print("=" * 60)

    # ── USD 로드 ──────────────────────────────────────────────────────────
    print(f"[Main] USD 로드: {USD_PATH}")
    _omni_usd.get_context().open_stage(USD_PATH)

    world = World()
    world.scene.add_default_ground_plane()
    world.reset()

    # ── ROS2 + 제어기 초기화 ──────────────────────────────────────────────
    rclpy.init()
    ctrl = JackalCmdVelController()

    print("\n[Main] 시뮬레이션 시작. 다른 터미널에서:")
    print("  전진:   ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "
          "'{linear: {x: 1.0}}' -r 10")
    print("  좌회전: ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "
          "'{linear: {x: 0.5}, angular: {z: 1.0}}' -r 10")
    print("  정지:   ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "
          "'{linear: {x: 0.0}}' --once")
    print("Ctrl+C 종료\n")

    # ── 메인 루프 ─────────────────────────────────────────────────────────
    try:
        while sim_app.is_running():
            world.step(render=True)
            ctrl.update()
    except KeyboardInterrupt:
        print("\n[Main] 종료 중...")
    finally:
        ctrl.destroy()
        rclpy.shutdown()
        sim_app.close()


if __name__ == "__main__":
    main()
