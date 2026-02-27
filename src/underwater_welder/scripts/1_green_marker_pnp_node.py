#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Empty
from cv_bridge import CvBridge


def order_points_clockwise(pts_xy):
    pts = np.array(pts_xy, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def find_green_square_corners_bgr(img_bgr, area_min_px2=800.0):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([35, 80, 50])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask, 0.0

    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < area_min_px2:
        return None, mask, area

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    corners = order_points_clockwise(box)
    return corners, mask, area


# ── 웨이포인트 분할 수 ──────────────────────────────────────────────────────
WAYPOINTS_PER_EDGE = 25   # 한 변을 몇 등분할지 (총 4 × N 점)

# ── 상태 정의 ──────────────────────────────────────────────────────────────
class State(Enum):
    SERVOING = auto()   # 시각 서보: 카메라로 초록 판에 정렬
    WELDING  = auto()   # 용접 시퀀스 실행


class FourCornerPublisher(Node):
    """
    시각 서보로 초록 마커에 카메라를 정렬한 뒤,
    EE 위치 기반으로 판 중심을 계산하고 4코너 용접 시퀀스를 실행.

    상태 전환:
      SERVOING → (정렬 완료) → WELDING → (weld_cycle_done) → SERVOING

    토픽:
      발행: /ee_servo_cmd    (PoseStamped) ← SERVOING 중 연속 EE 목표
      발행: /tool_target_pose (PoseStamped) ← WELDING 중 용접 웨이포인트
      수신: /ee_pose          (PointStamped) ← Isaac Sim에서 EE 월드 좌표
      수신: /corner_ack       (Empty)
      수신: /weld_cycle_done  (Empty)

    시각 서보 원리:
      카메라가 +Y 방향을 향한다고 가정.
      - 마커 중심이 이미지 오른쪽 → EE +X 이동
      - 마커 중심이 이미지 아래쪽 → EE -Z 이동
      - 마커 크기가 기준보다 작다 → EE +Y 이동 (판 방향 전진)
    """

    def __init__(self):
        super().__init__("four_corner_publisher")

        # ── 기존 파라미터 ────────────────────────────────────────────────
        self.declare_parameter("image_topic",             "/camera/image_raw")
        self.declare_parameter("target_pose_topic",       "/tool_target_pose")
        self.declare_parameter("marker_size_m",           0.11)
        # standoff_m: 전극 길이 + 용접 트리거 거리의 합 (test_tool_target_controller의 ELECTRODE_HEIGHT + WELD_TRIGGER_DIST*0.8)
        # 서보 정렬 완료 시 EE_Y + standoff_m = 판 표면 Y (카메라 기반 추정)
        self.declare_parameter("standoff_m",               0.142)
        self.declare_parameter("area_min_px2",            200.0)
        self.declare_parameter("follow_hz",               10.0)
        self.declare_parameter("debug_view",              False)
        self.declare_parameter("fixed_orientation_xyzw",  [0.0, 0.0, 0.0, 1.0])

        # ── 시각 서보 파라미터 ────────────────────────────────────────────
        # servo_ref_size_px: 정렬 기준 크기. 마커가 이 픽셀 크기로 보일 때 거리 OK
        self.declare_parameter("servo_ref_size_px",    700.0/2)
        # 중심/크기 허용 오차
        self.declare_parameter("servo_center_tol_px",   5.0)
        self.declare_parameter("servo_size_tol_px",     5.0)
        # 서보 게인: 1픽셀 오차당 EE 이동 거리 [m]
        # 부호가 반대면 음수로 설정 (카메라 장착 방향에 따라 조절)
        self.declare_parameter("servo_gain_xz",         0.0005)
        self.declare_parameter("servo_gain_y",          0.0005)
        # 카메라 → EE 오프셋 (world Z 기준): 카메라가 EE보다 위에 있으면 양수
        # 서보 정렬 완료 시 마커 Z = EE_Z + cam_z_offset → 코너 Z 계산에 사용
        self.declare_parameter("cam_z_offset",          -0.09)
        # 카메라 → EE 오프셋 (world X 기준): 카메라가 EE보다 오른쪽이면 양수
        # 용접 X 위치가 너무 왼쪽 → 양수, 너무 오른쪽 → 음수
        self.declare_parameter("cam_x_offset",           -0.01)
        # 기존 상태 변수들 아래에 추가
        self._corner_samples: list = []   # 정렬 완료 후 꼭짓점 누적 버퍼
        self._num_samples: int = 10       # 목표 샘플 수

        # ── 파라미터 읽기 ─────────────────────────────────────────────────
        self.image_topic       = self.get_parameter("image_topic").value
        self.target_pose_topic = self.get_parameter("target_pose_topic").value
        self.marker_size_m     = float(self.get_parameter("marker_size_m").value)
        self.standoff_m        = float(self.get_parameter("standoff_m").value)
        self.area_min_px2      = float(self.get_parameter("area_min_px2").value)
        self.follow_hz         = float(self.get_parameter("follow_hz").value)
        self.debug_view        = bool(self.get_parameter("debug_view").value)
        q                      = self.get_parameter("fixed_orientation_xyzw").value
        self.fixed_q           = np.array(q, dtype=np.float64)

        self.servo_ref_size_px    = float(self.get_parameter("servo_ref_size_px").value)
        self.servo_center_tol_px  = float(self.get_parameter("servo_center_tol_px").value)
        self.servo_size_tol_px    = float(self.get_parameter("servo_size_tol_px").value)
        self.servo_gain_xz        = float(self.get_parameter("servo_gain_xz").value)
        self.servo_gain_y         = float(self.get_parameter("servo_gain_y").value)
        self.cam_z_offset         = float(self.get_parameter("cam_z_offset").value)
        self.cam_x_offset         = float(self.get_parameter("cam_x_offset").value)

        # ── 상태 변수 ──────────────────────────────────────────────────────
        self._state:          State             = State.SERVOING
        self._current_ee_pos: np.ndarray | None = None   # /ee_pose 수신값
        self._servo_target:   np.ndarray | None = None   # 서보 발행 목표
        self._plate_front_y:  float             = self.standoff_m  # /plate_pose로 갱신됨
        self._plate_center_z: float             = 0.0              # /plate_pose로 갱신됨
        self.waypoints:       list              = []
        self.corner_idx:      int               = 0
        self.all_done:        bool              = False

        # ── 통신 ──────────────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.create_subscription(
            Image, self.image_topic, self.on_image, qos_profile_sensor_data)
        self.pub        = self.create_publisher(PoseStamped, self.target_pose_topic, 10)
        self._servo_pub = self.create_publisher(PoseStamped, '/ee_servo_cmd', 10)
        self.create_subscription(
            PointStamped, '/ee_pose',    self._on_ee_pose,    10)
        self.create_subscription(
            PointStamped, '/plate_pose', self._on_plate_pose, 10)
        self.create_subscription(Empty, '/corner_ack',      self.on_corner_ack,      10)
        self.create_subscription(Empty, '/weld_cycle_done', self.on_weld_cycle_done, 10)
        self.timer = self.create_timer(1.0 / max(self.follow_hz, 1.0), self.on_timer)

        self.get_logger().info(
            f"FourCornerPublisher 시작 (시각 서보 모드)\n"
            f"  ref_size={self.servo_ref_size_px:.0f}px  "
            f"center_tol={self.servo_center_tol_px:.0f}px  "
            f"size_tol={self.servo_size_tol_px:.0f}px\n"
            f"  gain_xz={self.servo_gain_xz}  gain_y={self.servo_gain_y}")

    # ── EE 위치 수신 ─────────────────────────────────────────────────────────
    def _on_ee_pose(self, msg: PointStamped):
        self._current_ee_pos = np.array(
            [msg.point.x, msg.point.y, msg.point.z], dtype=np.float64)

    # ── 판 위치 수신 → Y(깊이)/Z(높이)는 /plate_pose로 수신 ──────────────────
    def _on_plate_pose(self, msg: PointStamped):
        self._plate_front_y  = msg.point.y   # PLATE_FRONT_Y (용접면 Y)
        self._plate_center_z = msg.point.z   # PLATE_POS[2] (판 중심 Z)


    # ── 웨이포인트 재계산 (검출된 3D 꼭짓점 기반) ─────────────────────────────
    def _rebuild_waypoints_from_corners(self, corners_3d: list):
        """
        카메라에서 검출한 4개 꼭짓점 3D 좌표로 용접 웨이포인트 생성.
        순서: TL → TR → BR → BL → TL (order_points_clockwise 결과와 동일)
        """
        self.waypoints = []
        for i in range(4):
            start = corners_3d[i]
            end   = corners_3d[(i + 1) % 4]
            for j in range(WAYPOINTS_PER_EDGE):
                t = (j + 1) / WAYPOINTS_PER_EDGE
                self.waypoints.append(start + t * (end - start))

        total = len(self.waypoints)
        self.get_logger().info(f"[FCP] 꼭짓점 기반 웨이포인트 생성  총 {total}점")
        for i in range(4):
            edge_names = ["TL→TR", "TR→BR", "BR→BL", "BL→TL"]
            s = self.waypoints[i * WAYPOINTS_PER_EDGE]
            e = self.waypoints[i * WAYPOINTS_PER_EDGE + WAYPOINTS_PER_EDGE - 1]
            self.get_logger().info(
                f"  변{i}({edge_names[i]}): {s.round(3)} → {e.round(3)}")

    # ── 시각 서보 처리 ───────────────────────────────────────────────────────
    def _do_visual_servo(self, img_bgr):
        """
        초록 마커를 기준 사각형에 맞추도록 EE 이동 목표 계산.
        정렬 조건 충족 시 판 중심을 EE 위치에서 계산하고 WELDING 전환.
        """
        corners, _, area = find_green_square_corners_bgr(
            img_bgr, area_min_px2=self.area_min_px2)

        h, w = img_bgr.shape[:2]

        if self.debug_view:
            vis = img_bgr.copy()
            half_ref = int(self.servo_ref_size_px / 2)
            cx_f, cy_f = w // 2, h // 2
            # 파란 기준 사각형
            cv2.rectangle(vis,
                          (cx_f - half_ref, cy_f - half_ref),
                          (cx_f + half_ref, cy_f + half_ref),
                          (255, 0, 0), 2)
            if corners is not None:
                cv2.polylines(vis, [corners.astype(np.int32)], True, (0, 255, 0), 2)
            cv2.putText(vis, f"SERVO  area={area:.0f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            cv2.imshow("servo_view", vis)
            cv2.waitKey(1)

        if corners is None or self._current_ee_pos is None:
            return  # 마커 미검출 or EE 위치 미수신 → 대기

        # ── 마커 중심·크기 계산 ───────────────────────────────────────────
        cx_px   = float(corners[:, 0].mean())
        cy_px   = float(corners[:, 1].mean())
        w_px    = float(corners[:, 0].max() - corners[:, 0].min())
        h_px    = float(corners[:, 1].max() - corners[:, 1].min())
        size_px = math.sqrt(max(w_px * h_px, 1.0))

        x_err    = cx_px - w / 2.0                    # 양수 = 마커가 오른쪽
        z_err    = cy_px - h / 2.0                    # 양수 = 마커가 이미지 아래쪽
        size_err = self.servo_ref_size_px - size_px   # 양수 = 너무 멀다

        self.get_logger().info(
            f"[Servo] x_err={x_err:+.1f}  z_err={z_err:+.1f}  "
            f"size={size_px:.1f}/{self.servo_ref_size_px:.1f}  "
            f"ee={self._current_ee_pos.round(3)}",
            throttle_duration_sec=0.3)

        # ── 정렬 판정 ─────────────────────────────────────────────────────
        if (abs(x_err)    < self.servo_center_tol_px and
                abs(z_err)    < self.servo_center_tol_px and
                abs(size_err) < self.servo_size_tol_px):

            # ── 꼭짓점 1회 계산 후 버퍼에 누적 ──────────────────────────
            scale    = self.marker_size_m / self.servo_ref_size_px
            img_cx   = w / 2.0
            img_cy   = h / 2.0
            wy_surface = self._plate_front_y
            cam_z = cam_z = 1.2 + (self._current_ee_pos[2] + self.cam_z_offset - 1.2) / 0.77 # 받은 좌표 - 0.105
            cam_x = self._current_ee_pos[0] + self.cam_x_offset

            sample = []
            for (px, py) in corners:
                wx = cam_x + (px - img_cx) * scale
                wz = cam_z - (py - img_cy) * scale
                wy = wy_surface
                sample.append(np.array([wx, wy, wz]))

            self._corner_samples.append(sample)

            remaining = self._num_samples - len(self._corner_samples)
            self.get_logger().info(
                f"[Servo] 샘플 수집 중: {len(self._corner_samples)}/{self._num_samples} "
                f"(남은 샘플: {remaining})",
                throttle_duration_sec=0.1)

            # ── 목표 샘플 수 도달 시 평균 계산 → WELDING 전환 ────────────
            if len(self._corner_samples) >= self._num_samples:
                # corners_3d[i] = 4개 꼭짓점 중 i번째의 평균
                corners_3d = [
                    np.mean([s[i] for s in self._corner_samples], axis=0)
                    for i in range(4)
                ]
                self._corner_samples.clear()   # 버퍼 초기화

                self._rebuild_waypoints_from_corners(corners_3d)
                self._state      = State.WELDING
                self.corner_idx  = 0
                self.all_done    = False
                self._servo_target = None
                self.get_logger().info(
                    f"[FCP] ✓ {self._num_samples}회 평균 완료 → WELDING 전환\n"
                    f"  평균 꼭짓점: {[c.round(3).tolist() for c in corners_3d]}")
            return

        # ── 정렬 조건 미충족 시 버퍼 리셋 (흔들림 방지) ──────────────────
        if self._corner_samples:
            self.get_logger().info("[Servo] 정렬 이탈 → 버퍼 초기화")
            self._corner_samples.clear()

        # ── 서보 보정 계산 ─────────────────────────────────────────────────
        # 카메라가 +Y 방향을 바라본다고 가정:
        #   이미지 X 오른쪽(+x_err) → EE +X 이동
        #   이미지 Y 아래쪽(+z_err) → EE -Z 이동 (EE를 내리면 마커가 위로 올라옴)
        #   크기 너무 작음(+size_err) → EE +Y 이동 (판 방향으로 전진)
        # 게인 부호가 맞지 않으면 파라미터로 음수값 입력
        dx = +self.servo_gain_xz * x_err
        dy = +self.servo_gain_y  * size_err
        dz = -self.servo_gain_xz * z_err

        self._servo_target = self._current_ee_pos + np.array([dx, dy, dz])

    # ── 카메라 콜백 ──────────────────────────────────────────────────────────
    def on_image(self, msg: Image):
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return

        if self._state == State.SERVOING:
            self._do_visual_servo(img_bgr)
            return

        # WELDING 상태: debug 표시만
        if self.debug_view:
            corners, mask, _ = find_green_square_corners_bgr(
                img_bgr, area_min_px2=self.area_min_px2)
            vis = img_bgr.copy()
            if corners is not None:
                cv2.polylines(vis, [corners.astype(np.int32)], True, (0, 255, 0), 2)
            label = (f"WELD {self.corner_idx}/{len(self.waypoints)}"
                     f"  done={self.all_done}")
            cv2.putText(vis, label, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("mask", mask)
            cv2.imshow("view", vis)
            cv2.waitKey(1)

    # ── corner_ack 수신 ──────────────────────────────────────────────────────
    def on_corner_ack(self, _msg):
        if self._state != State.WELDING or self.all_done:
            return
        prev = self.corner_idx
        self.corner_idx += 1
        total = len(self.waypoints)
        if self.corner_idx >= total:
            self.all_done = True
            self.get_logger().info(f"[FCP] 전체 {total}점 완료 → weld_cycle_done 대기")
        else:
            edge = self.corner_idx // WAYPOINTS_PER_EDGE
            step = self.corner_idx %  WAYPOINTS_PER_EDGE
            self.get_logger().info(
                f"[FCP] ack: {prev}→{self.corner_idx}  "
                f"(변{edge} 스텝{step+1}/{WAYPOINTS_PER_EDGE})")

    # ── weld_cycle_done 수신 ─────────────────────────────────────────────────
    def on_weld_cycle_done(self, _msg):
        self._state        = State.SERVOING
        self.all_done      = False
        self._servo_target = None
        self.get_logger().info("[FCP] weld_cycle_done → SERVOING 재시작")

    # ── 타이머: 현재 목표 반복 발행 ─────────────────────────────────────────
    def on_timer(self):
        if self._state == State.SERVOING:
            if self._servo_target is not None:
                self._publish_pose(self._servo_pub, self._servo_target)
            return

        # WELDING
        if self.all_done or self.corner_idx >= len(self.waypoints):
            return
        self._publish_pose(self.pub, self.waypoints[self.corner_idx])

    # ── PoseStamped 발행 헬퍼 ────────────────────────────────────────────────
    def _publish_pose(self, publisher, target: np.ndarray):
        ps = PoseStamped()
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.header.frame_id = "World"
        ps.pose.position.x = float(target[0])
        ps.pose.position.y = float(target[1])
        ps.pose.position.z = float(target[2])
        ps.pose.orientation.x = float(self.fixed_q[0])
        ps.pose.orientation.y = float(self.fixed_q[1])
        ps.pose.orientation.z = float(self.fixed_q[2])
        ps.pose.orientation.w = float(self.fixed_q[3])
        publisher.publish(ps)


def main():
    rclpy.init()
    node = FourCornerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.debug_view:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
