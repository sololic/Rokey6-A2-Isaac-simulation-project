#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
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


# ── 웨이포인트 분할 수 (이 값을 바꾸면 한 변당 용접 점 수 조절) ──────────────────
WAYPOINTS_PER_EDGE = 80   # ← 변경 가능: 한 변을 몇 등분할지
# 총 웨이포인트 = WAYPOINTS_PER_EDGE × 4 = 40개
# ────────────────────────────────────────────────────────────────────────────


class FourCornerPublisher(Node):
    """
    초록 마커를 검출한 뒤 XZ 평면의 각 변을 WAYPOINTS_PER_EDGE 등분하여
    총 4 × WAYPOINTS_PER_EDGE 개 웨이포인트를 순서대로 발행한다.

    /corner_ack      (Empty 수신) → 다음 웨이포인트로 진행
    /weld_cycle_done (Empty 수신) → 초기화 후 마커 재검출 대기

    경로: TL→TR→BR→BL→TL (XZ 평면, PLATE_FRONT_Y 기준)
    """

    def __init__(self):
        super().__init__("four_corner_publisher")

        self.declare_parameter("image_topic",           "/camera/image_raw")
        self.declare_parameter("target_pose_topic",     "/tool_target_pose")
        self.declare_parameter("marker_center_world",   [-0.2, 1.0, 1.24])
        self.declare_parameter("marker_size_m",         0.2)
        self.declare_parameter("plate_front_y",         0.9875)
        self.declare_parameter("area_min_px2",          800.0)
        self.declare_parameter("follow_hz",             10.0)
        self.declare_parameter("debug_view",            False)
        self.declare_parameter("fixed_orientation_xyzw", [0.0, 0.0, 0.0, 1.0])

        self.image_topic       = self.get_parameter("image_topic").value
        self.target_pose_topic = self.get_parameter("target_pose_topic").value
        mc                     = self.get_parameter("marker_center_world").value
        self.marker_center     = np.array(mc, dtype=np.float64)
        self.marker_size_m     = float(self.get_parameter("marker_size_m").value)
        self.plate_front_y     = float(self.get_parameter("plate_front_y").value)
        self.area_min_px2      = float(self.get_parameter("area_min_px2").value)
        self.follow_hz         = float(self.get_parameter("follow_hz").value)
        self.debug_view        = bool(self.get_parameter("debug_view").value)
        q                      = self.get_parameter("fixed_orientation_xyzw").value
        self.fixed_q           = np.array(q, dtype=np.float64)

        # ── XZ 평면 4코너 → 각 변을 WAYPOINTS_PER_EDGE 등분 ───────────────
        half = self.marker_size_m / 2.0
        cx   = float(self.marker_center[0])
        cz   = float(self.marker_center[2])
        y    = self.plate_front_y
        corners = [
            np.array([cx - half, y, cz + half]),  # TL
            np.array([cx + half, y, cz + half]),  # TR
            np.array([cx + half, y, cz - half]),  # BR
            np.array([cx - half, y, cz - half]),  # BL
        ]
        # 각 변: 시작점 제외, 끝점 포함으로 WAYPOINTS_PER_EDGE개 생성
        # t = 1/N, 2/N, ..., N/N  (N = WAYPOINTS_PER_EDGE)
        self.waypoints = []
        for i in range(4):
            start = corners[i]
            end   = corners[(i + 1) % 4]
            for j in range(WAYPOINTS_PER_EDGE):
                t = (j + 1) / WAYPOINTS_PER_EDGE
                self.waypoints.append(start + t * (end - start))
        # self.waypoints[0..N-1]   = 변 0 (TL→TR)
        # self.waypoints[N..2N-1]  = 변 1 (TR→BR)
        # self.waypoints[2N..3N-1] = 변 2 (BR→BL)
        # self.waypoints[3N..4N-1] = 변 3 (BL→TL)

        # ── 상태 ────────────────────────────────────────────────────────────
        self.marker_detected: bool = False
        self.corner_idx:      int  = 0      # 현재 웨이포인트 인덱스 (0 ~ 4N-1)
        self.all_done:        bool = False  # 전체 완료 → weld_cycle_done 대기 중
        self._needs_reset:    bool = True   # True: 다음 감지 시 idx 리셋 / False: 재개

        # ── 통신 ────────────────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.create_subscription(
            Image, self.image_topic, self.on_image, qos_profile_sensor_data)
        self.pub = self.create_publisher(PoseStamped, self.target_pose_topic, 10)
        self.create_subscription(Empty, '/corner_ack',      self.on_corner_ack,      10)
        self.create_subscription(Empty, '/weld_cycle_done', self.on_weld_cycle_done, 10)
        self.timer = self.create_timer(1.0 / max(self.follow_hz, 1.0), self.on_timer)

        total = len(self.waypoints)
        self.get_logger().info(
            f"FourCornerPublisher 시작: {WAYPOINTS_PER_EDGE}점/변 × 4변 = {total}점")
        self.get_logger().info(f"  plate_front_y={self.plate_front_y:.4f}")
        for i in range(4):
            edge_names = ["TL→TR", "TR→BR", "BR→BL", "BL→TL"]
            s = self.waypoints[i * WAYPOINTS_PER_EDGE]
            e = self.waypoints[i * WAYPOINTS_PER_EDGE + WAYPOINTS_PER_EDGE - 1]
            self.get_logger().info(
                f"  변{i}({edge_names[i]}): {s.round(3)} → {e.round(3)}")

    # ── 카메라 콜백 ─────────────────────────────────────────────────────────
    def on_image(self, msg: Image):
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return

        corners, mask, area = find_green_square_corners_bgr(
            img_bgr, area_min_px2=self.area_min_px2)

        if corners is not None:
            if not self.marker_detected:
                if self._needs_reset:
                    # weld_cycle_done 이후 첫 감지 → 처음부터 시작
                    self.corner_idx    = 0
                    self.all_done      = False
                    self._needs_reset  = False
                    self.get_logger().info(
                        f"[FCP] 마커 감지 area={area:.0f} → waypoint[0] 시작")
                else:
                    # 작업 중 잠깐 소실 후 재감지 → 현재 인덱스 유지
                    self.get_logger().info(
                        f"[FCP] 마커 재감지 → waypoint[{self.corner_idx}] 재개")
            self.marker_detected = True
        else:
            if self.marker_detected:
                self.get_logger().info("[FCP] 마커 소실")
            self.marker_detected = False

        if self.debug_view:
            vis = img_bgr.copy()
            if corners is not None:
                pts = corners.astype(np.int32)
                cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            label = f"corner={self.corner_idx}  done={self.all_done}"
            cv2.putText(vis, label, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("mask", mask)
            cv2.imshow("view", vis)
            cv2.waitKey(1)

    # ── corner_ack 수신 ──────────────────────────────────────────────────────
    def on_corner_ack(self, _msg):
        if self.all_done:
            return
        prev = self.corner_idx
        self.corner_idx += 1
        total = len(self.waypoints)
        if self.corner_idx >= total:
            self.all_done = True
            self.get_logger().info(f"[FCP] 전체 {total}점 완료 → weld_cycle_done 대기")
        else:
            edge     = self.corner_idx // WAYPOINTS_PER_EDGE
            step     = self.corner_idx %  WAYPOINTS_PER_EDGE
            self.get_logger().info(
                f"[FCP] ack: {prev}→{self.corner_idx}  (변{edge} 스텝{step+1}/{WAYPOINTS_PER_EDGE})")

    # ── weld_cycle_done 수신 ─────────────────────────────────────────────────
    def on_weld_cycle_done(self, _msg):
        self._needs_reset    = True    # 다음 감지 시 idx=0부터 시작
        self.all_done        = False
        self.marker_detected = False   # 강제 재검출
        self.get_logger().info("[FCP] weld_cycle_done → 마커 재검출 대기")

    # ── 타이머: 현재 코너 반복 발행 ─────────────────────────────────────────
    def on_timer(self):
        if self.all_done:
            return
        # _needs_reset=True: 아직 시작 전 → 마커 감지 기다림
        # _needs_reset=False: 시퀀스 진행 중 → 마커 소실이어도 계속 발행
        if self._needs_reset:
            return
        if self.corner_idx >= len(self.waypoints):
            return

        target = self.waypoints[self.corner_idx]
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
        self.pub.publish(ps)


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
