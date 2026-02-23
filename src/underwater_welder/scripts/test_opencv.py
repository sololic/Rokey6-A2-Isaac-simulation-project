#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def order_points_clockwise(pts_xy):
    pts = np.array(pts_xy, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.int32)


def find_green_rectangle_corners_bgr(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 형광 초록(네온 그린) 포함한 Green HSV 범위
    # 기본 추천 범위: H 35~90, S/V는 넉넉히
    lower = np.array([35, 80, 50])
    upper = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask, 0.0

    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < 800.0:
        return None, mask, area

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    corners = order_points_clockwise(box)
    return corners.tolist(), mask, area


class RedRectMemoryNode(Node):
    def __init__(self):
        super().__init__('red_rect_memory_node')

        # 파라미터
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('save_path', '/home/rokey/hamtaro/src/underwater_welder/rectangle.json')
        self.declare_parameter('save_every_n', 10)  # N프레임마다 저장 (부하/파일쓰기 줄임)
        self.declare_parameter('debug_view', False)

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.save_path = Path(self.get_parameter('save_path').get_parameter_value().string_value)
        self.save_every_n = int(self.get_parameter('save_every_n').get_parameter_value().integer_value)
        self.debug_view = bool(self.get_parameter('debug_view').get_parameter_value().bool_value)

        self.bridge = CvBridge()
        self.frame_count = 0
        self.last_saved = None  # 마지막 저장 corners

        self.sub = self.create_subscription(Image, self.image_topic, self.on_image, 10)

        self.get_logger().info(f"Subscribed: {self.image_topic}")
        self.get_logger().info(f"Save path: {self.save_path}")
        self.get_logger().info(f"save_every_n: {self.save_every_n}, debug_view: {self.debug_view}")

    def on_image(self, msg: Image):
        self.frame_count += 1
        if self.save_every_n > 1 and (self.frame_count % self.save_every_n != 0):
            return

        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge convert failed: {e}")
            return

        corners, mask, area = find_green_rectangle_corners_bgr(img_bgr)
        if corners is None:
            self.get_logger().warn(f"No red rectangle. area={area:.1f}")
            if self.debug_view:
                cv2.imshow("mask", mask)
                cv2.imshow("view", img_bgr)
                cv2.waitKey(1)
            return

        # 같은 결과 반복 저장 방지(옵션)
        if self.last_saved == corners:
            if self.debug_view:
                self._show_debug(img_bgr, mask, corners)
            return

        data = {
            "timestamp": time.time(),
            "image_topic": self.image_topic,
            "corners_px": corners,  # [[x,y], [x,y], [x,y], [x,y]]: tl,tr,br,bl
            "area_px2": area,
        }

        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            self.last_saved = corners
            self.get_logger().info(f"Saved corners -> {self.save_path}  corners={corners}")
        except Exception as e:
            self.get_logger().error(f"Save failed: {e}")

        if self.debug_view:
            self._show_debug(img_bgr, mask, corners)

    def _show_debug(self, img_bgr, mask, corners):
        vis = img_bgr.copy()
        pts = np.array(corners, dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        for i, (x, y) in enumerate(corners):
            cv2.circle(vis, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(vis, str(i), (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("mask", mask)
        cv2.imshow("view", vis)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = RedRectMemoryNode()
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