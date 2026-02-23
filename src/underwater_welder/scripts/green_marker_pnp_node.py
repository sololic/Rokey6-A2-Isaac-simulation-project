#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
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

    # keep float for solvePnP
    return np.array([tl, tr, br, bl], dtype=np.float32)


def find_green_square_corners_bgr(img_bgr, area_min_px2=800.0):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Neon green 포함 green 범위 (필요하면 넓혀)
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
    if area < area_min_px2:
        return None, mask, area

    rect = cv2.minAreaRect(c)   # center,(w,h),angle
    box = cv2.boxPoints(rect)   # 4x2 float
    corners = order_points_clockwise(box)  # tl,tr,br,bl

    return corners, mask, area


def build_K_from_hfov(w: int, h: int, hfov_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build pinhole intrinsics from horizontal FOV (degrees) and image size.
    Returns (K, D). D is zeros (no distortion) by default.
    """
    # Safety clamp: pinhole model is unreliable near/over 180 degrees
    hfov = float(hfov_deg)
    if not np.isfinite(hfov) or hfov <= 1.0:
        hfov = 90.0
    if hfov >= 179.0:
        hfov = 170.0  # clamp for stability

    hfov_rad = np.deg2rad(hfov)

    fx = (w / 2.0) / np.tan(hfov_rad / 2.0)
    fy = fx  # simple assumption; good enough for sim unless you need high accuracy
    cx = w / 2.0
    cy = h / 2.0

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    D = np.zeros((5,), dtype=np.float64)
    return K, D


def solve_pnp_square(corners_px: np.ndarray,
                     K: np.ndarray,
                     D: np.ndarray,
                     marker_size_m: float):
    """
    corners_px: (4,2) float32 [tl,tr,br,bl]
    marker_size_m: side length in meters
    Returns: ok, rvec(3,1), tvec(3,1), distance_m
    """
    half = marker_size_m / 2.0

    # 3D model points on marker plane (z=0), order must match tl,tr,br,bl
    obj = np.array([
        [-half,  half, 0.0],  # tl
        [ half,  half, 0.0],  # tr
        [ half, -half, 0.0],  # br
        [-half, -half, 0.0],  # bl
    ], dtype=np.float32)

    img = corners_px.astype(np.float32)

    flag = cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE") else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=flag)
    if not ok:
        return False, None, None, 0.0

    dist = float(np.linalg.norm(tvec))
    return True, rvec, tvec, dist


class GreenMarkerPnPNode(Node):
    def __init__(self):
        super().__init__('green_marker_pnp_node')

        # Params
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('save_path', '/home/rokey/hamtaro/src/underwater_welder/marker_pose.json')
        self.declare_parameter('marker_size_m', 0.2)      # 너 코드 S=0.2 (보통 meters)
        self.declare_parameter('hfov_deg', 90.0)          # 수평 FOV(deg). 모르면 90부터 시작 추천
        self.declare_parameter('save_every_n', 5)
        self.declare_parameter('debug_view', False)
        self.declare_parameter('area_min_px2', 800.0)

        self.image_topic = self.get_parameter('image_topic').value
        self.save_path = Path(self.get_parameter('save_path').value)
        self.marker_size_m = float(self.get_parameter('marker_size_m').value)
        self.hfov_deg = float(self.get_parameter('hfov_deg').value)
        self.save_every_n = int(self.get_parameter('save_every_n').value)
        self.debug_view = bool(self.get_parameter('debug_view').value)
        self.area_min_px2 = float(self.get_parameter('area_min_px2').value)

        self.bridge = CvBridge()
        self.frame_count = 0
        self.last_saved = None

        # Sub (image only)
        self.create_subscription(Image, self.image_topic, self.on_image, qos_profile_sensor_data)

        self.get_logger().info(f"Subscribed image: {self.image_topic}")
        self.get_logger().info(f"marker_size_m={self.marker_size_m} hfov_deg={self.hfov_deg}")
        self.get_logger().info(f"save_path={self.save_path}")

    def on_image(self, msg: Image):
        self.frame_count += 1
        if self.save_every_n > 1 and (self.frame_count % self.save_every_n != 0):
            return

        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge convert failed: {e}")
            return

        h, w = img_bgr.shape[:2]
        K, D = build_K_from_hfov(w, h, self.hfov_deg)

        # intrinsics sanity check
        fx = float(K[0, 0]); fy = float(K[1, 1])
        if not np.isfinite(fx) or not np.isfinite(fy) or fx <= 1e-6 or fy <= 1e-6:
            self.get_logger().error(f"Invalid K computed. fx={fx}, fy={fy}, hfov_deg={self.hfov_deg}")
            return

        corners, mask, area = find_green_square_corners_bgr(img_bgr, area_min_px2=self.area_min_px2)
        if corners is None:
            self.get_logger().warn(f"No green square. area={area:.1f}")
            if self.debug_view:
                cv2.imshow("mask", mask)
                cv2.imshow("view", img_bgr)
                cv2.waitKey(1)
            return

        ok, rvec, tvec, dist = solve_pnp_square(corners, K, D, self.marker_size_m)
        if not ok or tvec is None or not np.all(np.isfinite(tvec)):
            self.get_logger().warn("solvePnP failed or produced invalid tvec")
            return

        corners_list = corners.astype(int).tolist()
        tvec_list = tvec.reshape(3).tolist()

        # avoid repeated writes
        key = (corners_list, [round(x, 6) for x in tvec_list])
        if self.last_saved == key:
            if self.debug_view:
                self._show_debug(img_bgr, mask, corners)
            return

        data = {
            "timestamp": time.time(),
            "frame": "camera",
            "image_topic": self.image_topic,
            "marker_size_m": self.marker_size_m,
            "hfov_deg_used": float(min(max(self.hfov_deg, 1.0), 170.0) if self.hfov_deg >= 179.0 else self.hfov_deg),
            "image_wh": [w, h],
            "K": K.tolist(),
            "corners_px_tl_tr_br_bl": corners_list,
            "tvec_cam_m": tvec_list,     # camera frame translation (meters)
            "distance_m": float(dist),   # norm(tvec)
        }

        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            self.last_saved = key
            self.get_logger().info(f"Saved -> {self.save_path}  dist={dist:.3f} m  tvec={tvec_list}")
        except Exception as e:
            self.get_logger().error(f"Save failed: {e}")

        if self.debug_view:
            self._show_debug(img_bgr, mask, corners)

    def _show_debug(self, img_bgr, mask, corners):
        vis = img_bgr.copy()
        pts = corners.astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        for i, (x, y) in enumerate(pts):
            cv2.circle(vis, (int(x), int(y)), 5, (255, 0, 0), -1)
            cv2.putText(vis, str(i), (int(x)+5, int(y)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.imshow("mask", mask)
        cv2.imshow("view", vis)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = GreenMarkerPnPNode()
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