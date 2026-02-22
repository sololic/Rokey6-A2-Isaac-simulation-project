#!/usr/bin/env python3
"""
균열 감지 ROS2 노드
- /camera/image_raw  구독 (sensor_msgs/Image)
- OpenCV Canny edge + contour 기반 균열 감지
- /crack_location    발행 (geometry_msgs/PointStamped)
- /crack_debug       발행 (sensor_msgs/Image, 시각화)

실행:
  ros2 run underwater_welder crack_detection_node
  또는
  python3 crack_detection_node.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np


class CrackDetectionNode(Node):
    def __init__(self):
        super().__init__('crack_detection_node')

        # 파라미터
        self.declare_parameter('canny_low',       50)
        self.declare_parameter('canny_high',      150)
        self.declare_parameter('min_crack_area',  100)   # px²
        self.declare_parameter('min_aspect_ratio', 3.0)  # 길쭉한 형태 필터

        self.canny_low       = self.get_parameter('canny_low').value
        self.canny_high      = self.get_parameter('canny_high').value
        self.min_crack_area  = self.get_parameter('min_crack_area').value
        self.min_aspect_ratio = self.get_parameter('min_aspect_ratio').value

        self.bridge = CvBridge()

        # 구독
        self.sub_image = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # 발행
        self.pub_location = self.create_publisher(
            PointStamped, '/crack_location', 10)
        self.pub_detected = self.create_publisher(
            Bool, '/crack_detected', 10)
        self.pub_debug = self.create_publisher(
            Image, '/crack_debug', 10)

        self.get_logger().info('Crack Detection Node 시작')
        self.get_logger().info(
            f'  Canny: {self.canny_low}/{self.canny_high}, '
            f'min_area: {self.min_crack_area}px²')

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'이미지 변환 실패: {e}')
            return

        crack_pt, debug_img = self.detect_crack(frame)

        # /crack_detected 발행
        detected_msg = Bool()
        detected_msg.data = (crack_pt is not None)
        self.pub_detected.publish(detected_msg)

        if crack_pt is not None:
            # /crack_location 발행
            pt_msg = PointStamped()
            pt_msg.header = msg.header
            # 픽셀 좌표 → 정규화 (-1~1 범위)
            h, w = frame.shape[:2]
            pt_msg.point.x = (crack_pt[0] - w / 2.0) / (w / 2.0)
            pt_msg.point.y = (crack_pt[1] - h / 2.0) / (h / 2.0)
            pt_msg.point.z = 0.0
            self.pub_location.publish(pt_msg)
            self.get_logger().info(
                f'균열 감지: 픽셀=({crack_pt[0]}, {crack_pt[1]})  '
                f'정규화=({pt_msg.point.x:.2f}, {pt_msg.point.y:.2f})')

        # /crack_debug 발행
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            debug_msg.header = msg.header
            self.pub_debug.publish(debug_msg)
        except Exception as e:
            self.get_logger().warn(f'디버그 이미지 발행 실패: {e}')

    def detect_crack(self, frame):
        """
        균열 감지 파이프라인
        Returns:
            crack_pt : (cx, cy) 픽셀 좌표 or None
            debug_img: 시각화 이미지
        """
        debug = frame.copy()
        h, w = frame.shape[:2]

        # 1) 전처리 - 수중 이미지 대비 향상
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # CLAHE: 수중 저대비 환경에 효과적
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2) Canny edge
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # 3) morphology - 끊어진 균열 연결
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # 4) contour 검출
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_crack = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_crack_area:
                continue

            # bounding rect로 종횡비 계산
            rect = cv2.minAreaRect(cnt)
            box_w, box_h = rect[1]
            if box_w == 0 or box_h == 0:
                continue
            aspect = max(box_w, box_h) / min(box_w, box_h)

            # 균열 판별: 길쭉하고 면적이 클수록 높은 점수
            if aspect < self.min_aspect_ratio:
                continue

            score = area * aspect
            if score > best_score:
                best_score = score
                best_crack = cnt

        crack_pt = None
        if best_crack is not None:
            M = cv2.moments(best_crack)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                crack_pt = (cx, cy)

                # 시각화
                box = cv2.boxPoints(cv2.minAreaRect(best_crack))
                box = np.intp(box)
                cv2.drawContours(debug, [box], 0, (0, 0, 255), 2)
                cv2.circle(debug, (cx, cy), 8, (0, 255, 0), -1)
                cv2.putText(debug, f'CRACK ({cx},{cy})',
                            (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 모든 후보 contour 표시 (회색)
            cv2.drawContours(debug, [best_crack], -1, (128, 128, 0), 1)

        # 상태 표시
        status = 'CRACK DETECTED' if crack_pt else 'NO CRACK'
        color  = (0, 0, 255) if crack_pt else (0, 200, 0)
        cv2.putText(debug, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        return crack_pt, debug


def main(args=None):
    rclpy.init(args=args)
    node = CrackDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
