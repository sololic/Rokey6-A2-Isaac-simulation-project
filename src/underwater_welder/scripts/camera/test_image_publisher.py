#!/usr/bin/env python3
"""
테스트용 균열 이미지 퍼블리셔 - 삐뚤빼뚤 사각형 경로
- 균열이 사각형 경로를 따라 이동하는 이미지를 /camera/image_raw 로 발행
- crack_detection_node.py 테스트용

실행:
  source /opt/ros/humble/setup.bash
  python3 test_image_publisher.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

# 사각형 웨이포인트 (정규화 이미지 좌표 -1~1)
# jackal_ur10_demo.py 의 CrackMarker 위치와 1:1 대응:
#   이미지 x → shoulder_pan (-1=좌60°, +1=우60°)
#   이미지 y → shoulder_lift (-1=낮음, +1=높음)
RECT_WAYPOINTS = [
    (-0.55,  0.15),   # 좌상  → CrackMarker_0 (-1.0,  2.5)
    ( 0.55,  0.15),   # 우상  → CrackMarker_1 (+1.0,  2.5)
    ( 0.55, -0.15),   # 우하  → CrackMarker_2 (+1.0,  3.5)
    (-0.55, -0.15),   # 좌하  → CrackMarker_3 (-1.0,  3.5)
]
WAYPOINT_HOLD_SEC = 4.0   # 각 꼭짓점 유지 시간
TRAVEL_SEC        = 3.0   # 꼭짓점 간 이동 시간
WOBBLE_AMP        = 0.06  # 경로 흔들림 (수중 파도 효과)
PUBLISH_HZ        = 10


class TestImagePublisher(Node):
    def __init__(self):
        super().__init__('test_image_publisher')
        self.pub    = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()
        self.t      = 0.0
        self.dt     = 1.0 / PUBLISH_HZ

        # 경로 파라미터
        self._n_wp     = len(RECT_WAYPOINTS)
        self._wp_idx   = 0       # 현재 목표 웨이포인트
        self._seg_t    = 0.0     # 현재 세그먼트 내 경과 시간
        self._holding  = True    # True=유지중, False=이동중
        self._crack_x  = RECT_WAYPOINTS[0][0]
        self._crack_y  = RECT_WAYPOINTS[0][1]

        self.timer = self.create_timer(self.dt, self.publish_image)
        self.get_logger().info('테스트 이미지 퍼블리셔 시작 (삐뚤빼뚤 사각형 경로)')
        self.get_logger().info(
            f'웨이포인트: {RECT_WAYPOINTS}')

    def _update_path(self):
        """삐뚤빼뚤 사각형 경로 업데이트"""
        self._seg_t += self.dt
        wp_cur  = RECT_WAYPOINTS[self._wp_idx]
        wp_next = RECT_WAYPOINTS[(self._wp_idx + 1) % self._n_wp]

        if self._holding:
            # 현재 꼭짓점 유지
            self._crack_x = wp_cur[0] + np.random.uniform(-WOBBLE_AMP, WOBBLE_AMP)
            self._crack_y = wp_cur[1] + np.random.uniform(-WOBBLE_AMP, WOBBLE_AMP)
            if self._seg_t >= WAYPOINT_HOLD_SEC:
                self._holding = False
                self._seg_t   = 0.0
                self.get_logger().info(
                    f'이동 시작: WP{self._wp_idx}→WP{(self._wp_idx+1)%self._n_wp}')
        else:
            # 다음 꼭짓점으로 이동 (smoothstep 보간)
            alpha = min(self._seg_t / TRAVEL_SEC, 1.0)
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)   # smoothstep
            # 삐뚤빼뚤: 진행 방향 수직으로 사인파 추가
            dx = wp_next[0] - wp_cur[0]
            dy = wp_next[1] - wp_cur[1]
            perp_x = -dy
            perp_y =  dx
            wobble = WOBBLE_AMP * 2.0 * math.sin(alpha * math.pi)
            self._crack_x = (wp_cur[0] + alpha * dx
                             + wobble * perp_x
                             + np.random.uniform(-WOBBLE_AMP * 0.3, WOBBLE_AMP * 0.3))
            self._crack_y = (wp_cur[1] + alpha * dy
                             + wobble * perp_y
                             + np.random.uniform(-WOBBLE_AMP * 0.3, WOBBLE_AMP * 0.3))
            if self._seg_t >= TRAVEL_SEC:
                self._wp_idx  = (self._wp_idx + 1) % self._n_wp
                self._holding = True
                self._seg_t   = 0.0
                self.get_logger().info(
                    f'WP{self._wp_idx} 도착 ({self._crack_x:.2f}, {self._crack_y:.2f})')

    def _make_frame(self, nx, ny):
        """균열이 (nx, ny) 위치에 있는 640×480 이미지 생성"""
        w, h = 640, 480
        # 픽셀 좌표 변환
        px = int((nx + 1.0) / 2.0 * w)
        py = int((1.0 - (ny + 1.0) / 2.0) * h)
        px = np.clip(px, 30, w - 30)
        py = np.clip(py, 30, h - 30)

        # 배경: 수중 금속 텍스처
        img = np.random.randint(35, 65, (h, w, 3), dtype=np.uint8)
        img[:, :, 0] = np.clip(img[:, :, 0].astype(int) + 25, 0, 255)  # 파란 틴트

        # 균열 (불규칙한 선)
        pts  = [(px, py)]
        ang  = np.random.uniform(0.2, 0.8)
        for _ in range(14):
            ang += np.random.uniform(-0.25, 0.25)
            nx2  = int(pts[-1][0] + 12 * math.cos(ang))
            ny2  = int(pts[-1][1] + 12 * math.sin(ang))
            pts.append((nx2, ny2))
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], (6, 6, 6),  2)
            cv2.line(img, pts[i], pts[i + 1], (75, 75, 75), 1)

        # 경로 오버레이 (디버그용 얇은 흰 점선)
        for k, (wx, wy) in enumerate(RECT_WAYPOINTS):
            wpx = int((wx + 1.0) / 2.0 * w)
            wpy = int((1.0 - (wy + 1.0) / 2.0) * h)
            color = (0, 255, 0) if k == self._wp_idx else (80, 80, 80)
            cv2.circle(img, (wpx, wpy), 6, color, -1)
        # 꼭짓점 연결선 (경로 표시)
        for k in range(len(RECT_WAYPOINTS)):
            p1 = RECT_WAYPOINTS[k]
            p2 = RECT_WAYPOINTS[(k + 1) % len(RECT_WAYPOINTS)]
            pp1 = (int((p1[0]+1)/2*w), int((1-(p1[1]+1)/2)*h))
            pp2 = (int((p2[0]+1)/2*w), int((1-(p2[1]+1)/2)*h))
            cv2.line(img, pp1, pp2, (50, 50, 50), 1)

        # 균열 중심 표시
        cv2.circle(img, (px, py), 5, (0, 0, 200), -1)

        return img

    def publish_image(self):
        self._update_path()
        frame = self._make_frame(self._crack_x, self._crack_y)
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TestImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
