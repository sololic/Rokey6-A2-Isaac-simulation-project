#!/usr/bin/env python3
"""
균열 위치 → UDP 소켓 브릿지
- /crack_location (PointStamped) 구독
- Isaac Sim (jackal_ur10_demo.py) 으로 UDP 전송
- 형식: "x,y"  (정규화 좌표 -1~1)

실행:
  source /opt/ros/humble/setup.bash
  python3 crack_to_socket.py

Isaac Sim 수신:
  import socket, json
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  sock.bind(('127.0.0.1', 9998))
  data, _ = sock.recvfrom(64)
  x, y = map(float, data.decode().split(','))
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool
import socket


ISAAC_IP   = '127.0.0.1'
ISAAC_PORT = 9998


class CrackToSocket(Node):
    def __init__(self):
        super().__init__('crack_to_socket')

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.sub_loc = self.create_subscription(
            PointStamped, '/crack_location', self.location_cb, 10)
        self.sub_det = self.create_subscription(
            Bool, '/crack_detected', self.detected_cb, 10)

        self.get_logger().info(
            f'crack_to_socket 시작 → UDP {ISAAC_IP}:{ISAAC_PORT}')

    def location_cb(self, msg: PointStamped):
        x = msg.point.x
        y = msg.point.y
        payload = f'{x:.4f},{y:.4f}'.encode()
        self.sock.sendto(payload, (ISAAC_IP, ISAAC_PORT))
        self.get_logger().info(f'전송: x={x:.3f}, y={y:.3f}')

    def detected_cb(self, msg: Bool):
        if not msg.data:
            # 균열 없음 → Isaac Sim에 알림
            self.sock.sendto(b'none', (ISAAC_IP, ISAAC_PORT))


def main(args=None):
    rclpy.init(args=args)
    node = CrackToSocket()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
