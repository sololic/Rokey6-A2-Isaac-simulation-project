#!/usr/bin/env python3
"""
ROS2 /cmd_vel → UDP 소켓 브릿지 (시스템 Python 3.10으로 실행)

실행: python3 ros_to_socket_bridge.py
      (isaac sim python.sh 이 아닌 일반 python3)
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import socket
import struct

ISAAC_HOST = "127.0.0.1"
ISAAC_PORT = 9999          # Isaac Sim 쪽 수신 포트

class CmdVelBridge(Node):
    def __init__(self):
        super().__init__("cmd_vel_udp_bridge")
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.create_subscription(Twist, "/cmd_vel", self._cb, 10)
        self.get_logger().info(
            f"✓ /cmd_vel → UDP {ISAAC_HOST}:{ISAAC_PORT} 브릿지 시작"
        )

    def _cb(self, msg: Twist):
        # linear.x, angular.z 두 float64 전송 (16 bytes)
        data = struct.pack("dd", msg.linear.x, msg.angular.z)
        self._sock.sendto(data, (ISAAC_HOST, ISAAC_PORT))

def main():
    rclpy.init()
    node = CmdVelBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
