#!/usr/bin/env python3
import math
import os
import sys
import numpy as np
from enum import Enum, auto
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, JointState
import cv2

sys.path.insert(0, os.path.dirname(__file__))
try:
    from ur10_ik import ik_deg, fk
except ImportError:
    print("[Error] ur10_ik.py를 찾을 수 없습니다.")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
#  ▌ 상수 및 설정
# ═══════════════════════════════════════════════════════════════════════════════
WHEEL_JOINTS = ["front_left_wheel_joint", "front_right_wheel_joint",
                "rear_left_wheel_joint", "rear_right_wheel_joint"]
ARM_JOINTS   = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

MAIN_HZ = 20.0

MARKER_SIZE_M = 0.1
MARKER_HALF   = MARKER_SIZE_M / 2.0
MARKER_CORNERS_LOCAL = np.array([
    [-MARKER_HALF,  MARKER_HALF, 0.0],
    [ MARKER_HALF,  MARKER_HALF, 0.0],
    [ MARKER_HALF, -MARKER_HALF, 0.0],
    [-MARKER_HALF, -MARKER_HALF, 0.0],
], dtype=np.float32)

# ── 카메라 파라미터 ──────────────────────────────────────────────────────────
IMG_WIDTH  = 1280
IMG_HEIGHT = 720
FOCAL_LENGTH_PX = (IMG_WIDTH / 2.0) / math.tan(math.radians(60.0) / 2.0)
CAMERA_MATRIX = np.array([
    [FOCAL_LENGTH_PX, 0,               IMG_WIDTH  / 2.0],
    [0,               FOCAL_LENGTH_PX, IMG_HEIGHT / 2.0],
    [0,               0,               1               ],
], dtype=np.float64)
DIST_COEFFS = np.zeros((5, 1))

# ── 주행 파라미터 ────────────────────────────────────────────────────────────
SEARCH_LIN_VEL     = 40.0
APPROACH_KP_LIN    = 60.0
MAX_LIN_VEL        = 50.0
ALIGN_THRESHOLD_PX = 15

# ── 용접 파라미터 ────────────────────────────────────────────────────────────
HOME_POSE_DEG = [0.0, -90.0, 0.0, 0.0, 0.0, 0.0]

# ── [핵심] 마커별 UR10 base 기준 Y, Z 고정값 ─────────────────────────────────
#
# 도출 근거 (water_env.py + 기하학적 계산):
#   PLATE_POS=(0.3, 0.7, 0.45), PLATE_ROT_Z=90°
#   UR10 base world = (0, jackal_Y, 0.25)
#
#   Y (마커까지 수평거리):
#     Jackal은 X축(World +Y)으로만 이동하므로
#     UR10 base → 용접판 앞면 거리 = PLATE_POS.x - cam_offset ≈ 0.295m
#     방향이 UR10 base -Y이므로 MARKER_FIXED_Y = -0.295
#
#   Z (높이, 마커별로 다름):
#     마커 world Z - UR10 base world Z(0.25)
#     마커1(GreenMarker_1): world Z=0.40 → UR10 base Z = 0.40-0.25 = 0.15
#     마커2(GreenMarker_2): world Z=0.50 → UR10 base Z = 0.50-0.25 = 0.25
#     마커3(GreenMarker_3): world Z=0.45 → UR10 base Z = 0.45-0.25 = 0.20
#
#   X (좌우 위치):
#     solvePnP tvec_X → base_X = -tvec_X (좌우는 이미지로 정확히 계산)
#
MARKER_FIXED_Y = 0.68   # 모든 마커 공통 Y (UR10 base 기준, 항상 일정)

# MARKER_FIXED_Z 제거: Z 중심은 solvePnP tvec을 통해 직접 계산
# (half_z = half_x 로 정사각형 강제 → Z 고정값 불필요)

# WELD_OFFSET: 판 표면에서 용접봉 접근 거리 (Y 방향, 양수 = 마커 쪽)
WELD_OFFSET_M = 0.1

# ── [USD 스케일 보정 - USD 상 ur10 크기와 실제 값이 맞지 않아서 보정한 수치] ──────────────────────────────────────────────────────────
# X_SCALE: solvePnP가 반환하는 표준 X (m) → USD UR10 X 로 변환하는 비율.
#   추정 근거: Y 고정값(0.75) / 기하학적 정답(0.295) ≈ 2.54
#   → X, Z 모두 동일 스케일 적용. 맞지 않으면 이 값만 조정하세요.
X_SCALE = 2.85   # ← 조정 포인트: X가 너무 좁으면 키우고, 너무 넓으면 줄이세요

# Z_OFFSET_M: solvePnP 계산 Z에 더할 오프셋 (양수 = 위로)
# 계산된 Z가 실제 마커보다 낮으면 이 값을 키우세요.
Z_OFFSET_M = 0.06  # ← 조정 포인트

# ── 비드 용접 파라미터 ────────────────────────────────────────────────────────
WELD_BEADS_PER_SIDE  = 15    # 한 변당 비드 수 (양 끝점 포함, 총 60점)
WELD_BEAD_HOLD_SEC   = 1.5   # 각 비드 지점에서 정지 대기 시간 (초)
WELD_INTERP_STEPS    = 30    # 비드 간 이동 보간 스텝 수

TOTAL_MARKERS = 3
SETTLE_SEC    = 1.5
LEAVE_SPEED   = 40.0

# ── 좌표 변환 (X, Z 성분 사용) ───────────────────────────────────────────────
# base_X = T_base_marker_raw[0,3] * X_SCALE  (solvePnP → UR10 base X)
# base_Y = MARKER_FIXED_Y                    (고정)
# base_Z = T_base_marker_raw[2,3] * X_SCALE  (solvePnP → UR10 base Z)
# T_BASE_TO_OPTICAL: optical frame → UR10 base frame 변환
T_BASE_TO_OPTICAL = np.array([
    [-1.,  0.,  0.,  0.   ],
    [ 0.,  0., -1., -0.1  ],
    [ 0., -1.,  0.,  0.19 ],
    [ 0.,  0.,  0.,  1.   ],
], dtype=np.float64)


class State(Enum):
    INIT            = auto()
    SEARCHING       = auto()
    ALIGNING_BASE   = auto()
    ALIGNING_ARM    = auto()
    WELDING         = auto()
    FINISHING_WELD  = auto()
    LEAVING_CURRENT = auto()
    DONE            = auto()
    ERROR_STOP      = auto()


def detect_green_marker(img_bgr):
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([40, 80, 80]), np.array([80, 255, 255]))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return None, 0.0, None, None
    c    = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 200:
        return None, 0.0, None, None

    rect = cv2.minAreaRect(c)
    box  = cv2.boxPoints(rect)
    box  = np.intp(box)
    pts  = np.array(box, dtype="float32")
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32), area, \
           (int(rect[0][0]), int(rect[0][1])), box


class VisionWeldingNode(Node):
    def __init__(self):
        super().__init__("vision_welding_node")
        self.joint_pub   = self.create_publisher(JointState, "/joint_command", 10)
        self.img_sub     = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, qos_profile_sensor_data)
        self.caminfo_sub = self.create_subscription(
            CameraInfo, "/camera/camera_info", self.caminfo_callback, 10)

        self.camera_matrix    = CAMERA_MATRIX.copy()
        self.dist_coeffs      = DIST_COEFFS.copy()
        self.img_width        = IMG_WIDTH
        self.img_height       = IMG_HEIGHT
        self.caminfo_received = False

        self.state               = State.INIT
        self.markers_done        = 0
        self.target_wheel_vel    = [0.0, 0.0]
        self.current_arm_deg     = list(HOME_POSE_DEG)
        self.last_ik_success_deg = list(HOME_POSE_DEG)

        self.latest_img      = None
        self.marker_img_pts  = None
        self.marker_area     = 0.0
        self.marker_center   = (IMG_WIDTH // 2, IMG_HEIGHT // 2)
        self.marker_visible  = False

        self.weld_trajectory = []
        self.weld_step_idx   = 0
        self.weld_interp_idx = 0
        self.weld_wait_until = None
        self.state_timer     = None

        self.create_timer(1.0 / MAIN_HZ, self.control_loop)
        self.get_logger().info(
            f"비전 기반 용접 시작 | Y고정={MARKER_FIXED_Y}m | "
            f"비드/변={WELD_BEADS_PER_SIDE} | 대기={WELD_BEAD_HOLD_SEC}s")
        self.change_state(State.SEARCHING)

    def caminfo_callback(self, msg):
        if self.caminfo_received:
            return
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        D = np.array(msg.d, dtype=np.float64)
        self.camera_matrix    = K
        self.dist_coeffs      = D.reshape(-1, 1) if D.ndim == 1 else D
        self.img_width        = msg.width
        self.img_height       = msg.height
        self.caminfo_received = True
        self.get_logger().info(
            f"[CameraInfo] {msg.width}x{msg.height}, fx={K[0,0]:.1f}")

    def change_state(self, new_state, delay=0.0):
        if delay > 0:
            self.target_wheel_vel = [0.0, 0.0]
            self.state_timer = self.get_clock().now().nanoseconds + int(delay * 1e9)
        else:
            self.state_timer = None
        self.state = new_state
        self.get_logger().info(f"State -> {self.state.name}")

    def image_callback(self, msg):
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, -1)
            if msg.encoding == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif msg.encoding == "rgba8":
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

            if not hasattr(self, '_res_warned') and \
               (msg.width != self.img_width or msg.height != self.img_height):
                self.get_logger().warn(
                    f"[해상도 불일치] {self.img_width}x{self.img_height} "
                    f"→ 실제 {msg.width}x{msg.height}")
                self.img_width  = msg.width
                self.img_height = msg.height
                if not self.caminfo_received:
                    fl = (msg.width / 2.0) / math.tan(math.radians(60.0) / 2.0)
                    self.camera_matrix = np.array(
                        [[fl,0,msg.width/2.],[0,fl,msg.height/2.],[0,0,1]],
                        dtype=np.float64)
                self._res_warned = True

            self.latest_img = img.copy()
            pts, area, center, box = detect_green_marker(img)

            if pts is not None:
                self.marker_img_pts = pts
                self.marker_area    = area
                self.marker_center  = center
                self.marker_visible = True
                cv2.drawContours(self.latest_img, [box], 0, (0, 0, 255), 2)
                cv2.circle(self.latest_img, center, 5, (0, 255, 255), -1)
            else:
                self.marker_visible = False

            label = (f"State:{self.state.name} | "
                     f"Done:{self.markers_done}/{TOTAL_MARKERS}")
            cv2.putText(self.latest_img, label,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Robot Vision", self.latest_img)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warn(f"image_callback 오류: {e}")

    def control_loop(self):
        now = self.get_clock().now().nanoseconds
        if self.state_timer and now < self.state_timer:
            self.publish_command()
            return
        self.state_timer = None

        if   self.state == State.SEARCHING:       self.process_searching()
        elif self.state == State.ALIGNING_BASE:   self.process_aligning_base()
        elif self.state == State.ALIGNING_ARM:    self.process_aligning_arm()
        elif self.state == State.WELDING:         self.process_welding(now)
        elif self.state == State.FINISHING_WELD:  self.process_finishing_weld()
        elif self.state == State.LEAVING_CURRENT: self.process_leaving_current()

        self.publish_command()

    def process_searching(self):
        self.current_arm_deg = list(HOME_POSE_DEG)
        if self.marker_visible:
            self.change_state(State.ALIGNING_BASE)
        else:
            self.target_wheel_vel = [SEARCH_LIN_VEL, 0.0]

    def process_aligning_base(self):
        if not self.marker_visible:
            self.change_state(State.SEARCHING)
            return
        cx    = self.img_width / 2.0
        err_x = cx - self.marker_center[0]
        if abs(err_x) <= ALIGN_THRESHOLD_PX:
            self.target_wheel_vel = [0.0, 0.0]
            self.change_state(State.ALIGNING_ARM, delay=SETTLE_SEC)
        else:
            lin_vel = np.clip(
                APPROACH_KP_LIN * (err_x / cx), -MAX_LIN_VEL, MAX_LIN_VEL)
            self.target_wheel_vel = [lin_vel, 0.0]

    def process_aligning_arm(self):
        if not self.marker_visible or self.marker_img_pts is None:
            self.change_state(State.SEARCHING)
            return

        success, rvec, tvec = cv2.solvePnP(
            MARKER_CORNERS_LOCAL, self.marker_img_pts,
            self.camera_matrix, self.dist_coeffs)
        if not success:
            return

        # ── X는 solvePnP, Y·Z는 계산값 ──────────────────────────────────
        R_cm, _ = cv2.Rodrigues(rvec)
        T_optical_marker = np.eye(4)
        T_optical_marker[:3, :3] = R_cm
        T_optical_marker[:3,  3] = tvec.squeeze()

        T_base_marker_raw = T_BASE_TO_OPTICAL @ T_optical_marker

        # 마커 중심 X: solvePnP + 스케일 보정
        center_x = T_base_marker_raw[0, 3] * X_SCALE

        # 마커 중심 Z: solvePnP tvec Z → base Z 변환 후 스케일 보정 + 오프셋
        center_z = T_base_marker_raw[2, 3] * X_SCALE + Z_OFFSET_M

        # ── 픽셀 박스 너비로 half_x 역산, half_z = half_x (정사각형) ─────
        pts = self.marker_img_pts  # shape (4, 2), [TL, TR, BR, BL]
        tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]

        pixel_width = float(np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
        pixel_width = max(pixel_width, 1.0)

        # m_per_px: 픽셀 → 미터 (X_SCALE 포함)
        m_per_px = (MARKER_SIZE_M * X_SCALE) / pixel_width
        half_x   = (pixel_width / 2.0) * m_per_px   # = MARKER_SIZE_M * X_SCALE / 2
        half_z   = half_x                            # 정사각형 → Z 크기 동일

        self.get_logger().info(
            f"[DEBUG] tvec=[{tvec[0][0]:.3f},{tvec[1][0]:.3f},{tvec[2][0]:.3f}] | "
            f"pix_w={pixel_width:.1f} | half={half_x:.4f}m | "
            f"center=({center_x:.3f}, {MARKER_FIXED_Y:.3f}, {center_z:.3f})")

        # ── 4 코너 정의 (TL→TR→BR→BL, 닫힌 경로) ────────────────────────
        corners = np.array([
            [center_x - half_x, MARKER_FIXED_Y, center_z + half_z],  # TL
            [center_x + half_x, MARKER_FIXED_Y, center_z + half_z],  # TR
            [center_x + half_x, MARKER_FIXED_Y, center_z - half_z],  # BR
            [center_x - half_x, MARKER_FIXED_Y, center_z - half_z],  # BL
        ])

        # ── 각 변을 WELD_BEADS_PER_SIDE 개 비드 점으로 보간 ──────────────
        # 한 변: 시작점 포함, 끝점 제외 (다음 변의 시작점과 중복 방지)
        # 마지막 변만 끝점(TL) 포함하여 완전히 닫음
        path_points = []
        for side in range(4):
            p_start = corners[side]
            p_end   = corners[(side + 1) % 4]
            include_end = (side == 3)   # 마지막 변만 끝점(=TL) 포함
            n = WELD_BEADS_PER_SIDE + (1 if include_end else 0)
            for k in range(n):
                t = k / float(WELD_BEADS_PER_SIDE)  # 0.0 ~ 1.0
                path_points.append(p_start + t * (p_end - p_start))
        path_points = np.array(path_points)  # shape (61, 3)

        # ── IK 계산: 모든 비드 점 ────────────────────────────────────────
        WELDING_ROD_LENGTH = 0.035
        trajectory_deg = []
        q_init = np.array(self.last_ik_success_deg)

        for i, pt in enumerate(path_points):
            target_pos    = pt.copy()
            target_pos[1] += WELD_OFFSET_M - WELDING_ROD_LENGTH

            q_sol_deg, ok = ik_deg(target_pos, q_init_deg=q_init)
            if ok:
                trajectory_deg.append(q_sol_deg)
                q_init = q_sol_deg
                self.get_logger().info(
                    f"  IK pt[{i:02d}/{len(path_points)}] 성공: "
                    f"({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
            else:
                self.get_logger().error(
                    f"IK 실패: pt[{i}]=({target_pos[0]:.3f}, "
                    f"{target_pos[1]:.3f}, {target_pos[2]:.3f})")
                self.change_state(State.SEARCHING)
                return

        self.weld_trajectory = trajectory_deg
        self.weld_step_idx   = 0
        self.weld_interp_idx = 0
        self.weld_wait_until = None
        self.change_state(State.WELDING)

    def process_welding(self, now_ns):
        traj = self.weld_trajectory
        idx  = self.weld_step_idx

        # 모든 비드 점 처리 완료
        if idx >= len(traj):
            self.get_logger().info("용접 완료.")
            self.change_state(State.FINISHING_WELD)
            return

        # ── 대기 중 ──────────────────────────────────────────────────────
        if self.weld_wait_until is not None:
            if now_ns < self.weld_wait_until:
                return   # 현재 비드 지점에서 정지 대기
            # 대기 끝 → 다음 비드로 이동 준비
            self.weld_wait_until  = None
            self.weld_step_idx   += 1
            self.weld_interp_idx  = 0
            idx = self.weld_step_idx
            if idx >= len(traj):
                self.get_logger().info("용접 완료.")
                self.change_state(State.FINISHING_WELD)
                return

        # ── 이전 비드 → 현재 비드 이동 보간 ─────────────────────────────
        if idx == 0:
            # 첫 번째 비드: 홈 포즈 → 첫 비드
            q_start = np.array(self.last_ik_success_deg)
        else:
            q_start = traj[idx - 1]
        q_end = traj[idx]

        if self.weld_interp_idx < WELD_INTERP_STEPS:
            ratio = self.weld_interp_idx / float(WELD_INTERP_STEPS)
            self.current_arm_deg  = (q_start + ratio * (q_end - q_start)).tolist()
            self.weld_interp_idx += 1
        else:
            # 비드 지점 도착 → 정지 & 대기 시작
            self.current_arm_deg     = q_end.tolist()
            self.last_ik_success_deg = q_end
            self.weld_wait_until     = now_ns + int(WELD_BEAD_HOLD_SEC * 1e9)
            bead_total = len(traj)
            self.get_logger().info(
                f"  비드 [{idx+1:02d}/{bead_total}] 도착 → {WELD_BEAD_HOLD_SEC}s 대기")

    def process_finishing_weld(self):
        self.current_arm_deg = list(HOME_POSE_DEG)
        self.markers_done   += 1
        if self.markers_done >= TOTAL_MARKERS:
            self.change_state(State.DONE, delay=SETTLE_SEC)
        else:
            self.change_state(State.LEAVING_CURRENT, delay=SETTLE_SEC)

    def process_leaving_current(self):
        if self.marker_visible:
            self.target_wheel_vel = [LEAVE_SPEED, 0.0]
        else:
            self.get_logger().info("마커 이탈 완료. 다음 마커 탐색.")
            self.change_state(State.SEARCHING, delay=0.5)

    def publish_command(self):
        msg              = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name         = WHEEL_JOINTS + ARM_JOINTS
        msg.velocity     = [self.target_wheel_vel[0]] * 4 + [0.0] * 6
        msg.position     = [0.0] * 4 + np.radians(self.current_arm_deg).tolist()
        self.joint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VisionWeldingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.target_wheel_vel = [0.0, 0.0]
            node.publish_command()
            node.destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
