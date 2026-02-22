#!/usr/bin/env python3
"""
균열 감지 로컬 테스트 스크립트
- ROS2 없이 OpenCV만으로 crack_detection_node 로직 테스트
- 샘플 이미지 자동 생성 후 감지 실행

실행:
  python3 test_crack_detection.py
"""

import cv2
import numpy as np
import os


# ── 샘플 이미지 생성 ──────────────────────────────────────────────────────────

def make_crack_image(w=640, h=480, num_cracks=3):
    """수중 금속 표면 + 균열 패턴 생성"""
    # 배경: 어두운 금속 텍스처
    img = np.random.randint(40, 80, (h, w, 3), dtype=np.uint8)
    # 약간의 노이즈로 질감 추가
    noise = np.random.randint(0, 20, (h, w, 3), dtype=np.uint8)
    img = cv2.add(img, noise)

    # 균열 그리기 (불규칙한 선)
    for _ in range(num_cracks):
        # 랜덤 시작점
        x1 = np.random.randint(50, w - 100)
        y1 = np.random.randint(50, h - 100)
        length = np.random.randint(80, 200)
        angle  = np.random.uniform(0, np.pi)

        pts = [(x1, y1)]
        for i in range(1, 12):
            angle += np.random.uniform(-0.3, 0.3)
            step = length / 12
            nx = int(pts[-1][0] + step * np.cos(angle))
            ny = int(pts[-1][1] + step * np.sin(angle))
            pts.append((nx, ny))

        # 가는 어두운 선 (균열)
        for i in range(len(pts) - 1):
            thickness = np.random.randint(1, 3)
            cv2.line(img, pts[i], pts[i+1], (10, 10, 10), thickness)
            # 약간 흰 하이라이트 (균열 가장자리)
            cv2.line(img, pts[i], pts[i+1], (90, 90, 90), 1)

    # 수중 효과: 파란색 틴트
    blue_tint = np.zeros_like(img)
    blue_tint[:, :, 0] = 20  # B
    img = cv2.add(img, blue_tint)

    return img


def make_no_crack_image(w=640, h=480):
    """균열 없는 깨끗한 금속 표면"""
    img = np.random.randint(50, 90, (h, w, 3), dtype=np.uint8)
    noise = np.random.randint(0, 15, (h, w, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    blue_tint = np.zeros_like(img)
    blue_tint[:, :, 0] = 20
    img = cv2.add(img, blue_tint)
    return img


# ── 감지 로직 (crack_detection_node와 동일) ───────────────────────────────────

def detect_crack(frame, canny_low=50, canny_high=150,
                 min_area=100, min_aspect=3.0):
    debug = frame.copy()
    h, w = frame.shape[:2]

    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray    = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, canny_low, canny_high)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges  = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_crack = None
    best_score = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        rect    = cv2.minAreaRect(cnt)
        box_w, box_h = rect[1]
        if box_w == 0 or box_h == 0:
            continue
        aspect = max(box_w, box_h) / min(box_w, box_h)
        if aspect < min_aspect:
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
            box = cv2.boxPoints(cv2.minAreaRect(best_crack))
            box = np.intp(box)
            cv2.drawContours(debug, [box], 0, (0, 0, 255), 2)
            cv2.circle(debug, (cx, cy), 8, (0, 255, 0), -1)
            cv2.putText(debug, f'CRACK ({cx},{cy})',
                        (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    status = 'CRACK DETECTED' if crack_pt else 'NO CRACK'
    color  = (0, 0, 255) if crack_pt else (0, 200, 0)
    cv2.putText(debug, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    return crack_pt, debug, edges


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    out_dir = '/tmp/crack_test'
    os.makedirs(out_dir, exist_ok=True)

    test_cases = [
        ('crack_1.png',    make_crack_image(num_cracks=1)),
        ('crack_3.png',    make_crack_image(num_cracks=3)),
        ('no_crack.png',   make_no_crack_image()),
    ]

    print("=" * 50)
    print("균열 감지 테스트")
    print("=" * 50)

    results = []
    for fname, img in test_cases:
        crack_pt, debug, edges = detect_crack(img)
        result = 'DETECTED' if crack_pt else 'NOT FOUND'
        print(f"  {fname:20s} → {result}", end='')
        if crack_pt:
            print(f"  위치: {crack_pt}")
        else:
            print()
        results.append((fname, img, debug, edges))

        # 저장
        cv2.imwrite(f'{out_dir}/{fname}', img)
        cv2.imwrite(f'{out_dir}/debug_{fname}', debug)
        cv2.imwrite(f'{out_dir}/edges_{fname}', edges)

    print(f"\n결과 이미지 저장: {out_dir}/")

    # 화면 표시 (X 환경 있을 때만)
    try:
        for fname, img, debug, edges in results:
            combined = np.hstack([
                img,
                debug,
                cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            ])
            cv2.imshow(f'{fname} | 원본 / 감지결과 / Edge', combined)
        print("\n아무 키나 누르면 종료...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        print("(디스플레이 없음 - 이미지는 파일로 저장됨)")


if __name__ == '__main__':
    main()
