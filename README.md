# 🐾 Welcome to Our 햄스터닷! 🚀
# Underwater Welding Robot Simulation

안녕하세요! 우리는 함께 성장하고 즐겁게 개발하는 **4인조 개발팀**입니다.  
귀여운 동물 친구들처럼 각자의 개성으로 멋진 프로젝트를 만들어가고 있어요.

---

### 한 줄 요약
Isaac Sim 기반 수중 환경에서 비전으로 균열을 감지하고 자동 용접하는 로봇 시뮬레이션

### 핵심 특징
- 🌊 **실시간 수중 물리**: 부력, 유체저항, 파도력 구현
- 👁️ **비전 기반 자율 제어**: HSV + PnP로 마커 인식
- 🤖 **ROS2 통합**: Isaac Sim ↔ 외부 노드 양방향 통신

---

버전 1 실행 방법과 버전 2 실행 방법이 따로 나뉘어 이후에는 아래와 같이 설명드리겠습니다.

**hello_world 폴더 안에 있는 파일 실행 방법을 v1** 

**src/underwater_welder 폴더 안에 있는 파일 실행 방법을 v2**

### 버전 차이점
| 구분 | v1 (hello_world) | v2 (underwater_welder) |
|------|------------------|------------------------|
| **구조** | BaseSample 확장 | Standalone Python |
| **실행** | Isaac Sim GUI 내부 | 터미널 직접 실행 |
| **제어** | 상태머신 (6단계) | Phase 기반 (4단계) |
| **장점** | 통합 환경, GUI 편의 | 독립 실행, 디버깅 용이 |
| **단점** | 설치 복잡 | 경로 설정 필요 |

**→ 두 버전 모두 동일한 결과, 편한 것 선택!**

---

## 목차

1. [개발 환경](#개발-환경)
2. [사용 장비](#사용-장비)
3. [디렉토리 구조](#디렉토리-구조)
4. [의존성](#의존성)
5. [주요 기능](#주요-기능)
6. [시스템 설계](#시스템-설계)
7. [알고리즘 플로우차트](#알고리즘-플로우차트)
8. [실행 방법](#실행-방법)

---

## 개발 환경

| 항목 | 내용 |
|---|---|
| **OS** | Ubuntu 22.04.5 LTS (Jammy Jellyfish) |
| **CPU** | Intel(R) Core(TM) Ultra 9 275HX |
| **RAM** | 64 GB (32GB + 32GB) DDR5 |
| **GPU** | NVIDIA GeForce RTX 5080 Laptop |
| **GPU VRAM** | 16 GB |
| **저장장치** | NVMe SSD 1TB |
| **Isaac Sim** | 5.0.0 |
| **ROS2** | Humble Hawksbill |
| **Python** | 3.10 |
| **CUDA** | 12.8 |

---

## 사용 장비

| 구분 | 내용 |
|---|---|
| **PC** | MSI Vector 16 HX AI — Intel Ultra 9 275HX, RTX 5080 Laptop (16GB), RAM 64GB |
| **시뮬레이터** | NVIDIA Isaac Sim 5.0.0 |

시뮬레이션 환경에서 사용된 로봇 및 센서 장비 목록입니다.

### 모바일 로봇 — Clearpath Jackal

| 항목 | 내용 |
|---|---|
| **제조사** | Clearpath Robotics |
| **모델** | Jackal UGV |
| **구동 방식** | 4륜 차동 구동 (Skid-steer) |
| **구동 관절 수** | 4 (front_left, front_right, rear_left, rear_right) |
| **시뮬레이션 질량** | 5,000 kg (파도 안정성을 위해 과부하 설정) |
| **USD 경로** | `jackal_and_ur10.usd` 내 `/jackal` |

### 매니퓰레이터 — Universal Robots UR10

| 항목 | 내용 |
|---|---|
| **제조사** | Universal Robots |
| **모델** | UR10 |
| **자유도 (DOF)** | 6 |
| **관절 구성** | shoulder_pan / shoulder_lift / elbow / wrist_1 / wrist_2 / wrist_3 |
| **링크 길이** | 상완 612 mm, 하완 572 mm (표준 사양) |
| **최대 가반하중** | 10 kg (표준 사양) |
| **USD 경로** | `jackal_and_ur10.usd` 내 `/jackal/ur10` |

> **참고:** 시뮬레이션에 사용된 USD 모델의 실제 스케일이 표준 UR10 DH 파라미터와 다소 차이가 있어, `X_SCALE` 등 보정 상수를 사용합니다.

### 용접봉 (End Effector)

| 항목 | 내용 |
|---|---|
| **형상** | 실린더 + 구형 팁 |
| **반지름** | 3 mm |
| **길이** | 35 mm |
| **질량** | 10 g |
| **마운트** | UR10 `ee_link`에 Fixed Joint로 고정 |
| **USD 경로** | `/World/WeldElectrode` |

### 카메라

| 항목 | 내용 |
|---|---|
| **형식** | Isaac Sim 가상 카메라 (`UsdGeom.Camera`) |
| **마운트** | UR10 `ee_link` 하위 CameraRig에 부착 |
| **해상도** | 1280 × 720 px |
| **수평 FOV** | 60° |
| **초점거리** | 8 mm (focal length) / 1,108 px (픽셀 환산) |
| **클리핑 범위** | 0.01 m ~ 100 m |
| **로컬 위치** | EE 기준 (−0.25, 0.05, 0.05) m |
| **퍼블리시 토픽** | `/camera/image_raw`, `/camera/camera_info` |

### 용접 대상 — 용접판 및 마커

**v1 기준으로 설명**
| 항목 | 내용 |
|---|---|
| **용접판 크기** | 4.5 m × 0.8 m × 12 mm (X × Z × 두께) |
| **용접판 색상** | 무광 검정 |
| **배치 위치** | World (0.3, 0.7, 0.45), Z축 90° 회전 |
| **마커 수** | 3개 (GreenMarker_1 / 2 / 3) |
| **마커 크기** | 100 mm × 100 mm 정사각형 |
| **마커 색상** | 초록 (발광 포함) |
| **마커 간격** | 로컬 X방향 0.5 m, Z방향 ±25 mm |

---

## 디렉토리 구조

**v1**
```
hello_world/
├── water_env.py                # Isaac Sim 환경 (BaseSample 상속)
├── welding_scenario_node.py    # ROS2 비전-제어 노드
├── ur10_ik.py                  # UR10 순/역기구학 모듈
└── jackal_and_ur10.usd         # 로봇 USD 파일
```

**v2**
```
src/underwater_welder/
├── jackal_and_ur10_2.usd                  # 메인 USD 씬 (Jackal + UR10)
├── scripts/
│   ├── test_tool_target_controller.py     # Isaac Sim 메인 제어 스크립트
│   ├── 1_green_marker_pnp_node.py         # ROS2 비전/용접 노드
│   ├── jackal_ur10_underwater_demo.py     # 수중 물리 데모
│   └── hello_world/
│       └── test_tool_target_controller.py # Isaac Sim 확장 복사본
```

---

## 의존성

```
# Isaac Sim 5.0.0 (별도 설치, NVIDIA Omniverse Launcher 또는 pip)
# isaacsim==5.0.0

# ROS2 Humble 패키지 (apt 설치)
# ros-humble-desktop
# ros-humble-cv-bridge
# ros-humble-sensor-msgs
# ros-humble-geometry-msgs

# Python 패키지 (pip)
numpy>=1.24.0
opencv-python>=4.8.0
```

**v1**의 경우 

cd ~/isaacsim/extension_examples/hello_world
에 /hello_world 안에 있는 아래 파일을 옮겨서 실행하여야합니다

water_env.py               
welding_scenario_node.py    
ur10_ik.py                  
jackal_and_ur10.usd         

| 항목 | 버전 |
|---|---|
| Isaac Sim | 5.0.0 |
| ROS2 | Humble |
| Python | 3.10 |
| OpenCV | 4.x |
| scipy | 1.x |
| numpy | 1.x |

---

## 주요 기능

### 수중 물리 환경 시뮬레이션

실제 수중 환경에 근접한 물리 현상을 Isaac Sim 위에서 구현합니다. 물체 표면을 격자 포인트로 이산화하여 파스칼 법칙 기반 수압과 레이놀즈 수 기반 항력을 매 물리 스텝마다 계산해 RigidBody에 직접 가합니다. 파도는 사인파 모델로 구현되며 5초마다 진행 방향이 랜덤하게 전환됩니다.

### 비전 기반 자율 마커 인식

카메라 영상에서 HSV 색공간 필터로 초록 마커를 검출하고, `cv2.solvePnP`로 카메라 프레임 기준 마커의 3D 자세를 추정합니다. Jackal이 World Y 방향으로만 이동하는 구조적 특성을 이용해 마커까지의 Y 거리를 고정값으로 처리하여 depth 추정 오차를 제거합니다.

### 자율 주행 및 마커 정렬

ROS2 상태머신이 20 Hz로 동작하며 마커 탐색 → 좌우 정렬 → 팔 정렬의 3단계로 Jackal을 제어합니다. 마커 중심이 이미지 수평 중앙 기준 ±15 px 이내로 들어오면 정렬 완료로 판정합니다.

### 용접 궤적 자동 생성 및 실행

마커 4 코너에서 각 변을 15등분하여 총 61개의 비드 지점을 생성합니다. WELDING 진입 전에 모든 지점의 역기구학을 사전 계산하고, 실행 중에는 지점 간 선형 보간으로 부드러운 궤적을 생성합니다.

### UR10 역기구학 (IK)

DH 파라미터 기반 순기구학 모델 위에서 L-BFGS-B 수치 최적화로 역기구학을 풉니다. 손목 3관절을 용접 자세로 유도하는 페널티 항을 비용 함수에 포함하고, 10개의 사전 정의 초기값으로 다중 시도하여 수렴 실패를 최소화합니다.

### 용접 이펙트 시뮬레이션

용접봉이 판 표면 5 cm 이내로 접근하면 예열 → 용접 → 냉각 상태머신이 자동 진행됩니다. 용접 비드(영구 잔류), 스파크(조명 기반, 8스텝 수명), 버블(60스텝 수명, 부상 모사)이 순차적으로 생성됩니다.

---

## 시스템 설계

### 전체 아키텍처

**v1**
```
┌─────────────────────────────────────────────────────┐
│                    Isaac Sim 5.0                     │
│                                                      │
│  ┌──────────────┐      USD Stage                    │
│  │ water_env.py │◄────  /jackal/ur10                │
│  │ (BaseSample) │       /World/WeldPlate            │
│  │              │       /World/WeldElectrode        │
│  │ • 수중 물리  │       /jackal/.../Camera          │
│  │ • 용접 이펙트│                                   │
│  │ • 씬 관리    │                                   │
│  └──────┬───────┘                                   │
│         │  ROS2 Bridge                              │
└─────────┼───────────────────────────────────────────┘
          │ /camera/image_raw
          │ /camera/camera_info        ┌──────────────┐
          ├────────────────────────────►              │
          │                            │ welding_     │
          │ /joint_command             │ scenario_    │
          ◄────────────────────────────│ node.py      │
                                       │              │
                                       │ • 상태머신   │
                                       │ • 마커 인식  │    ┌──────────┐
                                       │ • 궤적 생성  ├────► ur10_    │
                                       │              │    │ ik.py    │
                                       └──────────────┘    │          │
                                                           │ • FK/IK  │
                                                           └──────────┘
```

**v2**
```
┌─────────────────────────────────────────────────────────────────┐
│                      Isaac Sim 5.0.0                            │
│                                                                 │
│  ┌──────────┐    USD Scene     ┌──────────────────────────┐    │
│  │  Jackal  │◄───────────────►│  UR10 (6-DOF arm)        │    │
│  │ (모바일)  │  FixedJoint연결  │  Jacobian Transpose IK   │    │
│  └──────────┘                 └──────────────────────────┘    │
│       │                                  │                      │
│  kinematic                          /tool_target_pose           │
│  set_world_pose                     /ee_servo_cmd               │
│       │                                  │                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              test_tool_target_controller.py              │   │
│  │  Phase1(회전) → Phase2(주행) → Phase3(보정) → Idle(잠금)  │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │  ROS2 Bridge                     │                      │
└───────┼──────────────────────────────────┼──────────────────────┘
        │                                  │
        ▼ /camera/image_raw                ▼ /ee_pose, /plate_pose
┌───────────────────────────────────────────────────────────────┐
│                 1_green_marker_pnp_node.py                     │
│                                                               │
│  ┌─────────────────┐   정렬 완료    ┌─────────────────────┐   │
│  │  SERVOING 상태  │─────────────►│   WELDING 상태       │   │
│  │  HSV 검출       │              │  4코너 순차 발행       │   │
│  │  P제어 서보      │◄─────────────│  /tool_target_pose   │   │
│  └─────────────────┘  weld_cycle  └─────────────────────┘   │
│                          _done                                 │
└───────────────────────────────────────────────────────────────┘
```

### 모듈별 책임

| 모듈 | 책임 |
|---|---|
| `water_env.py` | Isaac Sim 씬 구성, 수중 물리 매 스텝 계산, 용접 이펙트 상태머신 |
| `welding_scenario_node.py` | ROS2 토픽 처리, 비전 인식, 주행·팔 제어 상태머신 |
| `ur10_ik.py` | DH 파라미터 기반 FK, 수치 최적화 IK |

### ROS2 토픽 인터페이스

| 토픽 | 타입 | 방향 | 내용 |
|---|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | Isaac Sim → 노드 | RGB 카메라 영상 (1280×720) |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | Isaac Sim → 노드 | 카메라 내부 파라미터 |
| `/joint_command` | `sensor_msgs/JointState` | 노드 → Isaac Sim | 바퀴 속도 (velocity) + 팔 관절각 (position) |

### 마커 좌표 추정 파이프라인

```
카메라 이미지
  └─ HSV 필터 + 컨투어 → 4코너 픽셀 좌표 (TL, TR, BR, BL)
       └─ cv2.solvePnP(MARKER_CORNERS_LOCAL)
            └─ rvec, tvec  [optical 프레임]
                 └─ T_BASE_TO_OPTICAL 행렬 곱
                      └─ UR10 base 프레임 좌표
                           ├─ X = raw_X × X_SCALE          (좌우)
                           ├─ Y = MARKER_FIXED_Y            (고정, 수직 거리)
                           └─ Z = raw_Z × X_SCALE + Z_OFFSET (높이)
```

Y를 고정값으로 대체하는 이유: Jackal이 World Y 방향으로만 이동하므로 UR10 base에서 용접판까지의 수평 거리가 항상 일정합니다. solvePnP의 depth 추정은 픽셀 노이즈와 마커 기울기에 민감하므로 기하학적 고정값이 더 안정적입니다.

### 용접 궤적 생성 구조

```
마커 중심 (center_x, FIXED_Y, center_z)
  └─ 픽셀 폭으로 half_x, half_z 역산
       └─ 4 코너 계산
            TL = (cx - hx, Y, cz + hz)
            TR = (cx + hx, Y, cz + hz)
            BR = (cx + hx, Y, cz - hz)
            BL = (cx - hx, Y, cz - hz)
                 └─ 각 변 15등분 선형 보간 → 총 61개 비드 점
                      └─ 전체 IK 사전 계산 (WELDING 진입 전)
                           └─ 실행: 30스텝 보간 이동 + 1.5초 정지 × 61회
```

### IK 최적화 전략

```
비용 함수: cost(q) = ||FK(q) - target|| + 0.1 × ||q[3:] - WELD_WRIST||
                      ↑ 위치 오차 (주 목표)     ↑ 손목 자세 유지 페널티

최적화: L-BFGS-B (경계 조건 포함 준 뉴턴법)
  └─ 초기값 우선순위:
       1. 이전 성공 관절각 (연속 수렴 가속)
       2. 사전 정의 10개 후보값
  └─ 수렴 판정: EE ↔ target 거리 < 15 mm
```

### 수중 물리 계산 구조

```
물체 등록 시 (1회):
  └─ 표면 샘플 포인트 + 법선 사전 계산 (큐브/구/실린더/원뿔별 격자)

매 물리 스텝 (_buoyancy_step):
  ├─ 파도 방향 업데이트 (5초 주기)
  ├─ 각 물체 × 각 포인트:
  │    ├─ 수압: P = ρ × g × 깊이 → 법선 방향 힘
  │    └─ 항력: Re < 1000 → 스토크스 / Re ≥ 1000 → 항력계수
  └─ 파도력: 수심별 지수 감쇠 적용
```

---

## 알고리즘 플로우차트

### 전체 시나리오 흐름

```
                        ┌─────────┐
                        │  START  │
                        └────┬────┘
                             │
                        ┌────▼────┐
                        │SEARCHING│◄──────────────────────┐
                        │전진 탐색 │                       │
                        └────┬────┘                       │
                    마커 발견  │                            │
                        ┌────▼──────────┐                 │
                        │ ALIGNING_BASE │                  │
                        │  좌우 P제어   │                  │
                        └────┬──────────┘                 │
               중앙 ±15px 이내│                            │
               1.5초 안정화  │                            │
                        ┌────▼──────────┐  IK 실패         │
                        │ ALIGNING_ARM  ├─────────────────►│
                        │ 마커 위치 계산 │                  │
                        │ 61점 IK 계산  │                  │
                        └────┬──────────┘                 │
              전체 IK 성공    │                            │
                        ┌────▼────┐                       │
                        │ WELDING │                       │
                        │61점 순회│                       │
                        └────┬────┘                       │
              61점 완료       │                            │
                        ┌────▼──────────┐                 │
                        │FINISHING_WELD │                  │
                        │ 홈 포즈 복귀  │                  │
                        └────┬──────────┘                 │
                             │markers_done++               │
               ┌─────────────┴──────────────┐             │
          < 3개 완료                    3개 완료            │
               │                            │             │
         ┌─────▼──────────┐         ┌───────▼──┐         │
         │LEAVING_CURRENT │         │   DONE   │         │
         │ 마커 시야 이탈  │         └──────────┘         │
         └─────┬──────────┘                               │
    마커 소실   │                                          │
               └──────────────────────────────────────────┘
```

### 용접 실행 루프 (WELDING 상태 내부)

```
             ┌──────────────────┐
             │  idx = 0 ~ 60    │
             └────────┬─────────┘
                      │
             ┌────────▼─────────┐
             │  보간 이동 중?    │
             │ interp_idx < 30  │
             └────┬─────────────┘
           Yes    │       No
     ┌────────────▼┐     ┌──────────────────┐
     │ 선형 보간   │     │  비드 지점 도착   │
     │ 관절각 갱신 │     │  1.5초 대기 설정  │
     └────────────┘     └────────┬──────────┘
                                  │ 1.5초 경과
                         ┌────────▼──────────┐
                         │    idx += 1       │
                         │  interp_idx = 0   │
                         └────────┬──────────┘
                                  │
                         ┌────────▼──────────┐
                         │   idx >= 61?      │
                         └────┬──────────────┘
                         Yes  │   No
                    ┌─────────┘   └──────► 처음으로
                    │
               ┌────▼──────────────┐
               │  FINISHING_WELD   │
               └───────────────────┘
```

### 용접 이펙트 상태머신 (Isaac Sim 내부)

```
         ┌──────┐
         │ IDLE │◄────────────────────────┐
         └──┬───┘                         │
  거리 ≤ 5cm│                             │
         ┌──▼──────┐                      │
         │ HEATING │  거리 > 5cm → IDLE   │
         │ (50스텝)│──────────────────────►│
         └──┬──────┘                      │
   50스텝 완료│                            │
         ┌──▼──────────────────────────┐  │
         │ WELDING (20스텝)             │  │
         │  1스텝:  비드 생성 (영구)    │  │
         │  매스텝: 스파크 (8스텝 수명) │  │
         │  20스텝: 버블 (60스텝 수명)  │  │
         └──┬──────────────────────────┘  │
  20스텝 완료│                             │
         ┌──▼────────┐                    │
         │ COOLDOWN  │                    │
         │  (15스텝) ├────────────────────┘
         └───────────┘
```

---

## 실행 방법

**v1**
---
### 1. Isaac Sim 실행

Isaac Sim을 실행하고 `Robotics Examples` 패널에서 `ROKEY → Water Environment`를 로드합니다.

`LOAD` 버튼을 누르면 `water_env.py`의 `setup_scene()`이 호출되며 다음이 자동으로 구성됩니다.

- 물리 씬 (중력, TGS 솔버)
- 수중 볼륨 렌더링
- 테스트 부유 물체 스폰
- Jackal + UR10 로봇 로드
- 용접판(검정) + 마커 3개(초록) 배치
- 용접봉 → UR10 EE Fixed Joint 연결
- 카메라 + 헤드라이트 리그 부착

`Play` 버튼으로 시뮬레이션을 시작합니다.

### 2. ROS2 노드 실행

Isaac Sim이 `Play` 상태인 것을 확인한 후 별도 터미널에서 실행합니다.

```bash
source /opt/ros/humble/setup.bash
cd ~/isaacsim/extension_examples/hello_world
python3 welding_scenario_node.py
```

노드가 시작되면 자동으로 SEARCHING 상태로 진입하고 Jackal이 마커를 탐색하기 시작합니다.


**v2**
---

### v2 실행 방법 (underwater_welder) ⭐ 권장

#### 0단계: 도메인 아이디 확인
```bash
echo $ROS_DOMAIN_ID
# 만약 107이 아닐경우
nano ~/.bashrc
```
```bash
# 맨 아래에 추가
export ROS_DOMAIN_ID=107
```
```bash
source ~/.bashrc
```

#### 1단계: 환경 설정
```bash
source /opt/ros/humble/setup.bash
source ~/Rokey6-A2-Isaac-simulation-project
/install/setup.bash
```

#### 2단계: Isaac Sim 시뮬레이터
```bash
~/isaacsim/python.sh \
  ~/Rokey6-A2-Isaac-simulation-project
/src/underwater_welder/scripts/test_tool_target_controller.py
```

#### 3단계: 비전 노드
```bash
# 새 터미널
python3 ~/Rokey6-A2-Isaac-simulation-project
/src/underwater_welder/scripts/1_green_marker_pnp_node.py
```

#### 4단계: 모니터링 (선택)
```bash
# 토픽 확인
ros2 topic list
ros2 topic echo /tool_target_pose

# 이미지 확인
ros2 run rqt_image_view rqt_image_view
```

#### 5단계: ros 통신이 안될때 (선택)
```bash
# 도메인 아이디 확인 이후 isaacsim 내부에서 변경
echo $ROS_DOMAIN_ID

# 아이작 심 실행
~/isaacsim/./isaac-sim.sh
```
```
아래 Content -> My Computer -> / -> home -> Rokey6-A2-Isaac-simulation-project(your folder) -> src -> underwater_welder -> jackal_and_ur10_2.usd
jackal_and_ur10_2.usd 우클릭 후 open

상단 Window -> Graph Editors -> Action Graph 클릭

하단 Action Graph -> Edit Action Graph -> /jackal/ActionGraph -> ROS2 Context 클릭
-> 오른쪽 ROS2Context Node -> Inputs -> domain_id 변경(echo $ROS_DOMAIN_ID 입력해서 뜬 도메인 아이디로)
```
---

### 빠른 시작 (v2 기준)

**한 줄 실행 (터미널 1)**:
```bash
~/isaacsim/python.sh ~/hamtaro/src/underwater_welder/scripts/test_tool_target_controller.py
```

**한 줄 실행 (터미널 2)**:
```bash
python3 ~/hamtaro/src/underwater_welder/scripts/1_green_marker_pnp_node.py
```


---

**Q. 용접봉이 마커에 닿지 않는다**  
`MARKER_FIXED_Y` 값을 0.05 단위로 조금씩 키워보세요.  
로그에서 `[DEBUG] tvec=` 라인의 depth 값과 비교해서 조정하면 빠릅니다.

**Q. IK 실패가 계속 발생한다**  
`ur10_ik.py`의 `_IK_INIT_GUESSES_DEG`에 현재 동작하는 관절각을 초기값으로 추가하세요.  
Isaac Sim에서 관절 슬라이더로 팔을 마커 앞에 수동 배치한 뒤 관절값을 읽어서 넣으면 됩니다.

**Q. 마커가 인식되지 않는다**  
`detect_green_marker`의 HSV 범위 `[40,80,80] ~ [80,255,255]`를 확인하세요.  
OpenCV 창에서 마스킹 결과를 추가 출력하면 디버깅이 편합니다.

**Q. ROS2 토픽이 연결되지 않는다**  
Isaac Sim ROS2 Bridge가 활성화되어 있는지 확인하세요.  
`Window → Extensions`에서 `isaacsim.ros2.bridge`가 Enabled인지 확인합니다.

---

## 🐹 Meet the Members

| 🐱 **냥이** | 🐶 **댕댕** | 🐰 **토끼** | 🦊 **여우** |
| :---: | :---: | :---: | :---: |
| <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Animals/Cat%20Face.png" width="100" /> | <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Animals/Dog%20Face.png" width="100" /> | <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Animals/Rabbit%20Face.png" width="100" /> | <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Animals/Fox.png" width="100" /> |
| **[@seohyeon-netizen](https://github.com/seohyeon-netizen)** | **[@parkyoonheon-debug](https://github.com/parkyoonheon-debug)** | **[@sololic](https://github.com/sololic)** | **[@Kim Hagyun](https://github.com/Hippasus-Y)** |

---
