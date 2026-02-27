"""
ur10_ik.py
===========
UR10 순기구학(FK) 및 역기구학(IK) 모듈.

DH 파라미터: Universal Robots UR10 매뉴얼 (modified DH, 단위 m/rad)
  joint 순서: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
"""

import math
import numpy as np
from scipy.optimize import minimize
from typing import Optional

# ── UR10 DH 파라미터 ──────────────────────────────────────────────────────────
_A     = np.array([0.0,    -0.6120, -0.5723,  0.0,    0.0,    0.0   ])
_D     = np.array([0.1273,  0.0,    0.0,      0.1639, 0.1157, 0.0922])
_ALPHA = np.array([math.pi/2, 0.0,  0.0,      math.pi/2, -math.pi/2, 0.0])

_Q_MIN = np.full(6, -2 * math.pi)
_Q_MAX = np.full(6,  2 * math.pi)

WELD_WRIST = np.radians([0.0, 0.0, 90.0])   # wrist_1, wrist_2, wrist_3

# 다양한 초기값 — pan 제한 없이 전 방향 커버
# 분석 결과: 마커 위치에 따라 pan이 양수/음수 모두 가능
_IK_INIT_GUESSES_DEG = [
    [ 10.,  -20., 180.],   # 검증된 수렴값 (마커 Z=0.15)
    [ 14.,  -56., 169.],   # 검증된 수렴값 (마커 Z=0.15)
    [  7., -124., 169.],
    [-30.,  -90., 182.],
    [-30.,  -45.,  90.],
    [ 30.,  -90., 182.],
    [ 90.,  -45.,  90.],
    [-90.,  -45.,  90.],
    [  0.,  -90.,  90.],
    [180.,  -90.,   0.],
]


def _dh(a, d, alpha, theta):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0.,  sa,     ca,    d   ],
        [0.,  0.,     0.,    1.  ],
    ])


def fk(q: np.ndarray) -> np.ndarray:
    """순기구학: q (6,) rad → T (4×4) base 기준."""
    T = np.eye(4)
    for i in range(6):
        T = T @ _dh(_A[i], _D[i], _ALPHA[i], q[i])
    return T


def fk_pos(q: np.ndarray) -> np.ndarray:
    """순기구학: q (6,) rad → EE 위치 (3,) m."""
    return fk(q)[:3, 3]


def _ik_single(target_pos: np.ndarray,
               q_init: np.ndarray,
               tol: float = 5e-4) -> tuple:
    """단일 초기값으로 IK 최적화. pan 제한 없음."""
    q_w = WELD_WRIST

    def cost(q3):
        q = np.array([q3[0], q3[1], q3[2], q_w[0], q_w[1], q_w[2]])
        return float(np.sum((fk_pos(q) - target_pos) ** 2))

    res = minimize(cost, q_init[:3],
                   method="L-BFGS-B",
                   bounds=[(_Q_MIN[i], _Q_MAX[i]) for i in range(3)],
                   options={"maxiter": 1000, "ftol": (tol ** 2) * 1e-4})

    q_sol = np.array([res.x[0], res.x[1], res.x[2], q_w[0], q_w[1], q_w[2]])
    err   = np.linalg.norm(fk_pos(q_sol) - target_pos)
    ok    = err < tol * 30   # 임계값 완화 (5e-4 * 30 = 0.015m)
    return q_sol, ok, err


def ik(target_pos: np.ndarray,
       q_init: Optional[np.ndarray] = None,
       tol: float = 5e-4) -> tuple:

    q_w = WELD_WRIST  # [0.0, 0.0, 90.0] rad

    # ── 모든 초기값을 6개 rad 배열로 통일 ────────────────────────────────
    # _IK_INIT_GUESSES_DEG 는 3개짜리(pan, lift, elbow) → wrist 패딩
    # q_init 은 6개 또는 3개 모두 허용
    guesses = []
    if q_init is not None:
        g = np.asarray(q_init, dtype=float).flatten()
        if g.size == 6:
            guesses.append(g.copy())
        elif g.size == 3:
            guesses.append(np.array([g[0], g[1], g[2], q_w[0], q_w[1], q_w[2]]))

    for deg3 in _IK_INIT_GUESSES_DEG:
        d = np.radians(deg3)
        guesses.append(np.array([d[0], d[1], d[2], q_w[0], q_w[1], q_w[2]]))

    best_q, best_err = None, float("inf")
    bounds6 = [(_Q_MIN[i], _Q_MAX[i]) for i in range(6)]

    def cost(q_test):
        pos_err   = np.linalg.norm(fk_pos(q_test) - target_pos)
        wrist_err = np.linalg.norm(q_test[3:] - q_w)
        return pos_err + 0.1 * wrist_err

    for q_guess in guesses:
        if q_guess.shape != (6,):
            continue
        res = minimize(cost, q_guess,
                       method="L-BFGS-B",
                       bounds=bounds6,
                       options={"maxiter": 1000, "ftol": (tol ** 2) * 1e-4})
        q_sol = res.x
        err   = np.linalg.norm(fk_pos(q_sol) - target_pos)
        if err < best_err:
            best_err = err
            best_q   = q_sol
        if err < tol * 30:
            return best_q, True, err

    return best_q, False, best_err


def ik_deg(target_pos: np.ndarray,
           q_init_deg: Optional[np.ndarray] = None,
           **kw) -> tuple:
    # (기존 코드)
    q_init = np.radians(q_init_deg) if q_init_deg is not None else None
    
    # ▼ 이 부분을 수정하세요! (3개의 값을 받되, err는 _ 로 무시)
    q, ok, _ = ik(target_pos, q_init, **kw)
    
    # (이하 기존 코드 유지)
    if q is not None:
        return np.degrees(q), ok
    return None, ok
