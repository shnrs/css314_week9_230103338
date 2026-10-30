# src/dense_kernel.py
import argparse
import hashlib
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from salt import get_student_params

RUNS = 3


# ---------- ядра (плоские 1D-буферы, строка длиной N) ----------

def mm_ijk(A, B, C, N, M):
    # наивный порядок: B[k*N + j] на внутреннем шаге прыгает на N элементов
    for i in range(M):
        for j in range(M):
            for k in range(M):
                C[i * N + j] += A[i * N + k] * B[k * N + j]


def mm_ikj(A, B, C, N, M):
    # A[i,k] вынесен в локальную переменную, внутри stride-1 по B и C
    for i in range(M):
        for k in range(M):
            a = A[i * N + k]
            for j in range(M):
                C[i * N + j] += a * B[k * N + j]


# Numba, если установлена (pip install numba): ускорение в сотни раз.
# Без fastmath, так что порядок суммирования по k одинаковый, и результаты совпадают бит в бит.
try:
    from numba import njit
    mm_ijk = njit(cache=True)(mm_ijk)
    mm_ikj = njit(cache=True)(mm_ikj)
    BACKEND = "numba"
except ImportError:
    BACKEND = "pure_python"


def bench(kernel, A, B, N, M, as_list):
    times = []
    C = None
    for _ in range(RUNS):
        C = [0.0] * (N * N) if as_list else np.zeros(N * N)
        t0 = time.perf_counter_ns()
        kernel(A, B, C, N, M)
        times.append(time.perf_counter_ns() - t0)
    return statistics.median(times), np.asarray(C, dtype=np.float64)


def digest(C, N, M):
    block = np.round(C.reshape(N, N)[:M, :M], 6)
    return hashlib.sha256(block.tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-id", default="230103338")
    ap.add_argument("--M", type=int, default=None,
                    help="переопределить размер подблока (по умолчанию N/2)")
    args = ap.parse_args()

    params = get_student_params(args.student_id)
    N = params["N"]
    M = args.M or N // 2
    rng = np.random.default_rng(params["seed"])
    A_np = rng.random(N * N)
    B_np = rng.random(N * N)

    as_list = BACKEND == "pure_python"
    A = A_np.tolist() if as_list else A_np
    B = B_np.tolist() if as_list else B_np

    if BACKEND == "numba":  # прогрев JIT, чтобы компиляция не попала в замер
        w = np.zeros(N * N)
        mm_ijk(A, B, w, N, 8)
        mm_ikj(A, B, w, N, 8)

    print(f"backend={BACKEND} N={N} M={M} runs={RUNS}")
    t_naive, C1 = bench(mm_ijk, A, B, N, M, as_list)
    print(f"i-j-k: {t_naive / 1e9:.3f} s")
    t_inter, C2 = bench(mm_ikj, A, B, N, M, as_list)
    print(f"i-k-j: {t_inter / 1e9:.3f} s")

    blk1 = C1.reshape(N, N)[:M, :M]
    blk2 = C2.reshape(N, N)[:M, :M]
    max_diff = float(np.max(np.abs(blk1 - blk2)))
    speedup = t_naive / t_inter
    print(f"speedup = {speedup:.2f}x, max|diff| = {max_diff:.2e}")
    assert max_diff <= 1e-12, "результаты не совпадают"

    os.makedirs("results", exist_ok=True)
    out = {
        "student_id": args.student_id,
        "backend": BACKEND,
        "N": N,
        "M": M,
        "runs": RUNS,
        "T_naive_ijk_ns": t_naive,
        "T_interchanged_ikj_ns": t_inter,
        "speedup": speedup,
        "max_abs_diff": max_diff,
        "sha256_ijk": digest(C1, N, M),
        "sha256_ikj": digest(C2, N, M),
    }
    with open("results/loop_interchange.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()