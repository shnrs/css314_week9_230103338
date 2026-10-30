# src/scaling_study.py
import argparse
import json
import math
import os
import statistics
import sys
import time
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from salt import get_student_params

ITERS = 200       # число шагов стенсила
REPEATS = 3       # медиана из 3 запусков
FLOPS_PER_POINT = 5   # 4 сложения + 1 умножение


# ---------- ядро и воркер ----------

def stencil_rows(src, dst, r0, r1):
    dst[r0:r1, 1:-1] = 0.25 * (src[r0 - 1:r1 - 1, 1:-1] + src[r0 + 1:r1 + 1, 1:-1]
                               + src[r0:r1, :-2] + src[r0:r1, 2:])


def worker(name_a, name_b, N, r0, r1, iters, barrier, times, p):
    shm_a = shared_memory.SharedMemory(name=name_a)
    shm_b = shared_memory.SharedMemory(name=name_b)
    A = np.ndarray((N, N), dtype=np.float64, buffer=shm_a.buf)
    B = np.ndarray((N, N), dtype=np.float64, buffer=shm_b.buf)
    barrier.wait()
    t0 = time.perf_counter_ns()
    for it in range(iters):
        src, dst = (A, B) if it % 2 == 0 else (B, A)
        stencil_rows(src, dst, r0, r1)
        barrier.wait()          # все закончили шаг, можно менять буферы
    times[p] = (time.perf_counter_ns() - t0) / 1e9
    del A, B, src, dst
    shm_a.close()
    shm_b.close()


def init_grid(N, seed):
    return np.random.default_rng(seed).random((N, N))


def run_parallel(N, P, seed, iters):
    nbytes = N * N * 8
    shm_a = shared_memory.SharedMemory(create=True, size=nbytes)
    shm_b = shared_memory.SharedMemory(create=True, size=nbytes)
    try:
        A = np.ndarray((N, N), dtype=np.float64, buffer=shm_a.buf)
        B = np.ndarray((N, N), dtype=np.float64, buffer=shm_b.buf)
        g = init_grid(N, seed)
        A[:] = g
        B[:] = g                 # границы одинаковы в обоих буферах
        times = mp.Array("d", P, lock=False)
        barrier = mp.Barrier(P)
        rows = N - 2
        base, rem = divmod(rows, P)
        procs = []
        for p in range(P):
            r0 = 1 + p * base + min(p, rem)
            r1 = r0 + base + (1 if p < rem else 0)
            procs.append(mp.Process(target=worker, args=(
                shm_a.name, shm_b.name, N, r0, r1, iters, barrier, times, p)))
        for pr in procs:
            pr.start()
        for pr in procs:
            pr.join()
        res = A if iters % 2 == 0 else B
        checksum = float(res.sum())
        elapsed = max(times[:P])
        del A, B, res
        return elapsed, checksum
    finally:
        shm_a.close()
        shm_b.close()
        shm_a.unlink()
        shm_b.unlink()


def reference_checksum(N, seed, iters):
    A = init_grid(N, seed)
    B = A.copy()
    for it in range(iters):
        src, dst = (A, B) if it % 2 == 0 else (B, A)
        stencil_rows(src, dst, 1, N - 1)
    res = A if iters % 2 == 0 else B
    return float(res.sum())


def measure(N, P, seed, iters, ref):
    runs = []
    for _ in range(REPEATS):
        t, cs = run_parallel(N, P, seed, iters)
        if cs != ref:
            raise RuntimeError(f"checksum mismatch N={N} P={P}: {cs} != {ref}")
        runs.append(t)
    return statistics.median(runs)


# ---------- вспомогательное ----------

def physical_cores():
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except ImportError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def p_list(max_p):
    ps, p = [], 1
    while p <= max_p:
        ps.append(p)
        p *= 2
    if ps[-1] != max_p:
        ps.append(max_p)
    return ps


def measure_bandwidth_gbs():
    n = 2 ** 24                   # 134 МБ на массив, больше любого L3
    a = np.ones(n)
    b = np.empty_like(a)
    ts = []
    for _ in range(7):
        t0 = time.perf_counter_ns()
        np.copyto(b, a)
        ts.append(time.perf_counter_ns() - t0)
    return (2 * n * 8) / statistics.median(ts)   # байт/нс = ГБ/с


def amdahl_fit(Ps, Ss):
    # 1/S - 1/P = s * (1 - 1/P), МНК через начало координат
    num = den = 0.0
    for P, S in zip(Ps, Ss):
        if P == 1:
            continue
        x = 1 - 1 / P
        y = 1 / S - 1 / P
        num += x * y
        den += x * x
    s = num / den if den else float("nan")
    return max(s, 0.0)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-id", default="230103338")
    ap.add_argument("--max-p", type=int, default=None,
                    help="максимум процессов (по умолчанию физические ядра)")
    ap.add_argument("--peak-bw", type=float, default=None,
                    help="пиковая пропускная способность памяти, ГБ/с (по умолчанию измеряется)")
    ap.add_argument("--peak-gflops", type=float, default=None,
                    help="пиковая производительность, GFLOP/s (по умолчанию оценка)")
    ap.add_argument("--ghz", type=float, default=3.0,
                    help="частота для оценки пика, ГГц")
    args = ap.parse_args()

    params = get_student_params(args.student_id)
    N, seed = params["N"], params["seed"]
    max_p = args.max_p or physical_cores()
    Ps = p_list(max_p)
    print(f"N={N} iters={ITERS} P={Ps}")

    rows = []

    # ----- strong scaling -----
    ref = reference_checksum(N, seed, ITERS)
    T1 = None
    strong_S = []
    for P in Ps:
        T = measure(N, P, seed, ITERS, ref)
        if P == 1:
            T1 = T
        S = T1 / T
        E = S / P * 100
        e = (1 / S - 1 / P) / (1 - 1 / P) if P > 1 else float("nan")
        strong_S.append(S)
        rows.append(("strong", P, N, T, S, E, e, float("nan")))
        print(f"strong P={P}: T={T:.3f}s S={S:.2f} E={E:.0f}%")

    s = amdahl_fit(Ps, strong_S)
    s_max = 1 / s if s > 0 else float("inf")
    print(f"Amdahl: s={s:.4f}  S_max={s_max:.2f}")

    # ----- weak scaling -----
    W1 = (N - 2) ** 2 * ITERS
    T1w = None
    for P in Ps:
        NP = math.ceil(N * math.sqrt(P))
        refw = reference_checksum(NP, seed, ITERS)
        T = measure(NP, P, seed, ITERS, refw)
        if P == 1:
            T1w = T
        WP = (NP - 2) ** 2 * ITERS
        S_scaled = (WP * T1w) / (W1 * T)       # отношение пропускных способностей
        E_w = S_scaled / P * 100
        rows.append(("weak", P, NP, T, S_scaled, E_w, float("nan"), T1w / T))
        print(f"weak   P={P}: N={NP} T={T:.3f}s S_scaled={S_scaled:.2f} E={E_w:.0f}%")

    os.makedirs("results", exist_ok=True)
    with open("results/scaling_metrics.tsv", "w") as f:
        f.write("mode\tP\tN\tT_s\tspeedup\tefficiency_pct\tkarp_flatt_e\tT1_over_TP\n")
        for r in rows:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.6f}\t{r[4]:.4f}\t"
                    f"{r[5]:.2f}\t{r[6]:.4f}\t{r[7]:.4f}\n")

    # ----- Roofline -----
    ai_ideal = FLOPS_PER_POINT / 16.0       # чтение 8 Б + запись 8 Б на точку
    ai_wa = FLOPS_PER_POINT / 24.0          # с учётом write-allocate (+8 Б)
    bw = args.peak_bw or measure_bandwidth_gbs()
    peak_gflops = args.peak_gflops or (16 * max_p * args.ghz)   # AVX2 FMA double
    ridge = peak_gflops / bw
    strong_Tmax = [r for r in rows if r[0] == "strong"][-1][3]
    achieved = FLOPS_PER_POINT * (N - 2) ** 2 * ITERS / strong_Tmax / 1e9
    roof = min(peak_gflops, bw * ai_ideal)
    regime = "memory-bandwidth bound" if ai_ideal < ridge else "compute bound"

    summary = {
        "student_id": args.student_id,
        "N": N,
        "kernel": "5-point Jacobi stencil, float64",
        "flops_per_point": FLOPS_PER_POINT,
        "bytes_per_point_ideal": 16,
        "bytes_per_point_with_write_allocate": 24,
        "arithmetic_intensity_flop_per_byte": ai_ideal,
        "arithmetic_intensity_with_write_allocate": ai_wa,
        "working_set_MB": 2 * N * N * 8 / 1e6,
        "peak_bandwidth_GBs": bw,
        "peak_bandwidth_source": "user" if args.peak_bw else "measured numpy copy",
        "peak_gflops": peak_gflops,
        "peak_gflops_source": "user" if args.peak_gflops else
                              f"estimate 16 FLOP/cycle x {max_p} cores x {args.ghz} GHz",
        "ridge_point_flop_per_byte": ridge,
        "roofline_bound_GFLOPs": roof,
        "achieved_GFLOPs_at_max_P": achieved,
        "regime": regime,
        "amdahl_serial_fraction_s": s,
        "amdahl_max_speedup": s_max,
        "max_P": max_p,
    }
    with open("results/roofline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"AI={ai_ideal:.3f} FLOP/B, ridge={ridge:.2f}, regime: {regime}")


if __name__ == "__main__":
    main()