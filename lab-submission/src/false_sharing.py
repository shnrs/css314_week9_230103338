# src/false_sharing.py
import multiprocessing as mp
import ctypes
import time
import statistics
import os

K = 5_000_000
P_LIST = [1, 2, 4, 8]
REPEATS = 3
STRIDE = 8                      # 8 * int64 = 64 байта = одна кэш-линия
ARR_LEN = 8 * STRIDE            # хватает для padded при P=8 (индекс до 56)


def worker_unpadded(arr, times, barrier, p, iters):
    # все воркеры пишут в одну кэш-линию: индексы 0..7
    barrier.wait()
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        arr[p] += 1
    times[p] = (time.perf_counter_ns() - t0) / 1e9


def worker_padded(arr, times, barrier, p, iters):
    # каждый воркер пишет в свою кэш-линию
    idx = p * STRIDE
    barrier.wait()
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        arr[idx] += 1
    times[p] = (time.perf_counter_ns() - t0) / 1e9


def worker_privatized(arr, times, barrier, p, iters):
    # локальный счётчик, одна запись в общую память в конце
    barrier.wait()
    t0 = time.perf_counter_ns()
    local = 0
    for _ in range(iters):
        local += 1
    arr[p * STRIDE] = local
    times[p] = (time.perf_counter_ns() - t0) / 1e9


def run_once(worker, slot_stride, P):
    # lock=False критичен: иначе Lock сериализует доступ и скрывает false sharing
    arr = mp.Array(ctypes.c_int64, ARR_LEN, lock=False)
    times = mp.Array(ctypes.c_double, P, lock=False)
    barrier = mp.Barrier(P)
    procs = [mp.Process(target=worker, args=(arr, times, barrier, p, K))
             for p in range(P)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
    # время итерации = самый медленный воркер (старт синхронизирован барьером)
    elapsed = max(times[:P])
    total = sum(arr[p * slot_stride] for p in range(P))
    return elapsed, total


def karp_flatt(T1, TP, P):
    # буквальная формула из задания: S = T1/TP
    # (при K итераций НА ВОРКЕР идеал даёт TP ~ T1, то есть e ~ 1)
    e_literal = (P * TP / T1 - 1) / (P - 1)
    # масштабированное ускорение: работа растёт в P раз, S = P*T1/TP, идеал e ~ 0
    S = P * T1 / TP
    e_scaled = (1 / S - 1 / P) / (1 - 1 / P)
    return e_literal, e_scaled


def main():
    variants = [
        ("unpadded", worker_unpadded, 1),
        ("padded", worker_padded, STRIDE),
        ("privatized", worker_privatized, STRIDE),
    ]
    rows = []
    for name, worker, slot in variants:
        T1 = None
        for P in P_LIST:
            runs = []
            for _ in range(REPEATS):
                t, total = run_once(worker, slot, P)
                assert total == P * K, f"{name} P={P}: {total} != {P*K}"
                runs.append(t)
            TP = statistics.median(runs)
            if P == 1:
                T1 = TP
                e_lit = e_sc = float("nan")
            else:
                e_lit, e_sc = karp_flatt(T1, TP, P)
            rows.append((name, P, K, TP, T1 / TP, P * T1 / TP, e_lit, e_sc))
            print(f"{name:11s} P={P} T={TP:.3f}s  T/T1={TP/T1:.2f}  "
                  f"e_scaled={e_sc:.3f}")

    os.makedirs("results", exist_ok=True)
    with open("results/false_sharing_karp_flatt.tsv", "w") as f:
        f.write("variant\tP\tK\tT_s\tS_fixed\tS_scaled\te_literal\te_scaled\n")
        for r in rows:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.6f}\t{r[4]:.4f}\t"
                    f"{r[5]:.4f}\t{r[6]:.4f}\t{r[7]:.4f}\n")


if __name__ == "__main__":
    main()