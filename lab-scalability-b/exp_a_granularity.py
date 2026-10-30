import sys
import time
from multiprocessing import Pool


def collatz_steps(n):
    steps = 0
    while n > 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


def run_granularity(workers, total_items=1_000_000, chunk_size=1):
    numbers = range(1, total_items + 1)
    t0 = time.perf_counter()
    with Pool(processes=workers) as pool:
        results = pool.map(collatz_steps, numbers, chunksize=chunk_size)
    return time.perf_counter() - t0, sum(results)


def karp_flatt(t1, tp, p):
    return ((p * tp / t1) - 1.0) / (p - 1.0)


def main():
    total_items = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    cores = [1, 2, 4, 8]
    reference = None
    for chunk_size in (1, 10_000):
        runtimes = {}
        for p in cores:
            t, s = run_granularity(p, total_items, chunk_size)
            if reference is None:
                reference = s
            assert s == reference, "checksum mismatch"
            runtimes[p] = t
        t1 = runtimes[1]
        print(f"\n--- chunk_size={chunk_size} ---")
        print(f"P= 1 | Time={t1:7.3f}s | Speedup= 1.00x | Efficiency=100.0%")
        for p in cores[1:]:
            tp = runtimes[p]
            sp = t1 / tp
            ep = sp / p * 100
            e = karp_flatt(t1, tp, p)
            print(f"P={p:2d} | Time={tp:7.3f}s | Speedup={sp:5.2f}x | Efficiency={ep:5.1f}% | Karp-Flatt e={e:6.3f}")


if __name__ == "__main__":
    main()