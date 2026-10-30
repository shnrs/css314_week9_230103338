import os
import time
import json
import numpy as np

def run_stride_benchmark():
    buffer_elements = 16_777_216
    if os.path.exists("results/salted_params.json"):
        with open("results/salted_params.json", "r") as f:
            params = json.load(f)
            buffer_elements = params.get("buffer_elements", buffer_elements)

    print(f"Allocating 1D contiguous array with {buffer_elements} float64 elements (~134 MB)...")
    data = np.ones(buffer_elements, dtype=np.float64)

    strides = [1, 2, 4, 8, 16, 32, 64, 128]
    tsv_lines = ["stride\ttouched_elements\ttime_ns\tlatency_per_element_ns"]

    for s in strides:
        for _ in range(5):
            _ = np.sum(data[::s])

        runtimes = []
        for _ in range(5):
            t0 = time.perf_counter_ns()
            total = np.sum(data[::s])
            t1 = time.perf_counter_ns()
            runtimes.append(t1 - t0)

        median_time_ns = int(np.median(runtimes))
        elements_touched = len(data[::s])
        latency_ns = median_time_ns / elements_touched

        tsv_lines.append(f"{s}\t{elements_touched}\t{median_time_ns}\t{latency_ns:.4f}")
        print(f"Stride {s:3d}: Latency = {latency_ns:.4f} ns/element (Total NS: {median_time_ns})")

    os.makedirs("results", exist_ok=True)
    with open("results/stride_results.tsv", "w") as f:
        f.write("\n".join(tsv_lines) + "\n")
    print("[+] Saved stride results to results/stride_results.tsv")

if __name__ == "__main__":
    run_stride_benchmark()