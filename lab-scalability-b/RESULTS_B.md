# RESULTS_B

Student ID: 230103338

| Configuration | T1 (s) | T2 (s) | T4 (s) | T8 (s) | S8 | e(2) | e(8) | Verdict |
|---|---|---|---|---|---|---|---|---|
| A: Micro (k=1) | 5.050 | 6.210 | 7.080 | 8.401 | 0.60 | 1.459 | 1.758 | Master dispatch serialization |
| B: Queue Choke | 5.034 | 52.319 | 48.310 | 53.236 | 0.095 | 19.787 | 11.943 | IPC lock contention |
| C: Buffered Batch | 1.175 | 0.623 | 0.566 | 1.162 | 1.01 | 0.060 | 0.988 | Amortized synchronization (limited by CPU count at P=8) |

## Notes

- Experiment A was run with total_items = 100,000 (not 1,000,000) to save time.
- Machine has 4 logical CPUs (os.cpu_count() = 4); P=8 oversubscribes the CPU, so P=8 results include scheduling contention and Windows process spawn overhead.
- Experiment B: time saturates at ~50 s for P >= 2, so e(p) is very high (~12-20) but not monotonically rising. T8 > T1, so negative speedup is confirmed.
- Experiment C processes 25x more items than B (10,000,000 vs 400,000), yet P=1 runs about 4x faster, so throughput increased by roughly 100x. Speedup is good at P=2 (1.89x) and limited at P=4 and P=8 by the number of CPUs.
