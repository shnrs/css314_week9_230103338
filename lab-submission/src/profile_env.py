import os
import sys
import json
import platform
import argparse
from salt import get_student_params

def profile_hardware():
    profile = {
        "os_kernel": f"{platform.system()} {platform.release()}",
        "python_version": sys.version,
        "architecture": platform.machine(),
        "cpu_model": platform.processor() or "Generic x86_64 CPU",
        "physical_cores": os.cpu_count() or 4,
        "logical_cores": os.cpu_count() or 4,
        "cache_line_size_bytes": 64,
        "l1_data_cache_kb": 32,
        "l2_cache_kb": 512,
        "l3_cache_kb": 16384
    }
    
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                lines = f.readlines()
                for line in lines:
                    if "model name" in line:
                        profile["cpu_model"] = line.split(":")[1].strip()
                        break
        except Exception:
            pass

    return profile

def main():
    parser = argparse.ArgumentParser(description="Host Topology Profiling and Parameter Salting")
    parser.add_argument("--student-id", type=str, default="12345678", help="8-digit Student ID")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # 1. Profile Environment
    host_profile = profile_hardware()
    with open("results/host_profile.json", "w") as f:
        json.dump(host_profile, f, indent=4)

    # 2. Compute Salted Parameters
    salted_params = get_student_params(args.student_id)
    with open("results/salted_params.json", "w") as f:
        json.dump(salted_params, f, indent=4)

    print(f"[+] Environment profiled. Salted N={salted_params['N']}. Config saved to results/")

if __name__ == "__main__":
    main()