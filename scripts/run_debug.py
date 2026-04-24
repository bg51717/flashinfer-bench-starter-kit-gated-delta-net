import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DATASET_ROOT = os.environ["FIB_DATASET_PATH"]
WORKLOAD_ID = 50  # max 0-22 for DSA Attention

from flashinfer_bench import Solution, TraceSet
from flashinfer_bench.agents import (
    flashinfer_bench_run_sanitizer,
    flashinfer_bench_list_ncu_options,
    flashinfer_bench_run_ncu,
)
from scripts.pack_solution import pack_solution


def sanitizer(solution: Solution, workload) -> str:
    """Run sanitizer on the solution and workload."""
    print("\nRunning sanitizers...")
    result = flashinfer_bench_run_sanitizer(
        solution=solution,
        workload=workload,
        trace_set_path=DATASET_ROOT,
        sanitizer_types=["memcheck", "racecheck", "synccheck", "initcheck"],
        timeout=300,
    )
    print(result)
    return result


def ncu_profile(solution: Solution, workload) -> str:
    """Run ncu profiler on the solution and workload."""
    print("\nRunning NCU profile...")
    result = flashinfer_bench_run_ncu(
        solution=solution,
        workload=workload,
        trace_set_path=DATASET_ROOT,
        set="detailed",
        page="details",
        timeout=120,
    )
    print(result)
    return result


def main():
    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    trace_set = TraceSet.from_path(DATASET_ROOT)  # 拿到数据集根目录
    # 取该 definition 的第一个 workload
    workloads = trace_set.workloads.get(solution.definition, [])
    print(f"Found {len(workloads)} workloads for definition '{solution.definition}'")
    workload = workloads[WORKLOAD_ID].workload
    # print(
    #     f"Selected workload num_tokens={workload.axes['num_tokens']}, num_pages={workload.axes['num_pages']}"
    # )

    # sanitizer
    # san_out = sanitizer(solution, workload)

    # ncu options
    # print(flashinfer_bench_list_ncu_options())

    # 5) ncu
    ncu_out = ncu_profile(solution, workload)


if __name__ == "__main__":
    main()
