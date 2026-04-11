"""
FlashInfer-Bench Local Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks locally.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
from scripts.pack_solution import pack_solution


def get_trace_set_path() -> str:
    """Get trace set path from environment variable."""
    path = os.environ.get("FIB_DATASET_PATH")
    if not path:
        raise EnvironmentError(
            "FIB_DATASET_PATH environment variable not set. "
            "Please set it to the path of your flashinfer-trace dataset."
        )
    return path


def run_benchmark(
    solution: Solution, config: BenchmarkConfig = None
) -> tuple[dict, TraceSet]:
    """Run benchmark locally and return summary results with trace set."""
    if config is None:
        # config = BenchmarkConfig(warmup_runs=2, iterations=2, num_trials=2)
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set_path = get_trace_set_path()
    trace_set = TraceSet.from_path(trace_set_path)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)

    try:
        result_trace_set = benchmark.run_all(dump_traces=True)
    finally:
        benchmark.close()

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }

            # 报错的话打印日志
            if entry["status"] != "PASSED":
                print(trace.evaluation.log)

            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = (
                    trace.evaluation.performance.reference_latency_ms
                )
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    return results, result_trace_set


def save_results_json(
    results: dict,
    trace: TraceSet,
    output: str,
    output_dir: Path = PROJECT_ROOT / "output",
) -> tuple[Path, Path]:
    """Save summary log and viewer-friendly JSONL to local files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = "benchmark_results"
    first_trace = next((ts[0] for ts in trace.traces.values() if ts), None)
    if first_trace is not None:
        solution = first_trace.solution
        timestamp = first_trace.evaluation.timestamp
        safe_timestamp = timestamp.replace(":", "-").replace(".", "-")
        stem = f"{solution}_{safe_timestamp}"

    summary_path = output_dir / f"{stem}_summary.log"
    viewer_path = output_dir / f"{stem}_traces.jsonl"

    summary_path.write_text(output)
    viewer_path.write_text(format_trace_jsonl(trace))

    return summary_path, viewer_path


def format_trace_jsonl(trace: TraceSet) -> str:
    """Render traces as JSONL: one trace JSON object per line."""
    lines = []
    for traces in trace.traces.values():
        for item in traces:
            lines.append(
                json.dumps(item.model_dump(mode="json"), separators=(",", ":"))
            )
    return "\n".join(lines) + ("\n" if lines else "")


def print_format_results(results: dict) -> str:
    """Render benchmark results with the same layout as print_results."""
    lines = []
    for def_name, traces in results.items():
        lines.append(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            line = f"  Workload {workload_uuid[:8]}...: {status}"

            if result.get("latency_ms") is not None:
                line += f" | {result['latency_ms']:.3f} ms"

            if result.get("speedup_factor") is not None:
                line += f" | {result['speedup_factor']:.2f}x speedup"

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                line += f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}"

            lines.append(line)
            print(line)  # Print each line immediately to terminal

    return "\n".join(lines).lstrip("\n") + "\n"


def main():
    """Pack solution and run benchmark."""
    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark...")
    results, trace = run_benchmark(solution)

    if not results:
        print("No results returned!")
        return

    output = print_format_results(results)

    # print
    summary_path, viewer_path = save_results_json(results, trace, output)
    print(f"\nSaved summary Logs: {summary_path}")
    print(f"Saved viewer JSONL: {viewer_path}")


if __name__ == "__main__":
    main()
