"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
output_volume = modal.Volume.from_name("output", create_if_missing=True)
TRACE_SET_PATH = "/data"
OUTPUT_PATH = "/outputs"

image = (
    modal.Image.from_registry(
        "flashinfer/flashinfer-ci-cu132:20260401-2c675fb", add_python="3.12"
    )
    .apt_install("git", "wget", "build-essential", "cmake")
    .run_commands(
        # Force-reinstall latest main of flashinfer-bench. Override the
        # PyPI 0.1.2 already in the image. Pip pulls/refreshes deps as
        # needed (pydantic, safetensors, tvm-ffi, ...).
        "pip install --force-reinstall --upgrade "
        "git+https://github.com/flashinfer-ai/flashinfer-bench.git@main",
    )
    .pip_install(
        # cupti-python is required for accurate GPU timing (per
        # EVALUATION.md). Without it, flashinfer-bench's timer falls
        # back to CUDA events, which include kernel-launch overhead
        # (~2-4 μs per call) and over-report latency. The official
        # image is *supposed* to ship this but the released 0.1.2
        # dep-chain doesn't pull it in.
        "cupti-python",
    )
    # Caches the CUTLASS clone + torch extension build dir across runs.
    .env({"TORCH_EXTENSIONS_DIR": "/outputs/torch_ext_cache"})
)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=14400,
    volumes={TRACE_SET_PATH: trace_volume, OUTPUT_PATH: output_volume},
)
def run_benchmark(solution: Solution, config: BenchmarkConfig = None) -> dict:
    """Run benchmark on Modal B200 and return results."""
    import subprocess

    # Lock clocks (best effort — fails without priv; logged, continue).
    r = subprocess.run(
        ["nvidia-smi", "-ac", "3996,1965"], capture_output=True, text=True
    )
    print(f"nvidia-smi -ac: rc={r.returncode} {r.stdout.strip()[:120]}")

    if config is None:
        # Match the equivalent CLI behavior:
        # --warmup-runs 10 --iterations 50 --num-trials 3
        # --use-isolated-runner --timeout 300
        # config = BenchmarkConfig(
        #     warmup_runs=10,
        #     iterations=50,
        #     num_trials=3,
        #     use_isolated_runner=False,
        #     timeout_seconds=300,
        # )
        config = BenchmarkConfig(
            warmup_runs=1,
            iterations=1,
            num_trials=1,
            use_isolated_runner=False,
            timeout_seconds=300,
        )

    trace_set = TraceSet.from_path(TRACE_SET_PATH + "/mlsys26-contest")
    # trace_set = TraceSet.from_path(TRACE_SET_PATH)

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
    result_trace_set = benchmark.run_all(dump_traces=True, resume=True)

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

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()


@app.local_entrypoint()
def main():
    """Pack solution and run benchmark on Modal."""
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark on Modal B200...")
    results = run_benchmark.remote(solution)

    if not results:
        print("No results returned!")
        return

    print_results(results)
