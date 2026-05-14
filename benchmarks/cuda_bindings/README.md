# cuda.bindings benchmarks

These benchmarks are intended to measure the latency overhead of calling CUDA
Driver APIs through cuda.bindings, relative to a similar C++ baseline.

The goal is to benchmark how much overhead does the Python layer adds to calling
CUDA APIs and what operations are not in our target of less than 1us of overhead.

Most Python benchmarks have a C++ counterpart that is used as a comparative
baseline. We try to make each implementation perform small operations and
nearly the same work as possible and are run under similar conditions.

A few benchmarks (e.g. in `bench_enum.py`) are intentionally Python-only
because they measure costs with no direct C++ equivalent — such as enum
construction and member access on `cuda.bindings` enum classes.

These are **not** throughput benchmarks to measure the overall performance
of kernels and applications.

## Usage

Requires pixi.

There are a couple of environments defined based on how `cuda.bindings` is installed:

- `wheel`: Installs from conda packages
- `source`: Installs from source

There are a couple of tasks defined:

- `bench`: Runs the Python benchmarks
- `bench-cpp`: Runs the C++ benchmarks

### System tuning

For more stable results on Linux, tune the system before running benchmarks.
See: https://pyperf.readthedocs.io/en/latest/system.html#system

```bash
# Show current system state
pixi run -e wheel -- python -m pyperf system show

# Apply tuning (may require root)
$(pixi run -e wheel -- which python) -m pyperf system tune
```

### Running benchmarks

To run the benchmarks combine the environment and task:

```bash
# Run the Python benchmarks in the wheel environment
pixi run -e wheel bench
pixi run -e wheel bench --min-time 0.1

# Run the Python benchmarks in the source environment
pixi run -e source bench

# Run the C++ benchmarks
pixi run -e wheel bench-cpp
pixi run -e wheel bench-cpp --min-time 0.1
```

Both runners automatically save results to JSON files in the benchmarks
directory: `results-python.json` and `results-cpp.json`.

## Output JSON and analysis

The benchmarks are run using [pyperf](https://pyperf.readthedocs.io/en/latest/).
Both Python and C++ results are saved in pyperf-compatible JSON format,
which can be analyzed with pyperf commands:

```bash
# Show results and statistics
pixi run -e wheel -- python -m pyperf stats results-python.json
pixi run -e wheel -- python -m pyperf stats results-cpp.json

# Compare C++ vs Python results
pixi run -e wheel -- python -m pyperf compare_to results-cpp.json results-python.json
```
