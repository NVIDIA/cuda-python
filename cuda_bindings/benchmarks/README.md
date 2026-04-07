# cuda.bindings benchmarks

These benchmarks are intended to measure the latency overhead of calling CUDA
Driver APIs through cuda.bindings, relative to a similar C++ baseline.

The goal is to benchmark how much overhead does the Python layer adds to calling
CUDA APIs and what operations are not in our target of less than 1us of overhead.

Each Python benchmark has a C++ counterpart, which is used to compare the
operations. We try to make each implementation perform small operations
and nearly the same work as possible and are run under similar conditions.

These are **note** throughput benchmarks to measure the overall performance
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
sudo $(pixi run -e wheel -- which python) -m pyperf system tune
```

### Running benchmarks

To run the benchmarks combine the environment and task:

```bash
# Run the Python benchmarks in the wheel environment
pixi run -e wheel bench

# Run the Python benchmarks in the source environment
pixi run -e source bench

# Run the C++ benchmarks
pixi run -e wheel bench-cpp
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
