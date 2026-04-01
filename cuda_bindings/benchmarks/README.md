# cuda.bindings Benchmarks

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
