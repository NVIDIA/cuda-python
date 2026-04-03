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

# Run the C++ benchmarks (environment is irrelavant here)
pixi run -e wheel bench-cpp
```

## pyperf JSON

The benchmarks are run using [pyperf](https://pyperf.readthedocs.io/en/latest/).
The results are written to a JSON file in the format expected by pyperf.

The C++ benchmarks also generate a valid JSON file, in the same format.

```
pixi run -e wheel bench-cpp -0 cpp.json

pixi run -e wheel pyperf stats cpp.json
```
