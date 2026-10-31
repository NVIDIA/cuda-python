# cuda.pathfinder benchmarks

These benchmarks measure filesystem-discovery latency in `cuda.pathfinder`.
They exercise the in-tree source package and do not require a GPU or CUDA
Toolkit.

The suite uses temporary directory trees prepared before timing starts. It has
two levels of coverage:

- `find_sub_dirs.*` isolates the underlying filesystem mechanism with realistic
  one-root and three-root cases, plus wildcard and cache diagnostics.
- `public_discovery.*` measures cold-cache calls through the public header,
  binary, and static-library APIs. Fixtures use real NVIDIA wheel layouts such
  as `nvidia/cuda_runtime/include` and `nvidia/cuda_nvcc/bin`.

The public cold-cache benchmarks clear pathfinder's process-lifetime caches on
each iteration. This intentionally models first use in a fresh process; the
small cache-clear cost is included in the reported time. Three roots represent
a virtual environment, user site-packages, and system site-packages, with the
target in the last root to measure the complete search order.

## Usage

Requires pixi. The `source` environment installs `cuda-pathfinder` from this
checkout and works on Linux and Windows.

```bash
# List benchmark IDs.
pixi run -e source bench --list

# Quick functional validation; timings are not meaningful.
pixi run -e source bench-smoke-test

# Run the full suite and write results-python.json.
pixi run -e source bench

# Reduce runtime while iterating.
pixi run -e source bench --min-time 0.1
```

## Comparing changes

Save results from the base revision and the modified checkout under distinct
names, then use pyperf's statistical comparison:

```bash
pixi run -e source bench -o results-before.json
pixi run -e source bench -o results-after.json
pixi run -e source -- python -m pyperf compare_to \
    results-before.json results-after.json --table
```

For stable results, minimize other system activity and use the same machine,
Python version, and Pixi environment for both runs. See pyperf's system tuning
guidance: https://pyperf.readthedocs.io/en/latest/system.html#system
