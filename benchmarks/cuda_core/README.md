# cuda.core benchmarks

These benchmarks measure the latency overhead of the `cuda.core` public API
on top of `cuda.bindings`. Every benchmark ID here has a 1:1 counterpart in
`../cuda_bindings/benchmarks/` so a `compare.py` run produces a side-by-side
"bindings vs core" overhead table for every operation.

This suite is **not** a throughput benchmark and does not test kernel
performance — it measures Python-side call overhead only. No C++ baseline
is built or run for `cuda.core`: the comparative baseline is the
`cuda.bindings` Python results file at
`../cuda_bindings/results-python.json`.

The pyperf runner (`runner/main.py`) is shared with the cuda.bindings
suite via a `sys.path` insert in `run_pyperf.py`; only the per-suite
`runtime.py` and `benchmarks/*.py` live here.

## Usage

Requires pixi.

Environments:

- `wheel`: Installs released `cuda-core` from conda-forge.
- `source`: Installs `cuda-core` and `cuda-bindings` from the in-tree
  sources, so local changes are exercised.

Tasks:

- `bench`: Runs the full suite.
- `bench-smoke-test`: Runs each bench with `--debug-single-value` for
  a quick smoke check (not meaningful for timing).
- `bench-compare`: Prints a side-by-side table against
  `../cuda_bindings/results-python.json`.

### System tuning

For more stable results on Linux, tune the system before running.
See: https://pyperf.readthedocs.io/en/latest/system.html#system

```bash
pixi run -e wheel -- python -m pyperf system show
$(pixi run -e wheel -- which python) -m pyperf system tune
```

### Running benchmarks

```bash
# Wheel env
pixi run -e wheel bench
pixi run -e wheel bench --min-time 0.1

# Source env (picks up local cuda.core / cuda.bindings changes)
pixi run -e source bench

# Side-by-side comparison vs cuda.bindings
pixi run -e wheel bench-compare
```

Results are saved to `results-python.json` in this directory. Compare
against the cuda.bindings baseline by running that suite's `bench` task
first so `../cuda_bindings/results-python.json` exists.

## Output JSON and analysis

The suite uses [pyperf](https://pyperf.readthedocs.io/en/latest/). The
output JSON is pyperf-compatible:

```bash
pixi run -e wheel -- python -m pyperf stats results-python.json
pixi run -e wheel -- python -m pyperf compare_to \
    ../cuda_bindings/results-python.json results-python.json
```
