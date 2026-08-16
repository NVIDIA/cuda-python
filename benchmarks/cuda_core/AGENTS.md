# cuda.core benchmarks

Read the README.md in this directory for more details about the benchmarks.

When generating code verify that that the code is correct based on the source for cuda-core
that can be found in ../../cuda_core.

This suite shares the pyperf runner with `../cuda_bindings/` via a sys.path
insert in `run_pyperf.py`. The per-suite setup (`runtime.py`, the `benchmarks/`
module files) lives here. Benchmark IDs are kept identical to the cuda.bindings
suite so `compare.py` can diff them directly.
