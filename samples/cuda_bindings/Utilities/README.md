# CUDA Bindings Sample Utilities

`cuda_bindings_utils.py` contains shared boilerplate for samples that call the
low-level `cuda.bindings` APIs directly. It provides:

- CUDA error tuple checking and result unwrapping
- NVRTC compilation and CUDA module loading through `KernelHelper`
- Runtime and driver API device selection
- Command-line flag helpers used by CUDA sample ports
- Consistent requirement waivers using exit code 2 standalone, or the distinct
  code negotiated by the automated sample runner

## Using the Helpers

Bindings samples retain the CUDA Samples category directories, so a sample
adds the namespace-local `Utilities` directory to `sys.path` as follows:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Utilities"))
from cuda_bindings_utils import KernelHelper, check_cuda_errors
```

See [`clockNvrtc`](../0_Introduction/clockNvrtc/) for a complete example.
Install dependencies from the individual sample directory before running it:

```bash
cd samples/cuda_bindings/0_Introduction/clockNvrtc
pip install -r requirements.txt
python clockNvrtc.py
```

These helpers support the samples and are not part of the public
`cuda.bindings` API.

`requirement_not_met()` reads `CUDA_PYTHON_SAMPLE_WAIVER_EXIT_CODE` when it is
set by an orchestrator. The repository sample runner sets it to `77`, keeping
intentional waivers distinct from command-line parser errors, which use exit
code `2`. Direct standalone execution retains exit code `2` for compatibility.
