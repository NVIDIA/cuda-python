# cuda.core test suite

Package-wide conventions live in `../AGENTS.md`; repository-wide ones in
`../../AGENTS.md`. This file covers conventions specific to the tests.

## Never create an uncapped memory pool

A memory pool created without `max_size` reserves virtual address space similar
in size to the installed physical device memory regardless of what the test
actually allocates. The reservation is charged to the process address space
even though it is not backed by physical memory, and it is not returned until
the pool is destroyed *and* the stream-ordered frees of its outstanding
allocations retire. The whole suite shares one process and one device, so these
reservations accumulate across tests.

When a test needs its own pool, use the suite-wide cap from
`helpers/constants.py`:

```python
from helpers.constants import POOL_SIZE

mr = DeviceMemoryResource(dev, DeviceMemoryResourceOptions(max_size=POOL_SIZE))
```

Use a larger value only if a test genuinely requires it, and prefer adding a
shared constant to `helpers/constants.py` over redefining one per module.

### Passing no options is different from passing empty options

`DeviceMemoryResource(dev)` with no options does **not** create a pool. It
wraps the device's existing default mempool (`_mempool_owned` is false) and
costs no additional address space. Passing *any* options object creates a new
owned pool, and a new pool without `max_size` is uncapped:

```python
DeviceMemoryResource(dev)  # wraps default pool, free
DeviceMemoryResource(dev, DeviceMemoryResourceOptions())  # NEW uncapped pool, expensive
DeviceMemoryResource(dev, {"ipc_enabled": True})  # NEW uncapped pool, expensive
```

Do not add `max_size` to a call that currently passes no options: that
converts a free default-pool wrapper into a new pool and makes things worse.

### Managed pools are exempt

`cuMemPoolCreate` requires `CUmemPoolProps.maxSize` to be zero for managed
pools, so `ManagedMemoryResourceOptions` has no `max_size` option. Managed
pools cannot be right-sized and are not checked.

### Document exemptions

When a call is deliberately exempt -- most often because it sits inside
`pytest.raises` and no pool is ever created -- annotate it:

```python
with pytest.raises(RuntimeError, match="IPC is not available"):
    # uncapped-pool-ok: raises before the pool is created
    DeviceMemoryResource(mempool_device, DeviceMemoryResourceOptions(ipc_enabled=True))
```

## Release resources at test boundaries

The `_init_cuda_context` fixture in `conftest.py` runs `gc.collect()` followed
by `cuCtxSynchronize()` before popping the context. Tests should not rely on
that as a substitute for cleaning up explicitly: prefer context managers for
resources whose lifetime fits a single scope, and keep pool lifetimes inside
the test that creates them.

## Shared test support

See also: https://docs.pytest.org/en/stable/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files

Follow these rules when adding or moving shared test code:

- Never import from a `conftest.py`.
- Put suite-wide fixtures and pytest hooks in `tests/conftest.py`. Put fixtures
  needed only by one test subtree in that subtree's nearest `conftest.py`.
- Put a pytest hook in a nested `conftest.py` only if pytest supports that hook
  there. If the hook receives suite-wide data, explicitly limit its effects to
  the intended subtree.
- Code used only to implement fixtures or hooks may remain in the same
  `conftest.py`. Put functions and constants imported by test modules in
  `tests/helpers/` instead.
- Import helpers explicitly from the test root, for example:
  `from helpers.memory import create_managed_memory_resource_or_skip`.
- Fixtures in a nested `conftest.py` are available to tests in its directory
  and descendants; fixtures from applicable parent `conftest.py` files remain
  available.
- Do not add `__init__.py` solely because a test directory contains a
  `conftest.py`.
- In directories without `__init__.py`, keep test-module basenames unique
  within this test suite.

## Skip only real setup failures

`pytest.skip(reason)` records the test as SKIPPED with `reason` in the
report. A helper that wraps `yield` in `except Exception: pytest.skip(...)`
therefore records every test-body failure as a skip — a real regression, a
`TypeError`, an `AttributeError` all become "SKIPPED: <reason>" instead of
"FAILED", and the suite goes green regardless of whether the code under
test works.

Catch only the specific exception that legitimately means "not available",
and only around the setup call — never around `yield`:

```python
@contextlib.contextmanager
def _gl_context():
    try:
        win, tex_id = _setup_gl_texture()      # setup only
    except (pyglet.NoSuchConfigException, GLContextError) as e:
        pytest.skip(f"GL unavailable: {e}")
    try:
        yield tex_id                          # body exceptions propagate
    finally:
        _cleanup(win, tex_id)
```

When a CUDA call's error means "feature refused by this driver" (e.g.
`CUDA_ERROR_OPERATING_SYSTEM` for CUDA-GL interop on WSL), skip at the call
site with a narrow catch on the specific error, not inside the GL helper —
see `_register_gl_buffer` / `_register_gl_image` in `tests/test_graphics.py`.

## `importorskip` is for optional dependencies only

`pytest.importorskip("X")` is correct when `X` is genuinely optional
(platform-gated binding, parametrized "test each available module"). It is
dead code when `X` is a declared test or runtime dependency: the skip then
fires only when the environment is broken, which is the case you want to fail
loudly, not hide. Use a bare top-level `import` for declared deps.

Before adding `importorskip`, check `cuda_core/pyproject.toml`'s `test`
and `test-cu*` groups and `cuda_core`'s `dependencies`. If the target is
listed, import it directly.

## Capability probes must not swallow real bugs

A probe function that answers "is feature X available?" by catching
`Exception` and returning `False` will report "not available" even when the
probed API failed for a real, unexpected reason — silently enabling a skip
that hides the bug. Catch only the exception that genuinely means "not
available", and split the checks so each catch is narrow:

```python
def _is_nvfatbin_available():
    try:
        from cuda.bindings import nvfatbin
    except ImportError:
        return False
    try:
        nvfatbin.version()
    except nvfatbin.nvFatbinError:
        return False
    return True
```

Do not catch `ImportError` for a hard runtime dependency (e.g.
`cuda.bindings` for `cuda.core`) — that is a broken environment and should
surface at collection time.

## Tests that touch CUDA must establish their own context

The `init_cuda` fixture pops the CUDA context on teardown, so a test
that calls a CUDA API without `init_cuda` (or an explicit
`Device.set_current()`) inherits whatever context the previous test happened
to leave current on the thread — possibly none. With `pytest-randomly` that
makes the pass/fail outcome depend on test order, so it moves seed to seed and
looks like flakiness. Request `init_cuda` for any test that calls into the
driver, or set up and tear down a context yourself.

## Assert on behavior, not implementation

Pin on observable behavior the contract guarantees — return values, raised
exception types, public state transitions. Avoid asserting on internal
call counts, private helper invocation order, or error message substrings
that are not part of the contract. A refactor that preserves behavior but
changes internals should not break the test.
