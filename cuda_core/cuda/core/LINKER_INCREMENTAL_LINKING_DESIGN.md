# Incremental Linking in `cuda.core.Linker`

Status: Proposed

Issue: [NVIDIA/cuda-python#2369](https://github.com/NVIDIA/cuda-python/issues/2369)

Minimum CUDA feature version: nvJitLink 13.2

## Decision Summary

Add `relocatable: bool | None = None` to `LinkerOptions`. When true, the nvJitLink
backend passes `-r`, allowing `nvJitLinkComplete` to produce a relocatable, or
incremental, link result.

The result type is determined by the nvJitLink getter used, not by whether the ELF is
final or relocatable:

- `Linker.link("cubin")` returns `ObjectCode(code_type="cubin")`. With `-r`, the bytes
  are normally a CUDA ELF with ELF type `ET_REL`; without `-r`, they are normally
  `ET_EXEC`.
- `Linker.link("ltoir")` returns `ObjectCode(code_type="ltoir")`. It is available when
  `link_time_optimization=True`, the linked-LTOIR getter is available, and there are
  no direct PTX or cubin inputs. nvJitLink silently omits inputs that carry no LTOIR
  from this output.
- `Linker.link("object")` remains invalid. In nvJitLink, `NVJITLINK_INPUT_OBJECT` means
  a *host object*, not every object whose ELF type is `ET_REL`.
- `Linker.link("ptx")` is rejected in relocatable mode. The `-r -lto -ptx`
  combination failed with `NVJITLINK_ERROR_INCORRECT_INPUT_TYPE` in the CUDA versions
  tested.

No new `ObjectCodeFormatType` is needed. A future metadata property such as
`is_relocatable` could describe link completeness, but it should not be encoded as a
new transport format.

For incremental LTO chains, callers should request `"ltoir"` at intermediate stages
and `"cubin"` at the final stage. Requesting a cubin at an intermediate LTO stage is
valid, but it lowers that stage to machine code and prevents later whole-chain LTO
across that boundary.

## Motivation

nvJitLink 13.2 added `-r`, described as performing a relocatable or incremental link
and producing another relocatable object. It permits a group of device-code inputs to
be combined while unresolved device references remain for a later link.

A usage survey found two concrete patterns:

1. Validation-only users invoke `-r` so otherwise-valid inputs with unresolved externs
   can be checked without constructing a final executable. A public example is
   [nvptx-tools][nvptx-tools-incremental-link],
   which uses the link result only as a validation outcome.
2. Multi-stage linkers aggregate several LTOIR units into one LTOIR unit, then pass
   that aggregate into a later link. This requires an LTOIR output path; exposing only
   the partial cubin would introduce an unwanted early machine-code boundary.

The second pattern makes linked-LTOIR retrieval part of the incremental-linking
design, even though the getter was added one CUDA minor release after `-r`.

## Terminology

The words "object", "relocatable", and "cubin" describe different dimensions and
must not be treated as synonyms.

- **Cubin** is the nvJitLink CUDA-binary input/output category. A cubin is an ELF
  device image and can be either final or relocatable.
- **`ET_REL`** is the ELF header value for a relocatable file. The `-r` cubins observed
  in testing use this value.
- **`ET_EXEC`** is the ELF header value normally observed for a final cubin.
- **Host object** is a host `.o` file, potentially containing embedded device images.
  This is what `NVJITLINK_INPUT_OBJECT` and cuda.core's `code_type="object"` mean.
- **LTOIR container** is the format returned by `nvJitLinkGetLinkedLTOIR`. CUDA
  explicitly documents it as a container, not raw LLVM bitcode. It is accepted by a
  later nvJitLink operation as `NVJITLINK_INPUT_LTOIR`.

`ProgramOptions.relocatable_device_code` and the proposed
`LinkerOptions.relocatable` are also distinct:

- The program option makes a compilation unit suitable for device linking.
- The linker option permits a link result to remain relocatable and to retain
  unresolved device references.

## Current Behavior

`Linker.link()` currently accepts `"cubin"` and `"ptx"`. Input `ObjectCode` instances
are mapped as follows in `_linker.pyx`:

| `ObjectCode.code_type` | nvJitLink input type |
|---|---|
| `"cubin"` | `NVJITLINK_INPUT_CUBIN` |
| `"ltoir"` | `NVJITLINK_INPUT_LTOIR` |
| `"object"` | `NVJITLINK_INPUT_OBJECT` |

The last mapping is why a partial cubin must not be labeled `"object"`. In an
experiment, adding the `ET_REL` cubin back as `NVJITLINK_INPUT_OBJECT` did not report
an API error, but the final image silently omitted the device symbols. Adding the
same bytes as `NVJITLINK_INPUT_CUBIN` preserved and resolved them.

`ObjectCodeFormatType.OBJECT` is currently documented as a "relocatable device
object". The implementation should correct that description to "host object
containing device code" so the public terminology matches the backend mapping.

## Goals

- Expose nvJitLink `-r` through `LinkerOptions`.
- Return a partial native result in a form that can be passed directly to another
  `Linker`.
- Preserve LTOIR across incremental stages when requested.
- Fail early and clearly when the selected backend, CUDA version, binding version, or
  target format cannot support the operation.
- Preserve all existing non-relocatable linker behavior.

## Non-goals

- Adding partial linking to the legacy driver `cuLink*` backend.
- Defining a new CUDA binary format or a new `ObjectCodeFormatType`.
- Automatically determining whether arbitrary externally supplied cubins have
  unresolved references.
- Making an incomplete partial result executable.
- Hiding architecture or CUDA-version compatibility rules between stages.

## Proposed API

Add one field to `LinkerOptions`:

```python
@dataclass
class LinkerOptions:
    # Existing fields ...
    relocatable: bool | None = None
```

`True` appends `-r` in `_prepare_nvjitlink_options()`. `False` and `None` do not append
it, matching the truthy-only behavior of options such as
`link_time_optimization`.

The basic non-LTO chain is:

```python
partial = Linker(
    caller_ptx,
    options=LinkerOptions(arch=arch, relocatable=True),
).link("cubin")

assert partial.code_type == "cubin"

final = Linker(
    partial,
    helper_ptx,
    options=LinkerOptions(arch=arch),
).link("cubin")
```

No conversion, byte extraction, or re-wrapping is required between stages.

### `target_type` matrix

`relocatable` selects whether the link is partial or final. `target_type` selects the
representation retrieved from the completed nvJitLink handle.

| `relocatable` | LTO | `target_type` | Result |
|---|---:|---|---|
| false/unset | false | `"cubin"` | Existing final cubin behavior |
| false/unset | true | `"cubin"` | Existing final LTO-optimized cubin behavior |
| false/unset | true | `"ptx"` | Existing linked-PTX behavior; requires `ptx=True` |
| either | true | `"ltoir"` | Linked LTOIR carried by the inputs, when its getter is available; direct PTX and cubin inputs are rejected |
| true | false | `"cubin"` | Partial native cubin, normally ELF `ET_REL` |
| true | true | `"cubin"` | Partial native cubin; later LTO cannot cross this binary boundary |
| true | true | `"ltoir"` | Partial linked LTOIR; preferred for incremental LTO |
| true | either | `"ptx"` | Rejected |
| either | either | `"object"` | Rejected; there is no corresponding output getter |

The linked-LTOIR getter is specified by CUDA in terms of `-lto`, not `-r`. Therefore
`link("ltoir")` should be accepted whenever `link_time_optimization=True`, whether or
not `relocatable` is set, provided there is no direct PTX or cubin input. Those formats
carry no LTOIR and nvJitLink silently omits them from the retrieved result. FATBIN,
host-object, and library inputs cannot be rejected from their outer format because
they may carry LTOIR, but callers should be aware that they may carry none. Without
`-r`, `nvJitLinkComplete` will still reject unresolved references in the usual way.

The existing `ObjectCodeFormatType.LTOIR` member is sufficient for the new target.
`Linker.link()` documentation and typing should be updated to list `"ltoir"` alongside
the existing output types.

## Round-tripping

### Native-code chain

A partial native result is returned through `nvJitLinkGetLinkedCubin`, so it receives
`code_type="cubin"`. The existing input mapping then adds it to the next nvJitLink
handle as `NVJITLINK_INPUT_CUBIN`.

Each intermediate stage sets `relocatable=True`; the final stage omits it. This is the
only cuda.core-specific state needed for the chain. The caller remains responsible for
using compatible architectures and CUDA versions.

### LTO-preserving chain

```python
stage1 = Linker(
    a_ltoir,
    b_ltoir,
    options=LinkerOptions(
        arch=arch,
        relocatable=True,
        link_time_optimization=True,
    ),
).link("ltoir")

stage2 = Linker(
    stage1,
    c_ltoir,
    options=LinkerOptions(
        arch=arch,
        relocatable=True,
        link_time_optimization=True,
    ),
).link("ltoir")

final = Linker(
    stage2,
    options=LinkerOptions(arch=arch, link_time_optimization=True),
).link("cubin")
```

The linked LTOIR `ObjectCode` can be passed directly because its existing `"ltoir"`
tag maps to `NVJITLINK_INPUT_LTOIR`. Every later linker that consumes it must still set
`link_time_optimization=True`; `ObjectCode` does not and should not implicitly copy
options into a new `Linker`.

### Compatibility across stages

The normal nvJitLink compatibility rules continue to apply:

- The nvJitLink library must be at least as new as the newest input.
- LTOIR compatibility is guaranteed only within a CUDA major release.
- ELF and PTX have broader cross-major compatibility, subject to architecture rules.
- Extended architectures and incompatible machine-code architectures retain their
  existing restrictions.

cuda.core should not attempt to encode all of these rules in `ObjectCode`; nvJitLink
remains the authority and reports incompatible inputs.

## Interaction with LTO

`relocatable` and `link_time_optimization` are independent controls:

- LTOIR input without `link_time_optimization=True` fails with
  `NVJITLINK_ERROR_LTO_NOT_ENABLED`.
- LTO with an unresolved reference and no `-r` fails during completion.
- `-r -lto` permits unresolved references and makes both a partial native cubin and,
  on supported CUDA versions, a linked LTOIR container available.

The choice of intermediate representation affects future optimization:

- `link("ltoir")` retains an IR-level aggregate. A later stage can optimize it jointly
  with newly supplied LTOIR.
- `link("cubin")` materializes the current aggregate as machine code. A later link can
  still resolve and combine it with PTX, cubin, or LTOIR inputs, but cannot optimize
  through the already-generated machine code.

Consequently, `"ltoir"` is the recommended target for every intermediate stage in an
LTO chain. `"cubin"` is appropriate for a non-LTO chain or when the machine-code
boundary is intentional.

## Backend and Version Handling

There are separate requirements for the option and the LTOIR output getter:

| Capability | nvJitLink library | cuda-bindings surface |
|---|---:|---|
| `relocatable=True` / `-r` | 13.2+ | No new symbol; string option only |
| `link("ltoir")` | 13.3+ | `get_linked_ltoir_size` and `get_linked_ltoir` |

The `cuLink*` fallback has no partial-link equivalent. If it is selected and
`relocatable=True`, `_prepare_driver_options()` should raise `ValueError`, following
the existing `split_compile` precedent.

If nvJitLink is selected but its runtime version is older than 13.2, construction
should fail before `nvJitLinkCreate` with a message that reports the required and
detected versions. This is preferable to exposing an opaque unrecognized-option
failure.

### Preserving cuda-bindings compatibility

cuda-core currently supports `cuda-bindings` 12.x and 13.x without a minimum minor
version. The linked-LTOIR declarations are absent from the `cynvjitlink.pxd` shipped
through cuda-bindings 13.3.1 and present by 13.4.1. Unconditionally cimporting and
calling these functions from `_linker.pyx` would therefore make cuda-core fail to build
against otherwise-supported older bindings.

The implementation should retrieve linked LTOIR through the Python-level
`cuda.bindings.nvjitlink.get_linked_ltoir_size()` and `get_linked_ltoir()` methods after
feature-detecting them. This path is not performance-sensitive and avoids a build-time
dependency on the newer PXD. Before completing a `link("ltoir")` operation, check both:

1. the Python binding exposes the two getter methods; and
2. the loaded nvJitLink library provides the corresponding functions and is 13.3+.

If either check fails, raise a clear capability error suggesting the specific CUDA or
cuda-bindings upgrade. Binary partial linking must remain usable on nvJitLink 13.2 and
with bindings that do not expose the LTOIR getters.

## Validation and Error Semantics

The implementation should reject unsupported combinations before
`nvJitLinkComplete` where possible:

- `relocatable=True` on the driver backend: `ValueError` naming the unsupported
  driver option.
- `relocatable=True` with nvJitLink older than 13.2: capability error with detected
  version.
- `target_type="ltoir"` without `link_time_optimization=True`: `ValueError` explaining
  that linked LTOIR requires `-lto`.
- `target_type="ltoir"` with a direct PTX or cubin input: `ValueError` explaining that
  the input carries no LTOIR and would otherwise be omitted from the output.
- `target_type="ltoir"` without getter support: capability error naming the missing
  CUDA or cuda-bindings support.
- `relocatable=True` with `ptx=True`, or `target_type="ptx"` in relocatable mode:
  `ValueError` explaining that partial PTX output is unsupported.
- `target_type="object"`: the existing unsupported-target `ValueError` remains.

CUDA errors caused by unresolved symbols, architecture mismatch, or incompatible input
versions should continue to surface through the existing nvJitLink error path and log.

## Module Loading

A partial cubin is linkable but is not necessarily loadable. The CUDA module loader's
behavior also depends on module loading mode:

- With eager loading, an incomplete partial cubin in the experiment failed while
  loading with CUDA error 500, `CUDA_ERROR_NOT_FOUND`.
- With default/lazy loading, module creation could appear to succeed and the error was
  deferred until kernel lookup or materialization.
- A fully resolved `ET_REL` result loaded and executed successfully in the same
  experiment.

These observations mean cuda.core should document that partial outputs must be
finalized before execution, but should not reject every `ET_REL` cubin in
`ObjectCode`. ELF type alone does not prove that unresolved references remain, and
externally supplied cubins do not carry cuda.core provenance.

No loader changes are proposed initially. If users need earlier diagnostics, a future
proposal may add non-format metadata such as `ObjectCode.is_relocatable` or an explicit
finalization check.

## Implementation Outline

1. Add and document `LinkerOptions.relocatable`.
2. Append `-r` in `_prepare_nvjitlink_options()` only when the field is true.
3. Reject the option in `_prepare_driver_options()`.
4. Gate `-r` on the loaded nvJitLink version before handle creation.
5. Extend `Linker.link()` to accept `"ltoir"`, retrieve the linked LTOIR container via
   build-compatible dynamic dispatch, and return `ObjectCode(..., "ltoir")`.
6. Add target/option validation for the matrix above.
7. Continue returning partial native output as `ObjectCode(..., "cubin")` so the
   existing input mapping provides direct round-tripping.
8. Correct the public description of `ObjectCodeFormatType.OBJECT` and clarify that
   not every `ObjectCode` is immediately loadable.
9. Add API documentation and examples for native and LTO-preserving chains.

## Test Plan

### Unit tests

- `LinkerOptions(relocatable=True).as_bytes()` contains exactly one `b"-r"`.
- False and unset values do not emit `-r`.
- The driver option builder rejects `relocatable=True`.
- Version gates distinguish nvJitLink 13.1, 13.2, and 13.3.
- Target validation covers all accepted and rejected combinations in the matrix.
- Absence of the Python linked-LTOIR getters produces a clear error without breaking
  cubin partial linking.
- The returned `ObjectCode.code_type` is `"cubin"` or `"ltoir"` according to the
  getter, never `"object"` merely because `-r` was used.

### CUDA integration tests

- Native three-stage chain: caller with unresolved helper -> partial cubin -> add
  helper -> final cubin -> launch kernel and verify output.
- Directly pass each partial `ObjectCode` to the next `Linker` without re-wrapping it.
- LTO three-stage chain: multiple LTOIR inputs -> partial linked LTOIR -> add LTOIR ->
  final cubin -> launch and verify output.
- Confirm a partial cubin can cross into a later non-LTO or LTO link while retaining
  its defined and undefined symbols.
- Exercise `-r -lto` with both `"cubin"` and `"ltoir"` targets on toolkits that expose
  the LTOIR getter.
- Skip capability-dependent cases using detected runtime functions and versions, not
  only the build-time CUDA version.

### Compatibility tests

- Build cuda-core against an older supported cuda-bindings 12.x/13.x PXD to ensure the
  LTOIR path adds no unconditional Cython symbol dependency.
- Run binary partial-link tests with CUDA 13.2.
- Run LTOIR round-trip tests with CUDA 13.3+ and a binding exposing the getter.
- Preserve the existing driver-backend linker test suite.

An incomplete partial-cubin module-load test should not assert a single failure point:
eager and lazy module loading intentionally surface the failure at different times.

## Experimental Evidence

The following experiments used an NVIDIA A30 targeting `sm_80` with nvJitLink
13.5.12, with the `-r -lto -ptx` incompatibility also reproduced on 13.4.52.

### Native-code experiment

1. Link caller PTX defining a kernel and declaring an unresolved helper with `-r`.
   Result: 3328-byte `ET_REL` cubin; the kernel was defined and the helper undefined.
2. Add that cubin as `NVJITLINK_INPUT_CUBIN` with helper PTX and link with `-r`.
   Result: 3968-byte `ET_REL` cubin; both symbols were defined.
3. Add the second result as `NVJITLINK_INPUT_CUBIN` and link without `-r`.
   Result: 4456-byte `ET_EXEC` cubin containing both symbols.
4. Relabeling the first result as `NVJITLINK_INPUT_OBJECT` completed but produced an
   `ET_EXEC` image with neither device symbol, demonstrating why `"object"` is wrong.

### LTO experiment

1. Caller LTOIR linked with `-r -lto` produced both a 3328-byte `ET_REL` cubin and a
   2156-byte LTOIR container.
2. The linked LTOIR plus helper LTOIR, again linked with `-r -lto`, produced a
   3584-byte `ET_REL` cubin and a 2288-byte linked LTOIR container.
3. Finalizing that LTOIR without `-r` produced a 2984-byte `ET_EXEC` cubin. The helper
   symbol was optimized/inlined away.
4. Using the intermediate `ET_REL` cubin instead, then adding helper LTOIR under
   `-lto`, produced a 4584-byte `ET_EXEC` cubin in which both symbols remained. This
   demonstrates the optimization boundary introduced by a cubin intermediate.
5. `-r -lto -ptx` failed with `NVJITLINK_ERROR_INCORRECT_INPUT_TYPE` on both CUDA
   13.4.52 and 13.5.12.

### Linked-LTOIR input filtering experiment

With nvJitLink 13.4.52, linking two LTOIR inputs and a PTX input under `-r -lto`
produced a cubin containing all three functions, but the linked-LTOIR getter omitted
the function supplied as PTX. A final LTO link of that retrieved LTOIR succeeded with
the function still absent and emitted no diagnostic. An `ET_REL` cubin and a SASS-only
FATBIN were omitted in the same way. FATBIN and host-object inputs that carried LTOIR
did contribute their LTOIR to the result.

## Alternatives Considered

### Return `code_type="object"`

Rejected. It selects the host-object input path on the next link. The terminology is
tempting because the output is relocatable, but the nvJitLink input category is wrong
and can silently lose device code.

### Add `code_type="relocatable_cubin"`

Rejected for the initial API. It would describe completeness rather than wire format,
require another input mapping to `NVJITLINK_INPUT_CUBIN`, and complicate external
cubin construction and serialization. Existing `"cubin"` already identifies the
correct getter and next-stage input type.

### Expose only partial cubin output

Rejected. It supports native incremental linking but prevents an LTO aggregation
consumer from retaining IR across stages.

### Require a newer cuda-bindings minor release globally

Rejected. `-r` itself needs no new binding symbol, and cuda-core intentionally supports
broad 12.x and 13.x binding ranges. Dynamic linked-LTOIR retrieval confines the newer
binding requirement to callers who request that target.

### Automatically mark or reject every `ET_REL` cubin at module load

Rejected. A resolved `ET_REL` can be loadable, external cubins lack provenance, and
parsing ELF headers would conflate file type with unresolved-link state.

## References

- [Feature issue #2369](https://github.com/NVIDIA/cuda-python/issues/2369)
- [nvJitLink 13.2 documentation](https://docs.nvidia.com/cuda/archive/13.2.0/nvjitlink/index.html)
- [nvJitLink 13.3 documentation](https://docs.nvidia.com/cuda/archive/13.3.0/nvjitlink/index.html)
- [cuda-python PR #2205: binding updates including linked-LTOIR getters][cuda-python-pr-2205]
- [cuda-python PR #2337: low-level linked-LTOIR getter coverage](https://github.com/NVIDIA/cuda-python/pull/2337)

[cuda-python-pr-2205]: https://github.com/NVIDIA/cuda-python/pull/2205
[nvptx-tools-incremental-link]: https://github.com/SourceryTools/nvptx-tools/blob/212da2e781ed0f9423824e85eb04819958513f7a/nvptx-as.cc
