# Continuous Integration

## Repository Customizations

The workflows in this repository use the `CI_CUSTOMIZATIONS_*` namespace for
GitHub Actions configuration variables that opt an alternative synchronized
repository into repository-specific CI behavior. This keeps the workflow logic
shared without hard-coding the names of private repositories into the public
source tree.

These variables are non-secret strings configured under
**Settings > Secrets and variables > Actions > Variables**.
An unset variable, or any value other than the literal string `true`, leaves
the customization disabled. Do not store credentials or other secret values in
these variables.

| Variable | Default | Purpose |
| --- | --- | --- |
| `CI_CUSTOMIZATIONS_SECURITY_SUITE_ENABLED` | Disabled | Enables the NVIDIA Security Suite after its runner, Actions variables, and OIDC/Vault authorization have been provisioned for the repository. |

The canonical `NVIDIA/cuda-python` repository does not need this variable
because its standard workflow behavior is enabled directly. Before enabling a
customization elsewhere, document the repository-specific prerequisites and
verification procedure in that repository's own documentation.

## CUDA Bindings Line Registry

`ci/versions.yml` is the authoritative public registry for CUDA bindings
release lines. Each stable line ID maps an exact CTK target and build/test pin
to an explicit source directory. Roles are orchestration aliases: `current`
selects one line, while `maintenance` and `unreleased` are ordered lists.
Public builds operate on `current` plus `maintenance`; the public registry
normally keeps `unreleased` empty.

Use `ci/tools/bindings_config.py` instead of reading the YAML directly. The
resolver validates the registry and emits normalized records containing the
line ID, source directory, CTK target, toolkit pin, tag series, role membership,
the line-specific alpha/beta tag policy, and the derived CUDA ABI major/variant.
For example:

```console
python ci/tools/bindings_config.py validate
python ci/tools/bindings_config.py list --scope public
python ci/tools/bindings_config.py get --role current
python ci/tools/bindings_config.py match-tag v13.3.0
```

A bindings line and a CUDA ABI major are different dimensions. Multiple
unreleased lines such as 13.4 and 13.5 may need separate roots, artifacts, and
tests even though both derive the `cu13` ABI variant. CUDA Core packaging may
deduplicate work by ABI major only where doing so preserves line-specific
compatibility checks.

The public wheel builder currently has an explicit transitional boundary: it
requires one `current` line and exactly one `maintenance` line with different
CUDA ABI majors. The registry, change planner, sdist jobs, and test-matrix
validation are list-based and preserve same-major line identity, but the
monolithic wheel job fails closed rather than pretending it can build multiple
maintenance lines or two release lines for one ABI. Extending that job is a
separate reviewer-visible design change.

Future private CTK overlays may add records assigned to `unreleased`, but must
produce the same validated effective schema and keep private roots out of the
public projection. Overlay merging and promotion are intentionally outside the
scope of the public resolver today.
