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
release lines. Each line declares its source directory, exact toolkit build/test
pin, and prerelease-tag policy. Roles such as `current` and `maintenance` are
orchestration aliases for those lines.

Use `ci/tools/bindings_config.py` instead of reading the YAML directly. The
resolver validates the registry and emits JSON with the configured values plus
the derived CTK target, tag series, and CUDA ABI major/variant:

```console
python ci/tools/bindings_config.py
python ci/tools/bindings_config.py --lines
python ci/tools/bindings_config.py --role current
```

A bindings line is distinct from a CUDA ABI major, so same-major lines remain
separate in the registry and CI plan. The public wheel builder currently
requires one `current` line and one `maintenance` line with different ABI
majors; it fails if the registry does not meet that narrower requirement.
