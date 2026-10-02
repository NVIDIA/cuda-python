# Continuous Integration

## Checkout History

Use the default shallow checkout for jobs that only read files or test prebuilt
wheels. Source builds need complete commit ancestry and release tags so
`setuptools-scm` can derive each package's version.

For those jobs, use `fetch-depth: 2147483647` with `fetch-tags: true`.
`2147483647` is Git's `INFINITE_DEPTH` value, also used by `git fetch
--unshallow`. It produces a complete, non-shallow history. Unlike
`fetch-depth: 0`, this positive value makes `actions/checkout` fetch only the
requested ref or event SHA plus tags, instead of every branch. In particular,
it avoids the large, unrelated `gh-pages` history discussed in issue #2197.
Keep `filter: blob:none` where historical file contents are unnecessary.

Jobs comparing branches must fetch the comparison ref explicitly. CI change
detection fetches the PR base branch before computing its merge base; non-PR
runs need only a shallow checkout. PR-preview cleanup checks out the script
shallowly and fetches `gh-pages` separately when needed.

Git documents this value in its
[shallow repository reference](https://git-scm.com/docs/shallow).
The pinned checkout action chooses its refspec in
[`git-source-provider.ts`](https://github.com/actions/checkout/blob/3d3c42e5aac5ba805825da76410c181273ba90b1/src/git-source-provider.ts#L173-L202)
and
[`ref-helper.ts`](https://github.com/actions/checkout/blob/3d3c42e5aac5ba805825da76410c181273ba90b1/src/ref-helper.ts#L79-L147).

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
