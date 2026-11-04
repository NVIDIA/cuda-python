# Approved released-CUDA downloads

`fetch_ctk` installs released CUDA using the reviewed manifest hashes in
`ci/tools/fetch_ctk_redistrib.py` (`KNOWN_REDISTRIB_SHA256`).
Unknown released CUDA versions emit a warning and proceed without a manifest pin.
The existing internal-kitpick setup path is unchanged.

## CUDA

The CUDA pin is the SHA256 of the exact public `redistrib_<version>.json` bytes.
For known versions, verify this before parsing the manifest; a mismatch fails.
For unknown versions, manifest verification warns and continues. Every component
archive is checked against its SHA256 in the downloaded manifest before extraction.
Without a manifest pin, those component hashes have no independent trust anchor.
The shared action caches the assembled toolkit. Cache keys use the CUDA
version, platform, and component list; cache restores trust GitHub Actions cache
integrity.

The initial 13.4.2 manifest was retrieved from:

<https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.4.2.json>

Its archive records match internal CUDA 13.4.2 kitpick 006; only `release_date`
differs. This is a checked release mapping, not a rule that the last kitpick is
the public release. Changing manifest whitespace also changes the pinned hash.

## Python

Python provisioning uses the shared `setup-hardened-python` action for both
released-CUDA and internal-kitpick jobs. It installs CPython through checksum-verified
uv distributions, creates a virtual environment, and checks that the interpreter
rejects the compromised CA keys. See that action for the pinned tool versions
and interpreter validation policy.

## Updates and scope

Update versions and hashes together in a reviewed PR. Obtain the expected bytes
through a trusted release process, independently cross-check the release, and
compute the digest over the actual downloaded file. Do not regenerate or accept
pins automatically during a test run. Test a supported matrix entry and verify
that changed bytes are rejected before extraction.

These pins protect these downloads, not the whole runner. Test-package locking,
MSVC provisioning, internal-kitpick integrity, runner isolation, and protection
against persistent host modifications are separate controls. GB10 hardware
validation is still required for toolchain changes.
