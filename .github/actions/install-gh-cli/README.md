# install-gh-cli

Installs the GitHub CLI (`gh`) from the `cli.github.com` apt repository on
Linux jobs that don't already have it. If `gh` is already on `PATH` the action
exits immediately.

## Why the key is checked in

The standard upstream install instructions download the repository signing key
over HTTPS and trust whatever bytes come back. The security of every `gh`
package installed in CI then rests solely on the TLS trust store of the runner
or container and the integrity of `cli.github.com`. Several workflows supply
`gh` with a `GITHUB_TOKEN`, so a substituted key could place a malicious
binary into those jobs.

Checking the public key into the repository means apt will only accept packages
signed by a key that was reviewed in a pull request. The file contains exactly
one key, because apt trusts every key listed in a `signed-by=` keyring; a
multi-key file would widen trust unnecessarily.

Packages from Ubuntu's own repositories don't need this: their signing keys
ship with the OS image in the `ubuntu-keyring` package, so apt checks them
against keys that are already on disk. `cli.github.com` is the only
third-party apt repository in our CI, and this action gives it the same
property: a key present before the job runs, not one fetched during it.

## Updating the key

GitHub occasionally rotates the signing key. The previous key (fingerprint
`2C61 0620 1985 B60E 6C7A C873 23F3 D4EA 7571 6059`) expired on 2026-09-05.
The current key (fingerprint `7F38 BBB5 9D06 4DBC B3D8 4D72 5612 B364 6231
3325`) was created on 2026-04-07 and has no expiry.

A key rotation shows up in CI as `apt-get update` failing with a `NO_PUBKEY`
or `EXPKEYSIG` error for the `cli.github.com` repository.

To update the key:

1. Find the new key's fingerprint in the [GitHub CLI Linux install docs](https://github.com/cli/cli/blob/trunk/docs/install_linux.md); the active fingerprints are listed near the top of the page. The docs are the reference; the key server is not an independent check since step 2 also fetches from the same host.

2. Download the armored key for the new fingerprint directly from GitHub CLI's
   published ASC file and extract only that key:

   ```bash
   tmp=$(mktemp -d)
   curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.asc \
     | gpg --homedir "$tmp" --batch --import
   gpg --homedir "$tmp" --batch --armor --export <FINGERPRINT> \
     > .github/actions/install-gh-cli/githubcli-archive-keyring.asc
   ```

3. Confirm the exported key signs the current repository metadata:

   ```bash
   gpg --dearmor < .github/actions/install-gh-cli/githubcli-archive-keyring.asc \
     > "$tmp/key.gpg"
   curl -fsSL https://cli.github.com/packages/dists/stable/InRelease \
     | gpgv --keyring "$tmp/key.gpg" -
   ```

   The output must say `Good signature from "GitHub CLI <...>"`.

4. Update this README: replace the old fingerprint and creation/expiry dates
   above with the new key's details.

5. Open a pull request. Include the new fingerprint, the creation and expiry
   dates from `gpg --show-keys`, and the `gpgv` output from step 3 in the PR
   description.
