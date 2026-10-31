# CI Tool Tests

These instructions apply only to tests under `ci/tools/tests/`. Package and
runtime tests follow their own guidance.

Keep a test only when wrong behavior would be silent: nothing would fail
loudly in CI, but the result would be wrong. Cut a test when a mistake already
produces an immediate, obvious CI failure. This includes environment-variable
wiring, wheel counts, argument validation, exact pip command lines, and the
schema of a checked-in file that every CI run loads.

Useful outcome coverage includes skipped required tests, wrong release
artifacts, incorrect version milestones, missing Core ABI variants, and
silently lost local-wheel matrix coverage. Assert those outcomes rather than
workflow text, step names, cosmetic labels, or intermediate command order.

Use self-contained metadata fixtures so routine version and dependency bumps
do not break unrelated tests. If a test intentionally uses the live registry,
derive its toolkit and package values from that registry. Keep fixtures and
parameter sets as small as the distinct silent-failure risks allow.
