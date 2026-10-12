# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a pytest suite in heap-bounded batches, then merge the JUnit XML.

A long-running pytest session keeps every executed item's ``TestReport``\\ s
and (with ``--junitxml``) JUnit XML node alive until the session ends, and
``gc.collect()`` walks the whole live heap on every call. On the cuda.core
suite under ``--count=N`` (nightly repeats), that makes the per-test
teardown collect get more expensive as the run proceeds: pytest-repeat
expands ``--count=N`` into N times the collected items up front, and the
executed-item bookkeeping grows throughout the run, so repeats cost
superlinearly (5 repeats measured at ~10x one repeat, not ~5x). See the
"cuda.core test time" investigation linked from
https://github.com/NVIDIA/cuda-python/pull/3041 and
https://github.com/NVIDIA/cuda-python/pull/3045.

This script collects the full, already-``--count``-expanded item list once,
then runs it across several pytest subprocesses of at most ``--batch-size``
items each. Each subprocess starts with an empty heap, so neither term above
accumulates past one batch. The per-batch JUnit XML files are merged back
into a single file at ``--junitxml``'s path, so this script is a drop-in
replacement for a plain ``pytest`` invocation from the caller's perspective:
same CLI, same final JUnit XML, same aggregate exit code.

Usage::

    python ci/tools/pytest_batched.py [--batch-size N] [--runner-prefix "CMD"] <pytest args...>

``--runner-prefix`` prepends tokens (split with ``shlex.split``) to each
batch's pytest invocation, for wrapping it in a tool such as
compute-sanitizer. It is not applied during collection.

The environment (including ``PYTEST_ADDOPTS``, e.g. ``--count=N`` from
pytest-repeat) is passed unchanged to every batch run. A repeated item's node
ID (``test_foo[1-3]``) only exists once pytest-repeat's collection hook
re-parametrizes the module, which requires ``--count`` to still be active;
pytest still only *runs* the items named explicitly on the command line, so
this does not re-execute anything, it only lets the batch rebuild the
collection tree needed to resolve the ID.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_BATCH_SIZE = 1000

# pytest sums -v/-q occurrences into a signed verbosity level rather than
# letting the last one win, so a plain substitution of "-v" with "-q" in the
# caller's args is not enough: if both end up present the counts can cancel
# out. The flat, one-node-ID-per-line "--collect-only -q" format needs a
# verbosity we control exactly, so these are stripped before collection and a
# fresh "-q" is appended (see _args_for_collection).
_VERBOSITY_FLAGS = frozenset({"-v", "-vv", "-vvv", "-vvvv", "--verbose", "-q", "-qq", "-qqq", "--quiet"})

# pytest exit codes (see _pytest/config/__init__.py ExitCode): 0 all passed, 1
# some tests failed, 5 no tests were collected. 2/3/4 are interrupted/internal
# error/usage error and must propagate from any batch.
_EXIT_NO_TESTS_COLLECTED = 5


def _args_for_collection(pytest_args: list[str]) -> list[str]:
    """``pytest_args`` with verbosity flags removed, for the collect-only pass."""
    return [a for a in pytest_args if a not in _VERBOSITY_FLAGS]


def parse_collected_ids(output: str) -> list[str]:
    """Extract node IDs from ``pytest --collect-only -q`` output.

    That format is one node ID per line, followed by a blank line and a
    summary line such as ``"12 tests collected in 0.05s"``. Warnings (e.g. an
    unregistered marker) can also be interleaved; node IDs always contain
    ``"::"``, which no summary or warning line does.
    """
    ids = []
    for line in output.splitlines():
        line = line.strip()
        if "::" not in line:
            continue
        ids.append(line)
    return ids


def collect_ids(pytest_args: list[str], env: dict[str, str]) -> list[str]:
    """Run ``pytest --collect-only`` and return the collected node IDs."""
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", *_args_for_collection(pytest_args)]
    result = subprocess.run(  # noqa: S603 - argv is passed directly without a shell.
        cmd, text=True, capture_output=True, env=env, check=False
    )
    if result.returncode not in (0, _EXIT_NO_TESTS_COLLECTED):
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return parse_collected_ids(result.stdout)


def chunk(items: list[str], batch_size: int) -> list[list[str]]:
    """Split ``items`` into consecutive batches of at most ``batch_size``."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def strip_path_positionals(args: list[str], root: Path) -> list[str]:
    """Drop positional path arguments (e.g. ``tests/``).

    Batches pass explicit node IDs instead; pytest would otherwise also
    collect everything under a leftover path argument.
    """
    return [a for a in args if a.startswith("-") or not (root / a).exists()]


def split_junitxml(args: list[str]) -> tuple[list[str], str | None]:
    """Remove ``--junitxml``/``--junit-xml`` from ``args``; return (remaining, path)."""
    remaining: list[str] = []
    junitxml: str | None = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--junitxml", "--junit-xml"):
            if i + 1 >= len(args):
                raise ValueError(f"{arg} requires a path argument")
            junitxml = args[i + 1]
            i += 2
            continue
        matched_prefix = False
        for prefix in ("--junitxml=", "--junit-xml="):
            if arg.startswith(prefix):
                junitxml = arg[len(prefix) :]
                matched_prefix = True
                break
        if not matched_prefix:
            remaining.append(arg)
        i += 1
    return remaining, junitxml


def merge_junit_xml(batch_paths: list[Path], output_path: Path) -> None:
    """Combine per-batch JUnit XML files into one ``<testsuites>`` document.

    Each batch file has pytest's usual single ``<testsuite>`` under
    ``<testsuites>``. This concatenates their ``<testcase>`` children into one
    ``<testsuite>`` and sums the summary attributes, so the merged file reads
    like a single pytest session ran every item.
    """
    merged_suite: ET.Element | None = None
    totals = {"errors": 0, "failures": 0, "skipped": 0, "tests": 0}
    total_time = 0.0
    earliest_timestamp: str | None = None
    global_properties: ET.Element | None = None

    for path in batch_paths:
        # This is JUnit XML this same CI job just wrote, not untrusted input.
        suite = ET.parse(path).getroot().find("testsuite")  # noqa: S314
        if suite is None:
            continue
        if merged_suite is None:
            merged_suite = ET.Element("testsuite", attrib=dict(suite.attrib))
            earliest_timestamp = suite.get("timestamp")
        for key in totals:
            totals[key] += int(suite.get(key, "0"))
        total_time += float(suite.get("time", "0"))
        timestamp = suite.get("timestamp")
        if timestamp and earliest_timestamp and timestamp < earliest_timestamp:
            earliest_timestamp = timestamp
        for child in list(suite):
            if child.tag == "properties":
                # Session-wide global properties (record_testsuite_property) are
                # identical across batches; keep only the first. A <testcase>'s
                # own per-test <properties> (record_property) stays nested
                # inside it and is unaffected, since this loop only sees
                # <testsuite>'s direct children.
                if global_properties is None:
                    global_properties = child
                continue
            merged_suite.append(child)

    if merged_suite is None:
        merged_suite = ET.Element(
            "testsuite", name="pytest", errors="0", failures="0", skipped="0", tests="0", time="0.000"
        )
    else:
        merged_suite.set("errors", str(totals["errors"]))
        merged_suite.set("failures", str(totals["failures"]))
        merged_suite.set("skipped", str(totals["skipped"]))
        merged_suite.set("tests", str(totals["tests"]))
        merged_suite.set("time", f"{total_time:.3f}")
        if earliest_timestamp:
            merged_suite.set("timestamp", earliest_timestamp)
        if global_properties is not None:
            merged_suite.insert(0, global_properties)

    testsuites = ET.Element("testsuites", name="pytest tests")
    testsuites.append(merged_suite)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>' + ET.tostring(testsuites, encoding="unicode"),
        encoding="utf-8",
    )


def _parse_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("PYTEST_BATCH_SIZE", DEFAULT_BATCH_SIZE)),
        help="Maximum collected items per batch session (default: %(default)s, or $PYTEST_BATCH_SIZE)",
    )
    parser.add_argument(
        "--runner-prefix",
        default="",
        help="Command tokens (shlex-split) to prepend to each batch's pytest invocation, e.g. a sanitizer wrapper",
    )
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns, pytest_args = _parse_args(argv)
    pytest_args, junitxml = split_junitxml(pytest_args)
    cwd = Path.cwd()

    ids = collect_ids(pytest_args, os.environ.copy())
    if not ids:
        print("pytest-batched: no tests collected", file=sys.stderr)
        if junitxml:
            merge_junit_xml([], Path(junitxml))
        return _EXIT_NO_TESTS_COLLECTED

    batches = chunk(ids, ns.batch_size)
    print(f"pytest-batched: {len(ids)} items -> {len(batches)} batch(es) of up to {ns.batch_size}", file=sys.stderr)

    # PYTEST_ADDOPTS (e.g. --count=N from pytest-repeat) must stay active for
    # the batch runs, not just collection: a repeated item's node ID, such as
    # "test_foo[1-3]", only exists once pytest-repeat's pytest_generate_tests
    # hook re-parametrizes the module, which only happens when --count is
    # still in effect. Passing the bare node ID without it is a collection
    # error ("not found"), not a harmless re-expansion. pytest still only
    # *runs* the items named on the command line, so this does not re-run
    # work: it only lets pytest rebuild the tree needed to resolve the ID.
    run_env = os.environ.copy()
    batch_opts = strip_path_positionals(pytest_args, cwd)
    runner_prefix = shlex.split(ns.runner_prefix)

    batch_xml_paths: list[Path] = []
    returncodes: list[int] = []
    for index, batch in enumerate(batches):
        extra: list[str] = []
        batch_xml: Path | None = None
        if junitxml:
            batch_xml = Path(f"{junitxml}.batch{index}.xml")
            extra = ["--junitxml", str(batch_xml)]
        print(f"pytest-batched: batch {index + 1}/{len(batches)} ({len(batch)} items)", file=sys.stderr)
        cmd = [*runner_prefix, sys.executable, "-m", "pytest", *batch_opts, *extra, *batch]
        result = subprocess.run(cmd, env=run_env, check=False)  # noqa: S603 - argv is passed directly without a shell.
        returncodes.append(result.returncode)
        if batch_xml is not None and batch_xml.exists():
            batch_xml_paths.append(batch_xml)

    if junitxml:
        merge_junit_xml(batch_xml_paths, Path(junitxml))
        for path in batch_xml_paths:
            path.unlink(missing_ok=True)

    significant = [rc for rc in returncodes if rc != 0 and rc != _EXIT_NO_TESTS_COLLECTED]
    if significant:
        return max(significant)
    if all(rc == _EXIT_NO_TESTS_COLLECTED for rc in returncodes):
        return _EXIT_NO_TESTS_COLLECTED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
