from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

GIT = shutil.which("git")
LOOKUP_RUN_ID = Path(__file__).resolve().parents[1] / "lookup-run-id"


def _run(cmd: list[str], cwd: Path) -> str:
    return subprocess.check_output(cmd, cwd=cwd, text=True).strip()  # noqa: S603


def _run_checked(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)  # noqa: S603


def _init_repo(tmp_path: Path) -> tuple[Path, str]:
    assert GIT is not None
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_checked([GIT, "init"], cwd=repo)
    _run_checked([GIT, "config", "user.name", "Test User"], cwd=repo)
    _run_checked([GIT, "config", "user.email", "test@example.com"], cwd=repo)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _run_checked([GIT, "add", "README.md"], cwd=repo)
    _run_checked([GIT, "commit", "-m", "init"], cwd=repo)
    commit_sha = _run([GIT, "rev-parse", "HEAD"], cwd=repo)
    _run_checked([GIT, "tag", "-a", "cuda-core-v0.6.0", "-m", "release"], cwd=repo)
    return repo, commit_sha


def _write_fake_gh(tmp_path: Path, commit_sha: str, tag_name: str) -> Path:
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    gh = fakebin / "gh"
    gh.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail

if [[ "${{1:-}}" == "run" && "${{2:-}}" == "list" ]]; then
    shift 2
    commit=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --commit)
                commit="$2"
                shift 2
                ;;
            --repo|--workflow|--status|--json|--limit|-R|-w|-s|-L|-b)
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done

    if [[ "$commit" == "{commit_sha}" ]]; then
        cat <<EOF
[{{"databaseId":123,"workflowName":"CI","status":"completed","conclusion":"success","headSha":"{commit_sha}","headBranch":"{tag_name}","event":"push","createdAt":"2026-01-01T00:00:00Z","url":"https://example.test/runs/123"}}]
EOF
    else
        echo '[]'
    fi
    exit 0
fi

if [[ "${{1:-}}" == "run" && "${{2:-}}" == "view" ]]; then
    echo '{{"url":"https://example.test/runs/123"}}'
    exit 0
fi

echo "unexpected gh invocation: $*" >&2
exit 1
""",
        encoding="utf-8",
    )
    gh.chmod(gh.stat().st_mode | stat.S_IXUSR)
    return fakebin


def test_lookup_run_id_should_peel_annotated_tag_to_commit_if_tag_mode(tmp_path):
    tag_name = "cuda-core-v0.6.0"
    repo, commit_sha = _init_repo(tmp_path)
    fakebin = _write_fake_gh(tmp_path, commit_sha, tag_name)
    env = os.environ.copy()
    env["GH_TOKEN"] = "test-token"  # noqa: S105
    env["PATH"] = f"{fakebin}{os.pathsep}{env['PATH']}"

    result = subprocess.run(  # noqa: S603
        [str(LOOKUP_RUN_ID), "--tag", tag_name, "NVIDIA/cuda-python"],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "123\n"
    assert f"Resolved tag '{tag_name}' to commit: {commit_sha}" in result.stderr
