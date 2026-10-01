# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json

import pytest

from ci.tools import migrate_precommit_checks as migration

REPOSITORY = "NVIDIA/cuda-python"
MAIN_SHA = "a" * 40
PR_SHA = "b" * 40
RULESET_ENDPOINT = f"repos/{REPOSITORY}/rulesets/12"


@pytest.fixture
def ruleset():
    return {
        "id": 12,
        "name": "Main protection",
        "source_type": "Repository",
        "source": REPOSITORY,
        "target": "branch",
        "enforcement": "active",
        "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}},
        "bypass_actors": [{"actor_id": 9, "actor_type": "Team", "bypass_mode": "pull_request"}],
        "rules": [
            {"type": "deletion"},
            {
                "type": "required_status_checks",
                "parameters": {
                    "strict_required_status_checks_policy": True,
                    "do_not_enforce_on_create": True,
                    "required_status_checks": [
                        {"context": "CI", "integration_id": 15368},
                        {"context": "pre-commit.ci - pr", "integration_id": 68672},
                        {"context": "Other service", "integration_id": 777},
                    ],
                },
            },
            {"type": "pull_request", "parameters": {"required_approving_review_count": 2}},
        ],
    }


@pytest.fixture
def github(monkeypatch, ruleset):
    effective_rule = {
        **copy.deepcopy(ruleset["rules"][1]),
        "ruleset_id": ruleset["id"],
        "ruleset_source_type": "Repository",
        "ruleset_source": REPOSITORY,
    }
    checks = [
        {
            "id": index,
            "name": context,
            "app": {"id": 15368},
            "status": "completed",
            "conclusion": "success",
            "details_url": f"https://github.com/{REPOSITORY}/actions/runs/{index}/job/100",
            "check_suite": {"id": index + 100},
        }
        for index, context in enumerate(migration.CHECK_WORKFLOWS, start=1)
    ]
    responses = {
        f"repos/{REPOSITORY}/rules/branches/main?per_page=100": [[effective_rule]],
        RULESET_ENDPOINT: copy.deepcopy(ruleset),
        f"repos/{REPOSITORY}": {"default_branch": "main"},
        f"repos/{REPOSITORY}/branches/main": {"commit": {"sha": MAIN_SHA}},
        f"repos/{REPOSITORY}/commits/{MAIN_SHA}/check-runs?filter=latest&per_page=100": [{"check_runs": checks}],
        f"repos/{REPOSITORY}/commits/{PR_SHA}/check-runs?filter=latest&per_page=100": [{"check_runs": checks}],
        f"repos/{REPOSITORY}/commits/{PR_SHA}/pulls?per_page=100": [
            [
                {
                    "state": "open",
                    "base": {"ref": "main", "repo": {"full_name": REPOSITORY}},
                    "head": {"sha": PR_SHA},
                    "merge_commit_sha": "c" * 40,
                }
            ]
        ],
    }
    for path in set(migration.CHECK_WORKFLOWS.values()):
        responses[f"repos/{REPOSITORY}/contents/{path}?ref={MAIN_SHA}"] = {"path": path, "type": "file"}
    for check in checks:
        responses[f"repos/{REPOSITORY}/actions/runs/{check['id']}"] = {
            "path": migration.CHECK_WORKFLOWS[check["name"]],
            "check_suite_id": check["check_suite"]["id"],
            "status": "completed",
            "conclusion": "success",
        }
    calls = []

    def api(endpoint, *, method="GET", payload=None, paginate=False):
        calls.append((method, endpoint, copy.deepcopy(payload)))
        if method == "PUT":
            responses[endpoint]["rules"] = copy.deepcopy(payload["rules"])
        return copy.deepcopy(responses[endpoint])

    monkeypatch.setattr(migration, "github_api", api)
    return responses, calls


@pytest.mark.agent_authored(model="gpt-6")
def test_replaces_service_without_changing_unrelated_rules_or_parameters(ruleset):
    original = copy.deepcopy(ruleset)
    migrated = migration.migrate_rules(ruleset["rules"])

    assert ruleset == original
    assert migrated[0] == original["rules"][0]
    assert migrated[2] == original["rules"][2]
    parameters = migrated[1]["parameters"]
    assert parameters["strict_required_status_checks_policy"] is True
    assert parameters["do_not_enforce_on_create"] is True
    checks = parameters["required_status_checks"]
    assert checks == [
        {"context": "CI", "integration_id": 15368},
        {"context": "Pre-commit (Linux)", "integration_id": 15368},
        {"context": "Pre-commit (Windows)", "integration_id": 15368},
        {"context": "Documentation links", "integration_id": 15368},
        {"context": "Other service", "integration_id": 777},
    ]
    assert migration.migrate_rules(migrated) == migrated


@pytest.mark.agent_authored(model="gpt-6")
def test_existing_actions_requirements_are_not_duplicated(ruleset):
    checks = ruleset["rules"][1]["parameters"]["required_status_checks"]
    existing = {"context": "Documentation links", "integration_id": 15368}
    checks.append(existing)

    migrated = migration.migrate_rules(ruleset["rules"])

    assert migrated[1]["parameters"]["required_status_checks"].count(existing) == 1


@pytest.mark.parametrize("integration", [None, 777])
@pytest.mark.agent_authored(model="gpt-6")
def test_refuses_unexpected_old_integration(ruleset, integration):
    ruleset["rules"][1]["parameters"]["required_status_checks"][1]["integration_id"] = integration

    with pytest.raises(RuntimeError, match="expected integration"):
        migration.migrate_rules(ruleset["rules"])


@pytest.mark.agent_authored(model="gpt-6")
def test_refuses_conflicting_replacement_integration(ruleset):
    ruleset["rules"][1]["parameters"]["required_status_checks"].append(
        {"context": "Documentation links", "integration_id": 777}
    )

    with pytest.raises(RuntimeError, match="different or unspecified integration"):
        migration.migrate_rules(ruleset["rules"])


@pytest.mark.agent_authored(model="gpt-6")
def test_duplicate_valid_context_cannot_hide_conflicting_integration(ruleset):
    ruleset["rules"][1]["parameters"]["required_status_checks"].extend(
        [
            {"context": "Documentation links", "integration_id": 777},
            {"context": "Documentation links", "integration_id": 15368},
        ]
    )

    with pytest.raises(RuntimeError, match="different or unspecified integration"):
        migration.migrate_rules(ruleset["rules"])


@pytest.mark.agent_authored(model="gpt-6")
def test_cli_defaults_to_read_only_preview(github, capsys):
    responses, calls = github

    assert migration.main(["--repo", REPOSITORY]) == 0

    preview = json.loads(capsys.readouterr().out)
    assert preview["mode"] == "dry-run"
    assert preview["changes"][0]["before"] == responses[RULESET_ENDPOINT]
    assert set(preview["changes"][0]["put_payload"]) == {"rules"}
    assert all(method == "GET" for method, endpoint, payload in calls)
    assert not any("/contents/" in endpoint for method, endpoint, payload in calls)


@pytest.mark.agent_authored(model="gpt-6")
def test_inherited_requirement_is_rejected_before_any_write(github):
    responses, calls = github
    rule = responses[f"repos/{REPOSITORY}/rules/branches/main?per_page=100"][0][0]
    rule["ruleset_source_type"] = "Organization"
    rule["ruleset_source"] = "NVIDIA"

    with pytest.raises(RuntimeError, match="inherited from Organization NVIDIA ruleset 12"):
        migration.discover_changes(REPOSITORY)

    assert len(calls) == 1


@pytest.mark.agent_authored(model="gpt-6")
def test_replacement_workflows_must_be_files_on_current_main(github):
    responses, calls = github
    responses[f"repos/{REPOSITORY}/contents/.github/workflows/lychee.yml?ref={MAIN_SHA}"] = []

    with pytest.raises(RuntimeError, match="not a workflow file on current main"):
        migration.apply_changes(REPOSITORY, migration.discover_changes(REPOSITORY), None)

    assert all(method == "GET" for method, endpoint, payload in calls)


@pytest.mark.parametrize("verified_sha", [None, PR_SHA])
@pytest.mark.agent_authored(model="gpt-6")
def test_apply_requires_green_representative_checks_and_preserves_settings(github, ruleset, verified_sha):
    responses, calls = github
    changes = migration.discover_changes(REPOSITORY)

    assert migration.apply_changes(REPOSITORY, changes, verified_sha) == (verified_sha or MAIN_SHA)

    writes = [call for call in calls if call[0] == "PUT"]
    assert writes == [("PUT", RULESET_ENDPOINT, {"rules": migration.migrate_rules(ruleset["rules"])})]
    assert {key: value for key, value in responses[RULESET_ENDPOINT].items() if key != "rules"} == {
        key: value for key, value in ruleset.items() if key != "rules"
    }


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "skipped", "neutral", None])
@pytest.mark.agent_authored(model="gpt-6")
def test_failed_or_skipped_checks_prevent_all_writes(github, conclusion):
    responses, calls = github
    checks = responses[f"repos/{REPOSITORY}/commits/{MAIN_SHA}/check-runs?filter=latest&per_page=100"][0]["check_runs"]
    checks[-1]["conclusion"] = conclusion
    changes = migration.discover_changes(REPOSITORY)

    with pytest.raises(RuntimeError, match="not completed and successful"):
        migration.apply_changes(REPOSITORY, changes, None)

    assert all(method == "GET" for method, endpoint, payload in calls)


@pytest.mark.agent_authored(model="gpt-6")
def test_newer_pending_run_cannot_be_masked_by_older_success(github):
    responses, calls = github
    checks = responses[f"repos/{REPOSITORY}/commits/{MAIN_SHA}/check-runs?filter=latest&per_page=100"][0]["check_runs"]
    checks.append({**checks[0], "id": 99, "status": "in_progress", "conclusion": None})

    with pytest.raises(RuntimeError, match="not completed and successful"):
        migration.apply_changes(REPOSITORY, migration.discover_changes(REPOSITORY), None)

    assert all(method == "GET" for method, endpoint, payload in calls)


@pytest.mark.agent_authored(model="gpt-6")
def test_other_apps_cannot_satisfy_replacement_checks(github):
    responses, calls = github
    checks = responses[f"repos/{REPOSITORY}/commits/{MAIN_SHA}/check-runs?filter=latest&per_page=100"][0]["check_runs"]
    checks[0]["app"]["id"] = 777

    with pytest.raises(RuntimeError, match="Missing GitHub Actions check"):
        migration.apply_changes(REPOSITORY, migration.discover_changes(REPOSITORY), None)

    assert all(method == "GET" for method, endpoint, payload in calls)


@pytest.mark.parametrize(
    ("field", "value"),
    [("path", ".github/workflows/unrelated.yml"), ("check_suite_id", 777), ("conclusion", "failure")],
)
@pytest.mark.agent_authored(model="gpt-6")
def test_check_must_belong_to_the_expected_successful_workflow(github, field, value):
    responses, calls = github
    responses[f"repos/{REPOSITORY}/actions/runs/1"][field] = value

    with pytest.raises(RuntimeError, match="does not belong to a successful"):
        migration.apply_changes(REPOSITORY, migration.discover_changes(REPOSITORY), None)

    assert all(method == "GET" for method, endpoint, payload in calls)


@pytest.mark.agent_authored(model="gpt-6")
def test_unassociated_sha_is_rejected(github):
    responses, calls = github
    responses[f"repos/{REPOSITORY}/commits/{PR_SHA}/pulls?per_page=100"] = [[]]

    with pytest.raises(RuntimeError, match="open PR targeting main"):
        migration.apply_changes(REPOSITORY, migration.discover_changes(REPOSITORY), PR_SHA)

    assert all(method == "GET" for method, endpoint, payload in calls)


@pytest.mark.agent_authored(model="gpt-6")
def test_ruleset_edit_during_preflight_prevents_writes(github):
    responses, calls = github
    changes = migration.discover_changes(REPOSITORY)
    responses[RULESET_ENDPOINT]["conditions"]["ref_name"]["exclude"].append("refs/heads/release")

    with pytest.raises(RuntimeError, match="changed after discovery"):
        migration.apply_changes(REPOSITORY, changes, None)

    assert all(method == "GET" for method, endpoint, payload in calls)


@pytest.mark.agent_authored(model="gpt-6")
def test_sha_must_be_explicit_full_commit(github):
    responses, calls = github

    with pytest.raises(SystemExit, match="2"):
        migration.main(["--repo", REPOSITORY, "--verified-sha", "main", "--apply"])

    assert not calls
