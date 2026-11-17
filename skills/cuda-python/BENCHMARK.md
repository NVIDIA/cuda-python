# Skill Benchmark: cuda-python

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cuda-python`
- Validation status: `passed`
- Evaluation date: 2026-11-17
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 20 evaluation tasks (18 positive, 2 negative)
- Dataset digest: `sha256:019be8a7baa74a5b59a3b1eefde1bc48ea803066854740ed47a1be25fbd83a0f` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| cuda-python | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cuda-python | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cuda-python | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cuda-python | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cuda-python | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `cuda-python`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 84.0% — baseline ran, but no comparable score was available; uplift unavailable | 74.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 80.0% → 90.0% (+10.0 points) | 82.5% → 67.5% (-15.0 points) |
| Correctness | 70.0% → 94.0% (+24.0 points) | 78.0% → 92.0% (+14.0 points) |
| Discoverability | 83.3% — baseline ran, but no comparable score was available; uplift unavailable | 83.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 56.0% → 68.0% (+12.0 points) | 50.3% → 62.5% (+12.2 points) |
| Efficiency | 84.5% — baseline ran, but no comparable score was available; uplift unavailable | 67.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 93.5% → 93.2% (-0.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 80.4% → 84.2% (+3.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 8,487,347 | 6,531,841 | +1,955,506 | +29.94% | skill 20/20; base 20/20 |
| claude-code | cuda-python-any-array | 820,091 | 432,158 | +387,933 | +89.77% | skill 1/1; base 1/1 |
| claude-code | cuda-python-core-coverage-question | 75,607 | 74,393 | +1,214 | +1.63% | skill 1/1; base 1/1 |
| claude-code | cuda-python-fix-hardcoded-paths | 1,050,391 | 798,047 | +252,344 | +31.62% | skill 1/1; base 1/1 |
| claude-code | cuda-python-fix-ignored-errors | 631,295 | 779,614 | -148,319 | -19.02% | skill 1/1; base 1/1 |
| claude-code | cuda-python-fix-scalar-args | 297,621 | 297,568 | +53 | +0.02% | skill 1/1; base 1/1 |
| claude-code | cuda-python-fix-stream-order | 339,681 | 297,100 | +42,581 | +14.33% | skill 1/1; base 1/1 |
| claude-code | cuda-python-gpu-report | 579,394 | 482,669 | +96,725 | +20.04% | skill 1/1; base 1/1 |
| claude-code | cuda-python-install-question | 74,522 | 36,307 | +38,215 | +105.26% | skill 1/1; base 1/1 |
| claude-code | cuda-python-kernel-timing | 631,674 | 346,931 | +284,743 | +82.07% | skill 1/1; base 1/1 |
| claude-code | cuda-python-migration-question | 331,087 | 452,529 | -121,442 | -26.84% | skill 1/1; base 1/1 |
| claude-code | cuda-python-perf-launch-loop | 399,171 | 651,455 | -252,284 | -38.73% | skill 1/1; base 1/1 |
| claude-code | cuda-python-port-core-experimental | 1,089,223 | 222,228 | +866,995 | +390.14% | skill 1/1; base 1/1 |
| claude-code | cuda-python-port-legacy-bindings | 764,385 | 226,758 | +537,627 | +237.09% | skill 1/1; base 1/1 |
| claude-code | cuda-python-port-pycuda | 561,572 | 477,365 | +84,207 | +17.64% | skill 1/1; base 1/1 |
| claude-code | cuda-python-review-calibrate | 343,807 | 343,245 | +562 | +0.16% | skill 1/1; base 1/1 |
| claude-code | cuda-python-saxpy-numpy | 274,825 | 429,952 | -155,127 | -36.08% | skill 1/1; base 1/1 |
| claude-code | cuda-python-strided-view-question | 75,299 | 37,905 | +37,394 | +98.65% | skill 1/1; base 1/1 |
| claude-code | cuda-python-struct-arg-question | 75,285 | 74,121 | +1,164 | +1.57% | skill 1/1; base 1/1 |
| claude-code | negative-cupy-arrays | 36,492 | 35,897 | +595 | +1.66% | skill 1/1; base 1/1 |
| claude-code | negative-pytorch-install | 35,925 | 35,599 | +326 | +0.92% | skill 1/1; base 1/1 |
| codex | All cases | 5,647,197 | 5,878,348 | -231,151 | -3.93% | skill 20/20; base 20/20 |
| codex | cuda-python-any-array | 696,485 | 555,273 | +141,212 | +25.43% | skill 1/1; base 1/1 |
| codex | cuda-python-core-coverage-question | 67,663 | 56,539 | +11,124 | +19.67% | skill 1/1; base 1/1 |
| codex | cuda-python-fix-hardcoded-paths | 310,399 | 341,299 | -30,900 | -9.05% | skill 1/1; base 1/1 |
| codex | cuda-python-fix-ignored-errors | 554,577 | 480,780 | +73,797 | +15.35% | skill 1/1; base 1/1 |
| codex | cuda-python-fix-scalar-args | 174,532 | 201,057 | -26,525 | -13.19% | skill 1/1; base 1/1 |
| codex | cuda-python-fix-stream-order | 167,752 | 411,975 | -244,223 | -59.28% | skill 1/1; base 1/1 |
| codex | cuda-python-gpu-report | 300,411 | 236,267 | +64,144 | +27.15% | skill 1/1; base 1/1 |
| codex | cuda-python-install-question | 47,753 | 33,153 | +14,600 | +44.04% | skill 1/1; base 1/1 |
| codex | cuda-python-kernel-timing | 301,015 | 141,423 | +159,592 | +112.85% | skill 1/1; base 1/1 |
| codex | cuda-python-migration-question | 45,051 | 35,257 | +9,794 | +27.78% | skill 1/1; base 1/1 |
| codex | cuda-python-perf-launch-loop | 432,239 | 486,179 | -53,940 | -11.09% | skill 1/1; base 1/1 |
| codex | cuda-python-port-core-experimental | 872,445 | 1,001,295 | -128,850 | -12.87% | skill 1/1; base 1/1 |
| codex | cuda-python-port-legacy-bindings | 725,616 | 897,704 | -172,088 | -19.17% | skill 1/1; base 1/1 |
| codex | cuda-python-port-pycuda | 293,928 | 341,115 | -47,187 | -13.83% | skill 1/1; base 1/1 |
| codex | cuda-python-review-calibrate | 302,505 | 367,867 | -65,362 | -17.77% | skill 1/1; base 1/1 |
| codex | cuda-python-saxpy-numpy | 213,515 | 161,230 | +52,285 | +32.43% | skill 1/1; base 1/1 |
| codex | cuda-python-strided-view-question | 54,682 | 48,896 | +5,786 | +11.83% | skill 1/1; base 1/1 |
| codex | cuda-python-struct-arg-question | 46,762 | 41,636 | +5,126 | +12.31% | skill 1/1; base 1/1 |
| codex | negative-cupy-arrays | 17,855 | 17,721 | +134 | +0.76% | skill 1/1; base 1/1 |
| codex | negative-pytorch-install | 22,012 | 21,682 | +330 | +1.52% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 14,134,544 | 12,410,189 | +1,724,355 | +13.89% | skill 40/40; base 40/40 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 10 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 20 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_discoverability: Description contains vague words (`skills/cuda-python/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/cuda-python/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/cuda-python/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (760 chars, recommend 50-150) (`skills/cuda-python/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/cuda-python/SKILL.md`)
- 5 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
