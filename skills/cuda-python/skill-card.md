## Description: <br>
Use for any task or question involving CUDA Python Core libraries (cuda-core, cuda-bindings and cuda-pathfinder), including writing, porting, reviewing and debugging Python code that compiles and launches its own CUDA kernels, porting PyCUDA or old CUDA bindings APIs, passing CuPy, PyTorch or Numba arrays and streams to custom kernels, locating CUDA headers or libraries, and installing cuda-python, cuda-core or conda packages; not for plain CuPy, Numba or PyTorch array code, or for cuda.compute (CCCL). <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to have an agent write, port, review, and debug Python code built on cuda.core, cuda.bindings, and cuda.pathfinder, including runtime kernel compilation and launch, CUDA graphs, interop with CuPy/PyTorch/Numba, migration from PyCUDA or legacy bindings, and installation guidance. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [cuda.core documentation](https://nvidia.github.io/cuda-python/cuda-core/latest/) <br>
- [NVIDIA/cuda-python source repository](https://github.com/NVIDIA/cuda-python) <br>
- [BENCHMARK.md](BENCHMARK.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline Python and shell code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
20 evaluation tasks (18 positive, 2 negative), 1 attempt per task, run in the `k8s-sandbox` environment with each task attempt in its own isolated sandbox pod; baseline is the same task attempted without the target skill (evaluator version `1.6.9`, evaluation date 2026-11-17). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from `security` (100%). <br>
- Correctness: Is the answer correct? Scored from `accuracy` (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from `skill_execution` (100%). <br>
- Effectiveness: Did the skill help complete the task? Scored from `goal_accuracy` (50%) + `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Scored from `skill_efficiency` (50%) + `token_efficiency` (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Security: unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Skill Execution: whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool Productivity: tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Accuracy: final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Goal Accuracy: whether the user's goal was achieved. <br>
- `behavior_check`: Behavior Check: whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Token Efficiency: actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
> ✅ **Overall verdict: PASS — Recommended for publication**

### Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

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

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 93.5% → 93.2% (-0.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 80.4% → 84.2% (+3.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Skill Version(s): <br>
f53f3d97 (source: git SHA, committed 2026-11-14) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
