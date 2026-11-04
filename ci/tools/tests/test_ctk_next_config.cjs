// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const { parseKitpick, parseCybindSha, resolveConfig, resolveActionValues } = require("../ctk_next_config.cjs");

const cybindSha = "4359948d95d5aaba056be96622f21924cfa1c58d";

const root = path.resolve(__dirname, "../../..");
const action = fs.readFileSync(path.join(root, ".github/actions/resolve-ctk-next/action.yml"), "utf8");

// Agent-authored tests (gpt-6); Node tests do not use pytest provenance markers.
test("kitpick parsing preserves the suffix and derives full, short, and major versions", () => {
  assert.deepEqual(parseKitpick("13.5.0/016"), {
    ctkKitpick: "13.5.0/016",
    cudaVersion: "13.5",
    cudaKitpickVersion: "13.5.0",
    cudaKitpick: "016",
    cudaMajor: "13",
  });
  assert.deepEqual(parseKitpick("14.0.1/002"), {
    ctkKitpick: "14.0.1/002",
    cudaVersion: "14.0",
    cudaKitpickVersion: "14.0.1",
    cudaKitpick: "002",
    cudaMajor: "14",
  });
});

for (const invalid of [
  null, undefined, 13.5, {}, [], "", "13.5/022", "13.5.0/22", "13.5.0/0022",
  "013.5.0/022", "13.05.0/022", "13.5.00/022", "0.5.0/022",
  "cuda-r13-5/13.5.0/022", "13.5.0/022\n", "13.5.0/022\r\n", " 13.5.0/022",
  "13.5.0/022 ", "13.5.0/022;echo injected",
]) {
  test(`invalid kitpick is rejected: ${JSON.stringify(invalid)}`, () => {
    assert.throws(() => parseKitpick(invalid), /major\.minor\.patch\/NNN/);
  });
}

test("override selection validates the file and never mutates its config", () => {
  const config = { ctk_kitpick: "13.5.0/016", cybind_sha: cybindSha };
  assert.deepEqual(resolveConfig(config), { ...parseKitpick("13.5.0/016"), cybindSha });
  assert.deepEqual(resolveConfig(config, "13.5.0/022"), { ...parseKitpick("13.5.0/022"), cybindSha });
  assert.deepEqual(config, { ctk_kitpick: "13.5.0/016", cybind_sha: cybindSha });
  assert.throws(() => resolveConfig({ ...config, ctk_kitpick: "invalid" }, "13.5.0/022"), /major\.minor/);
  assert.throws(() => resolveConfig({ ...config, cybind_sha: "main" }, "13.5.0/022"), /40-character lowercase/);
  assert.throws(() => resolveConfig(config, null), /major\.minor/);
});

test("cybind parsing preserves an exact SHA and the checked-in configuration resolves", () => {
  assert.equal(parseCybindSha(cybindSha), cybindSha);
  const config = JSON.parse(fs.readFileSync(path.join(root, "ci/ctk-next.json"), "utf8"));
  assert.deepEqual(resolveConfig(config), { ...parseKitpick(config.ctk_kitpick), cybindSha: config.cybind_sha });
});

for (const invalid of [
  null, undefined, 123, {}, [], "", "main", cybindSha.slice(0, 39), `${cybindSha}0`,
  cybindSha.toUpperCase(), `g${cybindSha.slice(1)}`, `${cybindSha}\n`, `${cybindSha}\r\n`,
  ` ${cybindSha}`, `${cybindSha} `, `${cybindSha};echo injected`,
]) {
  test(`invalid cybind SHA is rejected: ${JSON.stringify(invalid)}`, () => {
    assert.throws(() => parseCybindSha(invalid), /40-character lowercase/);
    assert.throws(() => resolveConfig({ ctk_kitpick: "13.5.0/016", cybind_sha: invalid }), /40-character lowercase/);
  });
}

for (const invalid of [
  null, [], "13.5.0/016", {}, { ctk_kitpick: "13.5.0/016" }, { cybind_sha: cybindSha },
  { ctk_kitpick: "13.5.0/016", typo: true }, { cybind_sha: cybindSha, typo: true },
  { ctk_kitpick: "13.5.0/016", cybind_sha: cybindSha, typo: true },
]) {
  test(`invalid config schema is rejected: ${JSON.stringify(invalid)}`, () => {
    assert.throws(() => resolveConfig(invalid), /exactly two fields/);
  });
}

test("action values include every declared output and CUDA field", () => {
  const { outputs, environment } = resolveActionValues(
    `{\r\n  "ctk_kitpick": "13.5.0/022",\r\n  "cybind_sha": "${cybindSha}"\r\n}\r\n`,
  );
  assert.deepEqual(outputs, {
    "ctk-kitpick": "13.5.0/022",
    "cuda-version": "13.5",
    "cuda-kitpick-version": "13.5.0",
    "cuda-kitpick": "022",
  });
  assert.deepEqual(environment, {
    CUDA_VERSION: "13.5",
    CUDA_KITPICK_VERSION: "13.5.0",
    CUDA_KITPICK: "022",
    CUDA_MAJOR: "13",
  });
  for (const name of Object.keys(outputs)) {
    assert.ok(action.includes(`value: \${{ steps.resolve.outputs.${name} }}`));
  }
});

test("action values use the override when supplied", () => {
  const config = `{"ctk_kitpick":"13.5.0/016","cybind_sha":"${cybindSha}"}\n`;
  const { outputs, environment } = resolveActionValues(config, "14.0.1/003");
  assert.equal(outputs["ctk-kitpick"], "14.0.1/003");
  assert.equal(environment.CUDA_MAJOR, "14");
  assert.equal(environment.CUDA_KITPICK, "003");
});

test("workflow expressions stay out of the action's inline script", () => {
  const scriptStart = action.indexOf("script: |");
  assert.ok(scriptStart >= 0, "The resolver action must contain an inline script block.");
  assert.match(action, /CTK_NEXT_KITPICK_OVERRIDE: \$\{\{ inputs\.ctk-kitpick \}\}/);
  assert.ok(!action.slice(scriptStart).includes("${{"), "Workflow inputs must enter the script through the environment.");
});

for (const [label, config, override] of [
  ["malformed JSON", "{", ""],
  ["missing cybind SHA", '{"ctk_kitpick":"13.5.0/016"}', ""],
  ["invalid kitpick", `{"ctk_kitpick":null,"cybind_sha":"${cybindSha}"}`, ""],
  ["invalid cybind SHA", '{"ctk_kitpick":"13.5.0/016","cybind_sha":"main"}', "13.5.0/022"],
  ["invalid override", `{"ctk_kitpick":"13.5.0/016","cybind_sha":"${cybindSha}"}`, "13.5.0/22"],
]) {
  test(`action values reject ${label}`, () => {
    assert.throws(() => resolveActionValues(config, override));
  });
}

test("all six reusable jobs resolve their own checkout before consuming the toolkit", () => {
  let jobsChecked = 0;
  for (const file of ["ci-linux.yml", "ci-linux-wheel.yml", "ci-windows.yml", "ci-windows-wheel.yml"]) {
    const workflow = fs.readFileSync(path.join(root, ".github/workflows", file), "utf8");
    const jobs = workflow.split(/^  [a-z][a-z-]*:\n/gm).slice(1).filter((part) => part.includes("    steps:\n"));
    assert.equal(jobs.length, file.includes("wheel") ? 2 : 1);
    for (const job of jobs) {
      const checkout = job.indexOf("uses: actions/checkout@");
      const resolver = job.indexOf("uses: ./.github/actions/resolve-ctk-next");
      assert.ok(checkout >= 0 && resolver > checkout, `${file}: checkout must precede the resolver`);
      assert.equal(job.match(/uses: \.\/\.github\/actions\/resolve-ctk-next/g).length, 1);
      const consumer = job.indexOf(file.includes("windows")
        ? "uses: ./.github/actions/setup-ctk-next-windows"
        : "python3 \"$GITHUB_WORKSPACE/qa/forge/download_from_kitpicks.py\"");
      assert.ok(consumer > resolver, `${file}: toolkit setup must follow the resolver`);
      if (file.includes("windows")) {
        assert.ok(job.indexOf("name: Set up Git for Windows ARM64") < checkout);
      }
      jobsChecked++;
    }
    assert.doesNotMatch(workflow, /^\s+CUDA_(?:VERSION|KITPICK_VERSION|KITPICK):/m);
    assert.doesNotMatch(workflow, /cuda-13\.5|cuda_bindings-13\.|CUDA_CORE_BUILD_MAJOR[=:] *'?13/);
  }
  assert.equal(jobsChecked, 6);
});
