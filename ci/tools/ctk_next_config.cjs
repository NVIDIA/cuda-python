// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

"use strict";

function parseKitpick(value) {
  const match = typeof value === "string"
    ? /^([1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\/([0-9]{3})$/.exec(value)
    : null;
  if (!match || match[0] !== value) {
    throw new TypeError("CTK kitpick must be major.minor.patch/NNN, for example 13.5.0/022.");
  }
  const [, major, minor, patch, kitpick] = match;
  return {
    ctkKitpick: value,
    cudaVersion: `${major}.${minor}`,
    cudaKitpickVersion: `${major}.${minor}.${patch}`,
    cudaKitpick: kitpick,
    cudaMajor: major,
  };
}

function parseCybindSha(value) {
  if (typeof value !== "string" || value.length !== 40 || !/^[0-9a-f]{40}$/.test(value)) {
    throw new TypeError("Cybind SHA must be a full 40-character lowercase hexadecimal commit SHA.");
  }
  return value;
}

function resolveConfig(config, override = "") {
  if (!config || typeof config !== "object" || Array.isArray(config)
      || Object.keys(config).length !== 2
      || !Object.hasOwn(config, "ctk_kitpick") || !Object.hasOwn(config, "cybind_sha")) {
    throw new TypeError("ci/ctk-next.json must contain exactly two fields: ctk_kitpick and cybind_sha.");
  }
  const configured = parseKitpick(config.ctk_kitpick);
  const cybindSha = parseCybindSha(config.cybind_sha);
  return { ...(override === "" ? configured : parseKitpick(override)), cybindSha };
}

function resolveActionValues(configText, override = "") {
  const resolved = resolveConfig(JSON.parse(configText), override);
  return {
    outputs: {
      "ctk-kitpick": resolved.ctkKitpick,
      "cuda-version": resolved.cudaVersion,
      "cuda-kitpick-version": resolved.cudaKitpickVersion,
      "cuda-kitpick": resolved.cudaKitpick,
    },
    environment: {
      CUDA_VERSION: resolved.cudaVersion,
      CUDA_KITPICK_VERSION: resolved.cudaKitpickVersion,
      CUDA_KITPICK: resolved.cudaKitpick,
      CUDA_MAJOR: resolved.cudaMajor,
    },
  };
}

module.exports = { parseKitpick, parseCybindSha, resolveConfig, resolveActionValues };
