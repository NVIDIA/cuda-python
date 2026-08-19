# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build cuda_bindings Cython test extensions."""

from cuda_python_test_helpers.cython_test_builder import build_cython_tests


def main() -> None:
    build_cython_tests(
        script_file=__file__,
        distribution_name="cuda_bindings_cython_tests",
        nthreads=1,
    )


if __name__ == "__main__":
    main()
