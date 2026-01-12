# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from cuda.bindings import utils
from cuda.bindings._version import __version__

assert tuple(int(_) for _ in __version__.split(".")[:2]) > (0, 1), "FATAL: invalid __version__"
