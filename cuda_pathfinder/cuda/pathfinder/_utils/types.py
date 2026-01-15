# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Type aliases for pathfinder utilities."""

from typing import Callable, Optional, Sequence

# Type alias for functions that generate filename variants
FilenameVariantFunc = Callable[[str], Sequence[str]]

# Type alias for functions that return a base directory
BaseDirFunc = Callable[[], Optional[str]]
