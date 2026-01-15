# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-platform filename resolution for CUDA artifacts."""

from cuda.pathfinder._utils.platform_aware import IS_WINDOWS


class FilenameResolver:
    """Resolves artifact names to platform-specific filenames.
    
    This class handles the platform-specific naming conventions for CUDA
    artifacts, allowing the search logic to remain platform-agnostic.
    
    Examples:
        >>> # Binary resolution
        >>> FilenameResolver.for_binary("nvcc")
        ('nvcc', 'nvcc.exe')
        
        >>> # Static library resolution on Linux
        >>> FilenameResolver.for_static_lib("cudadevrt")
        ('libcudadevrt.a',)
        
        >>> # Static library resolution on Windows
        >>> FilenameResolver.for_static_lib("cudadevrt")
        ('cudadevrt.lib',)
        
        >>> # Files with extensions remain unchanged
        >>> FilenameResolver.for_static_lib("libdevice.10.bc")
        ('libdevice.10.bc',)
    """

    @staticmethod
    def for_binary(name: str) -> tuple[str, ...]:
        """Generate platform-appropriate binary filename variants.
        
        Returns both the exact name and Windows .exe variant to support
        cross-platform fuzzy search. The filesystem naturally filters to
        what exists.
        
        Args:
            name: Base binary name (e.g., "nvcc", "nvdisasm").
            
        Returns:
            Tuple of possible filenames. Always includes exact name first,
            then .exe variant.
        """
        # Try exact name first, then with .exe extension
        # This works across platforms - non-existent files won't be found
        return (name, f"{name}.exe")

    @staticmethod
    def for_static_lib(name: str) -> tuple[str, ...]:
        """Generate platform-appropriate static library filename variants.
        
        Handles platform-specific naming conventions for static libraries
        and preserves files that already have extensions (like .bc bitcode).
        
        Args:
            name: Canonical artifact name (e.g., "cudadevrt", "libdevice.10.bc").
            
        Returns:
            Tuple of possible filenames for the current platform.
            
        Examples:
            On Linux:
                "cudadevrt" -> ("libcudadevrt.a",)
                "libdevice.10.bc" -> ("libdevice.10.bc",)
            On Windows:
                "cudadevrt" -> ("cudadevrt.lib",)
                "libdevice.10.bc" -> ("libdevice.10.bc",)
        """
        # Files that already have extensions (e.g., .bc bitcode files)
        # are the same on all platforms
        if "." in name:
            return (name,)

        # Platform-specific library naming conventions
        if IS_WINDOWS:
            return (f"{name}.lib",)
        else:
            return (f"lib{name}.a",)
