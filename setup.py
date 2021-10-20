# Copyright 2021 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
import os
import shutil
import sys
import sysconfig

from setuptools import find_packages, setup
from setuptools.extension import Extension
from Cython.Build import cythonize

from distutils.sysconfig import get_python_lib
import versioneer

install_requires = ["cython"]

try:
    nthreads = int(os.environ.get("PARALLEL_LEVEL", "0") or "0")
except Exception:
    nthreads = 0

include_dirs = [
    os.path.dirname(sysconfig.get_path("include")),
]

library_dirs = [get_python_lib(), os.path.join(os.sys.prefix, "lib")]

extra_cythonize_kwargs = {}
if sys.platform == 'win32':
    extra_compile_args = []
else:
    extra_compile_args = [
        '-std=c++14',
        '-fpermissive',
        '-Wno-deprecated-declarations',
        '-D _GLIBCXX_ASSERTIONS',
        '-fno-var-tracking-assignments'
    ]
    if '--debug' in sys.argv:
        extra_cythonize_kwargs['gdb_debug'] = True
        extra_compile_args += ['-g', '-O0']
    else:
        extra_compile_args += ['-O3']

# private
extensions = cythonize(
    [
        Extension(
            "*",
            sources=["cuda/_cuda/*.pyx", "cuda/_cuda/loader.cpp"],
            include_dirs=[],
            library_dirs=[],
            runtime_library_dirs=[],
            libraries=[],
            language="c++",
            extra_compile_args=extra_compile_args,
        )
    ],
    nthreads=nthreads,
    compiler_directives=dict(
        profile=True, language_level=3, embedsignature=True, binding=True
    ),
    **extra_cythonize_kwargs
)

# utils
extensions += cythonize(
    [
        Extension(
            "*",
            sources=["cuda/_lib/*.pyx", "cuda/_lib/param_packer.cpp"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            runtime_library_dirs=[],
            libraries=[],
            language="c++",
            extra_compile_args=extra_compile_args,
        )
    ],
    nthreads=nthreads,
    compiler_directives=dict(
        profile=True, language_level=3, embedsignature=True, binding=True
    ),
    **extra_cythonize_kwargs
)
extensions += cythonize(
    [
        Extension(
            "*",
            sources=["cuda/_lib/ccudart/*.pyx"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            runtime_library_dirs=[],
            libraries=[],
            language="c++",
            extra_compile_args=extra_compile_args,
        )
    ],
    nthreads=nthreads,
    compiler_directives=dict(
        profile=True, language_level=3, embedsignature=True, binding=True
    ),
    **extra_cythonize_kwargs
)

# public
extensions += cythonize(
    [
        Extension(
            "*",
            sources=["cuda/*.pyx"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            runtime_library_dirs=[],
            libraries=[],
            language="c++",
            extra_compile_args=extra_compile_args,
        )
    ],
    nthreads=nthreads,
    compiler_directives=dict(
        profile=True, language_level=3, embedsignature=True, binding=True
    ),
    **extra_cythonize_kwargs
)

# tests:
extensions += cythonize(
    [
        Extension(
            "*",
            sources=["cuda/tests/*.pyx"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            runtime_library_dirs=[],
            libraries=[],
            language="c++",
            extra_compile_args=["-std=c++14"],
        )
    ],
    nthreads=nthreads,
    compiler_directives=dict(
        profile=True, language_level=3, embedsignature=True, binding=True
    ),
)

setup(
    name="cuda-python",
    version=versioneer.get_version(),
    description="Python bindings for CUDA",
    url="https://github.com/NVIDIA/cuda-python",
    author="NVIDIA Corporation",
    author_email="cuda-python-conduct@nvidia.com",
    license="Other",
    license_files = ('LICENSE',),
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Scientific/Engineering",
        "License :: Other/Proprietary License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Environment :: GPU :: NVIDIA CUDA",
        "Environment :: GPU :: NVIDIA CUDA :: 11.0",
        "Environment :: GPU :: NVIDIA CUDA :: 11.1",
        "Environment :: GPU :: NVIDIA CUDA :: 11.2",
        "Environment :: GPU :: NVIDIA CUDA :: 11.3",
        "Environment :: GPU :: NVIDIA CUDA :: 11.4",
        "Environment :: GPU :: NVIDIA CUDA :: 11.5",
    ],
    # Include the separately-compiled shared library
    setup_requires=["cython"],
    ext_modules=extensions,
    packages=find_packages(include=["cuda", "cuda.*"]),
    package_data=dict.fromkeys(
        find_packages(include=["cuda", "cuda.*"]),
        ["*.pxd", "*.pyx", "*.h", "*.cpp"],
    ),
    cmdclass=versioneer.get_cmdclass(),
    install_requires=install_requires,
    zip_safe=False,
)
