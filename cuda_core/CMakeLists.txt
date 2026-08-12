# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.21...4.4)

project(cuda_core LANGUAGES CXX)

find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED
)
find_package(Cython 3.2.5...<3.3 MODULE REQUIRED)
include(UseCython)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CUDA_CORE_CUDA_ROOT "" CACHE PATH "CUDA Toolkit root used to compile cuda-core")
set(CUDA_CORE_BUILD_MAJOR "" CACHE STRING "CUDA major version for Cython compile-time conditionals")
set(CUDA_PYTHON_COVERAGE "$ENV{CUDA_PYTHON_COVERAGE}" CACHE STRING "Build Cython line tracing support")

if(NOT CUDA_CORE_CUDA_ROOT)
  if(DEFINED ENV{CUDA_HOME} AND NOT "$ENV{CUDA_HOME}" STREQUAL "")
    set(CUDA_CORE_CUDA_ROOT "$ENV{CUDA_HOME}")
  elseif(DEFINED ENV{CUDA_PATH} AND NOT "$ENV{CUDA_PATH}" STREQUAL "")
    set(CUDA_CORE_CUDA_ROOT "$ENV{CUDA_PATH}")
  endif()
endif()
if(NOT CUDA_CORE_CUDA_ROOT)
  message(FATAL_ERROR "CUDA_CORE_CUDA_ROOT, CUDA_HOME, or CUDA_PATH must point to a CUDA Toolkit installation")
endif()

cmake_path(ABSOLUTE_PATH CUDA_CORE_CUDA_ROOT NORMALIZE)
set(_cuda_header "${CUDA_CORE_CUDA_ROOT}/include/cuda.h")
if(NOT EXISTS "${_cuda_header}")
  message(FATAL_ERROR "CUDA header not found: ${_cuda_header}")
endif()

if(NOT CUDA_CORE_BUILD_MAJOR)
  file(STRINGS "${_cuda_header}" _cuda_version_line REGEX "^#[ \t]*define[ \t]+CUDA_VERSION[ \t]+[0-9]+[ \t]*$")
  if(NOT _cuda_version_line)
    message(FATAL_ERROR "Cannot determine CUDA major version from ${_cuda_header}; set CUDA_CORE_BUILD_MAJOR")
  endif()
  string(REGEX MATCH "[0-9]+" _cuda_version "${_cuda_version_line}")
  math(EXPR CUDA_CORE_BUILD_MAJOR "${_cuda_version} / 1000")
endif()
if(NOT CUDA_CORE_BUILD_MAJOR MATCHES "^[0-9]+$")
  message(FATAL_ERROR "CUDA_CORE_BUILD_MAJOR must be an integer, got ${CUDA_CORE_BUILD_MAJOR}")
endif()
message(STATUS "Building cuda-core for CUDA ${CUDA_CORE_BUILD_MAJOR} using ${CUDA_CORE_CUDA_ROOT}")

execute_process(
  COMMAND
    "${Python_EXECUTABLE}" -c
    "from pathlib import Path; import cuda.bindings; print(Path(cuda.bindings.__file__).resolve().parents[2])"
  RESULT_VARIABLE _cuda_bindings_result
  OUTPUT_VARIABLE _cuda_bindings_package_root
  ERROR_VARIABLE _cuda_bindings_error
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _cuda_bindings_result STREQUAL "0" OR NOT EXISTS "${_cuda_bindings_package_root}/cuda/bindings")
  message(
    FATAL_ERROR
    "Could not resolve the physical cuda.bindings package root needed by Cython. "
    "Ensure the CUDA-major-compatible cuda-bindings build dependency is installed.\n${_cuda_bindings_error}"
  )
endif()
message(STATUS "Using cuda-bindings Cython declarations from ${_cuda_bindings_package_root}")

if(WIN32)
  execute_process(
    COMMAND
      "${Python_EXECUTABLE}" -c
      "import sysconfig; print(int(bool(sysconfig.get_config_var('Py_GIL_DISABLED'))))"
    RESULT_VARIABLE _python_freethreaded_result
    OUTPUT_VARIABLE _python_freethreaded
    ERROR_VARIABLE _python_freethreaded_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT _python_freethreaded_result STREQUAL "0")
    message(FATAL_ERROR "Could not determine whether Python is free-threaded.\n${_python_freethreaded_error}")
  endif()
endif()

if(CUDA_PYTHON_COVERAGE STREQUAL "")
  set(CUDA_PYTHON_COVERAGE OFF)
endif()
if(CUDA_PYTHON_COVERAGE)
  set(_coverage_directive -X linetrace=True)
endif()

if(WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(FATAL_ERROR "Debuggable builds are not supported on Windows")
endif()

set(_cuda_core_source_root "${CMAKE_CURRENT_SOURCE_DIR}/cuda/core")
set(_cython_common_args
  --cplus
  -3
  -Werror
  -X embedsignature=True
  -X warn.deprecated.IF=False
  -X freethreading_compatible=True
  -E CUDA_CORE_BUILD_MAJOR=${CUDA_CORE_BUILD_MAJOR}
  -I "${CMAKE_CURRENT_SOURCE_DIR}"
  -I "${_cuda_bindings_package_root}"
  ${_coverage_directive}
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT WIN32)
  list(APPEND _cython_common_args --gdb)
endif()

file(
  GLOB_RECURSE _cuda_core_pyx_files
  CONFIGURE_DEPENDS
  RELATIVE "${_cuda_core_source_root}"
  "${_cuda_core_source_root}/*.pyx"
)

# Stage regular wheels through CMake to keep them independent of ignored,
# in-place extension artifacts. Editable builds map these files to the live
# source tree instead.
if(NOT SKBUILD_STATE STREQUAL "editable")
  install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cuda/"
    DESTINATION cuda
    FILES_MATCHING
      PATTERN "*.h"
      PATTERN "*.hpp"
      PATTERN "*.pxd"
      PATTERN "*.py"
      PATTERN "*.pyi"
      PATTERN "py.typed"
  )
  if(CUDA_PYTHON_COVERAGE)
    install(
      DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cuda/"
      DESTINATION cuda
      FILES_MATCHING
        PATTERN "*.cpp"
        PATTERN "*.pxi"
        PATTERN "*.pyx"
    )
  endif()
else()
  # Cython resolves external .pxd files from physical sys.path entries, not
  # PEP 660 meta-path finders. Preserve the bindings declaration path for
  # downstream Cython consumers of an editable cuda-core installation.
  set(_cuda_bindings_cython_pth "${CMAKE_CURRENT_BINARY_DIR}/_cuda_core_cython_include.pth")
  file(WRITE "${_cuda_bindings_cython_pth}" "${_cuda_bindings_package_root}\n")
  install(FILES "${_cuda_bindings_cython_pth}" DESTINATION ".")
endif()

if(MSVC)
  set(_aoti_shim_def "${_cuda_core_source_root}/_include/aoti_shim.def")
  set(_aoti_shim_lib "${CMAKE_CURRENT_BINARY_DIR}/aoti_shim.lib")
  if(CMAKE_GENERATOR_PLATFORM MATCHES "^(ARM64|arm64)$" OR CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64)$")
    set(_aoti_machine ARM64)
  else()
    set(_aoti_machine X64)
  endif()
  add_custom_command(
    OUTPUT "${_aoti_shim_lib}"
    COMMAND
      "${CMAKE_AR}"
      "/DEF:${_aoti_shim_def}"
      "/OUT:${_aoti_shim_lib}"
      "/MACHINE:${_aoti_machine}"
    DEPENDS "${_aoti_shim_def}"
    COMMENT "Generating the AOTI shim import library"
    VERBATIM
  )
  add_custom_target(cuda_core_aoti_shim DEPENDS "${_aoti_shim_lib}")
endif()

function(cuda_core_add_cython_module module_path)
  string(REPLACE "/" "." module_suffix "${module_path}")
  string(REPLACE "/" "_" target_suffix "${module_path}")
  set(module_name "cuda.core.${module_suffix}")
  set(target_name "cuda_core${target_suffix}")
  set(pyx_file "${_cuda_core_source_root}/${module_path}.pyx")
  set(cython_cpp "${CMAKE_CURRENT_BINARY_DIR}/cython/${module_path}.cpp")
  cmake_path(GET cython_cpp PARENT_PATH cython_output_directory)
  file(MAKE_DIRECTORY "${cython_output_directory}")
  cmake_path(GET module_path FILENAME module_filename)
  cmake_path(GET module_path PARENT_PATH module_directory)
  if(module_directory)
    set(install_directory "cuda/core/${module_directory}")
    set(module_output_directory "${CMAKE_CURRENT_BINARY_DIR}/extensions/${module_directory}")
  else()
    set(install_directory "cuda/core")
    set(module_output_directory "${CMAKE_CURRENT_BINARY_DIR}/extensions")
  endif()

  cython_transpile(
    "${pyx_file}"
    LANGUAGE CXX
    OUTPUT "${cython_cpp}"
    OUTPUT_VARIABLE generated_cpp
    CYTHON_ARGS ${_cython_common_args} --module-name "${module_name}"
  )

  set(module_sources "${generated_cpp}")
  string(REGEX REPLACE "^_" "" helper_cpp_path "${module_path}")
  set(extra_cpp "${_cuda_core_source_root}/_cpp/${helper_cpp_path}.cpp")
  if(EXISTS "${extra_cpp}")
    list(APPEND module_sources "${extra_cpp}")
  endif()

  python_add_library("${target_name}" MODULE WITH_SOABI ${module_sources})
  set_target_properties(
    "${target_name}"
    PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "${module_output_directory}"
      LIBRARY_OUTPUT_DIRECTORY "${module_output_directory}"
      OUTPUT_NAME "${module_filename}"
      RUNTIME_OUTPUT_DIRECTORY "${module_output_directory}"
  )
  target_include_directories(
    "${target_name}"
    PRIVATE
      "${_cuda_core_source_root}"
      "${_cuda_core_source_root}/_include"
      "${_cuda_core_source_root}/_cpp"
      "${CUDA_CORE_CUDA_ROOT}/include"
  )

  if(MSVC)
    target_compile_options("${target_name}" PRIVATE /std:c++17)
  else()
    target_compile_options(
      "${target_name}"
      PRIVATE
        $<$<CONFIG:Debug>:-g;-O0>
        $<$<NOT:$<CONFIG:Debug>>:-O2>
    )
    target_compile_definitions("${target_name}" PRIVATE $<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)
    target_link_options("${target_name}" PRIVATE $<$<NOT:$<CONFIG:Debug>>:-Wl,--strip-all>)
  endif()

  if(CUDA_PYTHON_COVERAGE)
    target_compile_definitions(
      "${target_name}"
      PRIVATE
        CYTHON_TRACE_NOGIL=1
        CYTHON_USE_SYS_MONITORING=0
    )
    install(FILES "${generated_cpp}" DESTINATION "${install_directory}")
  endif()

  if(WIN32 AND _python_freethreaded)
    # Windows shares Python headers between GIL and free-threaded builds, so
    # extension builds must supply the ABI selection macro themselves.
    target_compile_definitions("${target_name}" PRIVATE Py_GIL_DISABLED=1)
  endif()

  if(MSVC AND module_path STREQUAL "_tensor_bridge")
    add_dependencies("${target_name}" cuda_core_aoti_shim)
    target_link_libraries("${target_name}" PRIVATE "${_aoti_shim_lib}")
  endif()

  install(TARGETS "${target_name}" LIBRARY DESTINATION "${install_directory}" RUNTIME DESTINATION "${install_directory}")
endfunction()

foreach(pyx_file IN LISTS _cuda_core_pyx_files)
  string(REGEX REPLACE "\\.pyx$" "" module_path "${pyx_file}")
  if(WIN32 AND module_path STREQUAL "_utils/_wsl_locale")
    continue()
  endif()
  cuda_core_add_cython_module("${module_path}")
endforeach()
