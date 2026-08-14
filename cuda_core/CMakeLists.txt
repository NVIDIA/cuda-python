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

set(CUDA_PYTHON_COVERAGE "$ENV{CUDA_PYTHON_COVERAGE}" CACHE STRING "Build Cython line tracing support")

if(DEFINED ENV{CUDA_PATH} AND NOT "$ENV{CUDA_PATH}" STREQUAL "")
  set(_cuda_core_cuda_root "$ENV{CUDA_PATH}")
elseif(DEFINED ENV{CUDA_HOME} AND NOT "$ENV{CUDA_HOME}" STREQUAL "")
  set(_cuda_core_cuda_root "$ENV{CUDA_HOME}")
endif()
if(NOT _cuda_core_cuda_root)
  message(FATAL_ERROR "CUDA_PATH or CUDA_HOME must point to a CUDA Toolkit installation")
endif()

if(DEFINED ENV{CUDA_CORE_BUILD_MAJOR} AND NOT "$ENV{CUDA_CORE_BUILD_MAJOR}" STREQUAL "")
  set(_cuda_core_build_major "$ENV{CUDA_CORE_BUILD_MAJOR}")
endif()

cmake_path(ABSOLUTE_PATH _cuda_core_cuda_root NORMALIZE)
set(_cuda_header "${_cuda_core_cuda_root}/include/cuda.h")
if(NOT EXISTS "${_cuda_header}")
  message(FATAL_ERROR "CUDA header not found: ${_cuda_header}")
endif()

file(STRINGS "${_cuda_header}" _cuda_version_line REGEX "^#[ \t]*define[ \t]+CUDA_VERSION[ \t]+[0-9]+[ \t]*$")
if(NOT _cuda_version_line)
  message(FATAL_ERROR "Cannot determine CUDA major version from ${_cuda_header}")
endif()
string(REGEX MATCH "[0-9]+" _cuda_version "${_cuda_version_line}")
math(EXPR _cuda_header_major "${_cuda_version} / 1000")

if("${_cuda_core_build_major}" STREQUAL "")
  set(_cuda_core_build_major "${_cuda_header_major}")
elseif(NOT _cuda_core_build_major MATCHES "^[0-9]+$")
  message(FATAL_ERROR "CUDA_CORE_BUILD_MAJOR must be an integer, got ${_cuda_core_build_major}")
elseif(NOT "${_cuda_core_build_major}" STREQUAL "${_cuda_header_major}")
  message(
    FATAL_ERROR
    "CUDA_CORE_BUILD_MAJOR=${_cuda_core_build_major} does not match CUDA ${_cuda_header_major} "
    "reported by ${_cuda_header}"
  )
endif()
message(STATUS "Building cuda-core for CUDA ${_cuda_core_build_major} using ${_cuda_core_cuda_root}")

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
  -3
  -Werror
  -X embedsignature=True
  -X warn.deprecated.IF=False
  -X freethreading_compatible=True
  -E CUDA_CORE_BUILD_MAJOR=${_cuda_core_build_major}
  -I "${CMAKE_CURRENT_SOURCE_DIR}"
  ${_coverage_directive}
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT WIN32)
  list(APPEND _cython_common_args --gdb)
endif()

add_library(cuda_core_extension_settings INTERFACE)
target_include_directories(
  cuda_core_extension_settings
  INTERFACE
    "${_cuda_core_source_root}"
    "${_cuda_core_source_root}/_include"
    "${_cuda_core_cuda_root}/include"
)
target_compile_definitions(
  cuda_core_extension_settings
  INTERFACE
    "$<$<AND:$<CONFIG:Debug>,$<PLATFORM_ID:Linux>,$<CXX_COMPILER_ID:GNU,Clang>>:_GLIBCXX_ASSERTIONS>"
)
if(CUDA_PYTHON_COVERAGE)
  target_compile_definitions(
    cuda_core_extension_settings
    INTERFACE
      CYTHON_TRACE_NOGIL=1
      CYTHON_USE_SYS_MONITORING=0
  )
endif()
if(WIN32 AND _python_freethreaded)
  # Windows shares Python headers between GIL and free-threaded builds, so
  # extension builds must supply the ABI selection macro themselves.
  target_compile_definitions(cuda_core_extension_settings INTERFACE Py_GIL_DISABLED=1)
endif()

set(_cuda_core_modules
  _context
  _device
  _device_resources
  _dlpack
  _event
  _graphics
  _kernel_arg_handler
  _launch_config
  _launcher
  _layout
  _linker
  _memory/_buffer
  _memory/_copy_attributes
  _memory/_copy_ops
  _memory/_device_memory_resource
  _memory/_graph_memory_resource
  _memory/_ipc
  _memory/_managed_memory_ops
  _memory/_managed_memory_resource
  _memory/_memory_pool
  _memory/_peer_access_utils
  _memory/_pinned_memory_resource
  _memoryview
  _module
  _program
  _resource_handles
  _stream
  _tensor_bridge
  _tensor_map
  _utils/_weak_handles
  _utils/_wsl_locale
  _utils/cuda_utils
  _utils/version
  graph/_adjacency_set_proxy
  graph/_graph_builder
  graph/_graph_definition
  graph/_graph_node
  graph/_host_callback
  graph/_subclasses
  system/_device
  system/_nvml_context
  system/_system
  system/_system_events
  texture/_array
  texture/_mipmapped_array
  texture/_surface
  texture/_texture
)

file(
  GLOB_RECURSE _cuda_core_module_sources
  CONFIGURE_DEPENDS
  RELATIVE "${_cuda_core_source_root}"
  "${_cuda_core_source_root}/*.pyx"
)
list(TRANSFORM _cuda_core_module_sources REPLACE "\\.pyx$" "")
set(_cuda_core_listed_modules ${_cuda_core_modules})
list(SORT _cuda_core_module_sources)
list(SORT _cuda_core_listed_modules)
if(NOT "${_cuda_core_module_sources}" STREQUAL "${_cuda_core_listed_modules}")
  message(FATAL_ERROR "Update _cuda_core_modules to match the cuda.core .pyx files")
endif()

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
endif()

if(MSVC)
  set(_aoti_shim_def "${_cuda_core_source_root}/_include/aoti_shim.def")
  set(_aoti_shim_lib "${CMAKE_CURRENT_BINARY_DIR}/aoti_shim.lib")
  set(_aoti_shim_exp "${CMAKE_CURRENT_BINARY_DIR}/aoti_shim.exp")
  if(CMAKE_GENERATOR_PLATFORM MATCHES "^(ARM64|arm64)$" OR CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64)$")
    set(_aoti_machine ARM64)
  else()
    set(_aoti_machine X64)
  endif()
  add_custom_command(
    OUTPUT "${_aoti_shim_lib}"
    BYPRODUCTS "${_aoti_shim_exp}"
    COMMAND
      "${CMAKE_AR}"
      "/DEF:${_aoti_shim_def}"
      "/OUT:${_aoti_shim_lib}"
      "/MACHINE:${_aoti_machine}"
    DEPENDS "${_aoti_shim_def}"
    COMMENT "Generating the AOTI shim import library"
    VERBATIM
  )
  add_custom_target(cuda_core_aoti_shim_generation DEPENDS "${_aoti_shim_lib}")
  add_library(cuda_core_aoti_shim SHARED IMPORTED)
  set_target_properties(cuda_core_aoti_shim PROPERTIES IMPORTED_IMPLIB "${_aoti_shim_lib}")
  add_dependencies(cuda_core_aoti_shim cuda_core_aoti_shim_generation)
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
  if(module_path STREQUAL "_resource_handles")
    list(APPEND module_sources "${_cuda_core_source_root}/_cpp/resource_handles.cpp")
  elseif(module_path STREQUAL "_tensor_map")
    list(APPEND module_sources "${_cuda_core_source_root}/_cpp/tensor_map.cpp")
  endif()

  python_add_library("${target_name}" MODULE WITH_SOABI ${module_sources})
  set_target_properties(
    "${target_name}"
    PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "${module_output_directory}"
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS OFF
      CXX_VISIBILITY_PRESET hidden
      LIBRARY_OUTPUT_DIRECTORY "${module_output_directory}"
      OUTPUT_NAME "${module_filename}"
      RUNTIME_OUTPUT_DIRECTORY "${module_output_directory}"
      VISIBILITY_INLINES_HIDDEN YES
  )
  target_link_libraries("${target_name}" PRIVATE cuda_core_extension_settings)

  if(CUDA_PYTHON_COVERAGE)
    install(FILES "${generated_cpp}" DESTINATION "${install_directory}")
  endif()

  if(MSVC AND module_path STREQUAL "_tensor_bridge")
    target_link_libraries("${target_name}" PRIVATE cuda_core_aoti_shim)
  endif()

  install(TARGETS "${target_name}" LIBRARY DESTINATION "${install_directory}" RUNTIME DESTINATION "${install_directory}")
endfunction()

foreach(module_path IN LISTS _cuda_core_modules)
  if(WIN32 AND module_path STREQUAL "_utils/_wsl_locale")
    continue()
  endif()
  cuda_core_add_cython_module("${module_path}")
endforeach()
