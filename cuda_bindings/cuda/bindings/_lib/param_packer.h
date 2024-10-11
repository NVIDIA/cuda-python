// Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
#pragma once
#include <Python.h>

int feed(void* ptr,  PyObject* value, PyObject* type);
