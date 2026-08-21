# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes

import numpy as np
import pytest

import cuda.bindings._v2.driver as cuda
import cuda.bindings._v2.nvrtc as nvrtc
import cuda.bindings.runtime as cudart


def ASSERT_DRV(err):
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cudart Error: {err}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")


def common_nvrtc(allKernelStrings, dev):
    major = cuda.device_get_attribute(cuda.DeviceAttribute.COMPUTE_CAPABILITY_MAJOR, dev)
    minor = cuda.device_get_attribute(cuda.DeviceAttribute.COMPUTE_CAPABILITY_MINOR, dev)
    _, nvrtc_minor = nvrtc.version()
    use_cubin = nvrtc_minor >= 1
    prefix = "sm" if use_cubin else "compute"
    arch_arg = bytes(f"--gpu-architecture={prefix}_{major}{minor}", "ascii")

    prog = nvrtc.create_program(str.encode(allKernelStrings), b"allKernelStrings.cu")
    opts = (b"--fmad=false", arch_arg)
    nvrtc.compile_program(prog, opts)

    log = nvrtc.get_program_log(prog)
    result = log.decode()
    if len(result) > 1:
        print(result)

    if use_cubin:
        data = nvrtc.get_cubin(prog)
    else:
        data = nvrtc.get_ptx(prog)

    return cuda.module_load_data(np.char.array(data))


def test_kernelParams_empty(device):
    kernelString = """\
    static __device__ bool isDone;
    extern "C" __global__
    void empty_kernel()
    {
        isDone = true;
        if (isDone) return;
    }
    """

    module = common_nvrtc(kernelString, device)

    # cudaStructs kernel
    kernel = cuda.module_get_function(module, "empty_kernel")

    stream = cuda.stream_create(0)

    cuda.launch_kernel(
        kernel,
        1,
        1,
        1,  # grid dim
        1,
        1,
        1,  # block dim
        0,
        stream,  # shared mem and stream
        ((), ()),
        0,
    )  # arguments
    cuda.launch_kernel(
        kernel,
        1,
        1,
        1,  # grid dim
        1,
        1,
        1,  # block dim
        0,
        stream,  # shared mem and stream
        None,
        0,
    )  # arguments

    # Retrieve global and validate
    isDone_host = ctypes.c_bool()
    isDonePtr_device, isDonePtr_device_size = cuda.module_get_global_v2(module, b"isDone")
    assert isDonePtr_device_size == ctypes.sizeof(ctypes.c_bool)
    # memcpy_dtoh_async_v2 requires a 1D buffer or a raw address; a scalar
    # ctypes object's buffer has ndim 0, so pass its address instead.
    cuda.memcpy_dtoh_async_v2(ctypes.addressof(isDone_host), isDonePtr_device, ctypes.sizeof(ctypes.c_bool), stream)
    cuda.stream_synchronize(stream)
    assert isDone_host.value is True

    cuda.stream_destroy_v2(stream)
    cuda.module_unload(module)


@pytest.mark.parametrize("use_ctypes_as_values", [False, True], ids=["no-ctypes", "ctypes"])
def test_kernelParams(use_ctypes_as_values, device):
    if use_ctypes_as_values:
        assertValues_host = (
            ctypes.c_bool(True),
            ctypes.c_char(b"Z"),
            ctypes.c_wchar("Ā"),
            ctypes.c_byte(-127),
            ctypes.c_ubyte(255),
            ctypes.c_short(1),
            ctypes.c_ushort(1),
            ctypes.c_int(2),
            ctypes.c_uint(2),
            ctypes.c_long(3),
            ctypes.c_ulong(3),
            ctypes.c_longlong(4),
            ctypes.c_ulonglong(4),
            ctypes.c_size_t(5),
            ctypes.c_float(123.456),
            ctypes.c_float(123.456),
            ctypes.c_void_p(0xDEADBEEF),
        )
    else:
        assertValues_host = (
            True,
            b"Z",
            "Ā",
            -127,
            255,
            90,
            72,
            85,
            82,
            66,
            65,
            86,
            90,
            33,
            123.456,
            123.456,
            0xDEADBEEF,
        )
    assertTypes_host = (
        ctypes.c_bool,
        ctypes.c_char,
        ctypes.c_wchar,
        ctypes.c_byte,
        ctypes.c_ubyte,
        ctypes.c_short,
        ctypes.c_ushort,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_long,
        ctypes.c_ulong,
        ctypes.c_longlong,
        ctypes.c_ulonglong,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_double,
        ctypes.c_void_p,
    )

    basicKernelString = """\
    extern "C" __global__
    void basic(bool b,
               char c, wchar_t wc,
               signed char byte, unsigned char ubyte,
               short s, unsigned short us,
               int i, unsigned int ui,
               long l, unsigned long ul,
               long long ll, unsigned long long ull,
               size_t size,
               float f, double d,
               void *p,
               bool *pb,
               char *pc, wchar_t *pwc,
               signed char *pbyte, unsigned char *pubyte,
               short *ps, unsigned short *pus,
               int *pi, unsigned int *pui,
               long *pl, unsigned long *pul,
               long long *pll, unsigned long long *pull,
               size_t *psize,
               float *pf, double *pd)
    {
        assert(b == {});
        assert(c == {});
        assert(wc == {});
        assert(byte == {});
        assert(ubyte == {});
        assert(s == {});
        assert(us == {});
        assert(i == {});
        assert(ui == {});
        assert(l == {});
        assert(ul == {});
        assert(ll == {});
        assert(ull == {});
        assert(size == {});
        assert(f == {});
        assert(d == {});
        assert(p == (void*){});
        *pb = b;
        *pc = c;
        *pwc = wc;
        *pbyte = byte;
        *pubyte = ubyte;
        *ps = s;
        *pus = us;
        *pi = i;
        *pui = ui;
        *pl = l;
        *pul = ul;
        *pll = ll;
        *pull = ull;
        *psize = size;
        *pf = f;
        *pd = d;
    }
    """
    idx = 0
    while "{}" in basicKernelString:
        val = assertValues_host[idx].value if use_ctypes_as_values else assertValues_host[idx]
        if assertTypes_host[idx] == ctypes.c_float:
            basicKernelString = basicKernelString.replace("{}", str(float(val)) + "f", 1)
        elif assertTypes_host[idx] == ctypes.c_double:
            basicKernelString = basicKernelString.replace("{}", str(float(val)), 1)
        elif assertTypes_host[idx] == ctypes.c_char:
            basicKernelString = basicKernelString.replace("{}", str(val)[1:], 1)
        elif assertTypes_host[idx] == ctypes.c_wchar:
            basicKernelString = basicKernelString.replace("{}", str(ord(val)), 1)
        else:
            basicKernelString = basicKernelString.replace("{}", str(int(val)), 1)
        idx += 1

    module = common_nvrtc(basicKernelString, device)

    kernel = cuda.module_get_function(module, "basic")

    stream = cuda.stream_create(0)

    # Prepare kernel
    pb = cuda.mem_alloc_v2(ctypes.sizeof(ctypes.c_bool))
    pc = cuda.mem_alloc_v2(ctypes.sizeof(ctypes.c_char))
    pwc = cuda.mem_alloc_v2(ctypes.sizeof(ctypes.c_wchar))
    pbyte = cuda.mem_alloc_v2(ctypes.sizeof(ctypes.c_byte))
    pubyte = cuda.mem_alloc_v2(ctypes.sizeof(ctypes.c_ubyte))
    ps = cuda.mem_alloc_v2(ctypes.sizeof(ctypes.c_short))
    pus = cuda.mem_alloc_v2(ctypes.sizeof(ctypes.c_ushort))
    pi = cuda.mem_alloc_v2(ctypes.sizeof(ctypes.c_int))
    pui = cuda.mem_alloc_v2(ctypes.sizeof(ctypes.c_uint))
    pl = cuda.mem_alloc_v2(ctypes.sizeof(ctypes.c_long))
    pul = cuda.mem_alloc_v2(ctypes.sizeof(ctypes.c_ulong))
    pll = cuda.mem_alloc_v2(ctypes.sizeof(ctypes.c_longlong))
    pull = cuda.mem_alloc_v2(ctypes.sizeof(ctypes.c_ulonglong))
    psize = cuda.mem_alloc_v2(ctypes.sizeof(ctypes.c_size_t))
    pf = cuda.mem_alloc_v2(ctypes.sizeof(ctypes.c_float))
    pd = cuda.mem_alloc_v2(ctypes.sizeof(ctypes.c_double))

    assertValues_device = (pb, pc, pwc, pbyte, pubyte, ps, pus, pi, pui, pl, pul, pll, pull, psize, pf, pd)
    # mem_alloc_v2 returns a plain int device pointer. Pairing it with the
    # ctypes.c_void_p type marker is enough for _HelperKernelParams to marshal
    # it by pointer -- no need to box the value itself.
    assertTypes_device = (ctypes.c_void_p,) * len(assertValues_device)

    basicKernelValues = assertValues_host + assertValues_device
    basicKernelTypes = assertTypes_host + assertTypes_device
    cuda.launch_kernel(
        kernel,
        1,
        1,
        1,  # grid dim
        1,
        1,
        1,  # block dim
        0,
        stream,  # shared mem and stream
        (basicKernelValues, basicKernelTypes),
        0,
    )  # arguments

    # Retrieve each dptr
    host_params = tuple([valueType() for valueType in assertTypes_host[:-1]])
    for i in range(len(host_params)):
        cuda.memcpy_dtoh_async_v2(
            ctypes.addressof(host_params[i]), assertValues_device[i], ctypes.sizeof(assertTypes_host[i]), stream
        )

    # Validate retrieved values
    cuda.stream_synchronize(stream)
    for i in range(len(host_params)):
        val = basicKernelValues[i].value if use_ctypes_as_values else basicKernelValues[i]
        if basicKernelTypes[i] == ctypes.c_float:
            if use_ctypes_as_values:
                assert val == host_params[i].value
            else:
                assert val == (int(host_params[i].value * 1000) / 1000)
        else:
            assert val == host_params[i].value

    cuda.mem_free_v2(pb)
    cuda.mem_free_v2(pc)
    cuda.mem_free_v2(pwc)
    cuda.mem_free_v2(pbyte)
    cuda.mem_free_v2(pubyte)
    cuda.mem_free_v2(ps)
    cuda.mem_free_v2(pus)
    cuda.mem_free_v2(pi)
    cuda.mem_free_v2(pui)
    cuda.mem_free_v2(pl)
    cuda.mem_free_v2(pul)
    cuda.mem_free_v2(pll)
    cuda.mem_free_v2(pull)
    cuda.mem_free_v2(psize)
    cuda.mem_free_v2(pf)
    cuda.mem_free_v2(pd)
    cuda.stream_destroy_v2(stream)
    cuda.module_unload(module)


def test_kernelParams_types_cuda(device):
    uvaSupported = cuda.device_get_attribute(cuda.DeviceAttribute.UNIFIED_ADDRESSING, device)

    err, perr = cudart.cudaMalloc(ctypes.sizeof(ctypes.c_int))
    ASSERT_DRV(err)
    err, pSurface_host = cudart.cudaHostAlloc(cudart.sizeof(cudart.cudaSurfaceObject_t), cudart.cudaHostAllocMapped)
    ASSERT_DRV(err)
    err, pDim3_host = cudart.cudaHostAlloc(cudart.sizeof(cudart.dim3), cudart.cudaHostAllocMapped)
    ASSERT_DRV(err)

    # Get device pointer if UVM is not enabled
    if uvaSupported:
        kernelValues = (
            cudart.cudaError_t.cudaErrorUnknown,
            perr,  # enums
            cudart.cudaSurfaceObject_t(248),
            cudart.cudaSurfaceObject_t(_ptr=pSurface_host),  # typedef of primative
            cudart.dim3(),
            cudart.dim3(_ptr=pDim3_host),
        )  # struct
    else:
        err, pSurface_device = cudart.cudaHostGetDevicePointer(pSurface_host, 0)
        ASSERT_DRV(err)
        err, pDim3_device = cudart.cudaHostGetDevicePointer(pDim3_host, 0)
        ASSERT_DRV(err)
        kernelValues = (
            cudart.cudaError_t.cudaErrorUnknown,
            perr,  # enums
            cudart.cudaSurfaceObject_t(248),
            cudart.cudaSurfaceObject_t(_ptr=pSurface_device),  # typedef of primative
            cudart.dim3(),
            cudart.dim3(_ptr=pDim3_device),
        )  # struct
    kernelTypes = (None, ctypes.c_void_p, None, ctypes.c_void_p, None, ctypes.c_void_p)
    kernelValues[4].x = 1
    kernelValues[4].y = 2
    kernelValues[4].z = 3

    kernelString = """\
    extern "C" __global__
    void structsCuda(cudaError_t err, cudaError_t *perr,
                     cudaSurfaceObject_t surface, cudaSurfaceObject_t *pSurface,
                     dim3 dim, dim3* pdim)
    {
        *perr = err;
        *pSurface = surface;
        pdim->x = dim.x;
        pdim->y = dim.y;
        pdim->z = dim.z;
    }
    """

    module = common_nvrtc(kernelString, device)

    # cudaStructs kernel
    kernel = cuda.module_get_function(module, "structsCuda")

    stream = cuda.stream_create(0)

    cuda.launch_kernel(
        kernel,
        1,
        1,
        1,  # grid dim
        1,
        1,
        1,  # block dim
        0,
        stream,  # shared mem and stream
        (kernelValues, kernelTypes),
        0,
    )  # arguments

    # Retrieve each dptr
    host_err = ctypes.c_int()
    (err,) = cudart.cudaMemcpyAsync(
        ctypes.addressof(host_err),
        perr,
        ctypes.sizeof(ctypes.c_int()),
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        stream,
    )
    ASSERT_DRV(err)

    # Validate kernel values
    cuda.stream_synchronize(stream)
    cuda_err = cudart.cudaError_t(host_err.value)

    if uvaSupported:
        assert kernelValues[0] == cuda_err
        assert int(kernelValues[2]) == int(kernelValues[3])
        assert kernelValues[4].x == kernelValues[5].x
        assert kernelValues[4].y == kernelValues[5].y
        assert kernelValues[4].z == kernelValues[5].z
    else:
        surface_host = cudart.cudaSurfaceObject_t(_ptr=pSurface_host)
        dim3_host = cudart.dim3(_ptr=pDim3_host)
        assert kernelValues[0] == cuda_err
        assert int(kernelValues[2]) == int(surface_host)
        assert kernelValues[4].x == dim3_host.x
        assert kernelValues[4].y == dim3_host.y
        assert kernelValues[4].z == dim3_host.z

    (err,) = cudart.cudaFree(perr)
    ASSERT_DRV(err)
    (err,) = cudart.cudaFreeHost(pSurface_host)
    ASSERT_DRV(err)
    (err,) = cudart.cudaFreeHost(pDim3_host)
    ASSERT_DRV(err)
    cuda.stream_destroy_v2(stream)
    cuda.module_unload(module)


def test_kernelParams_struct_custom(device):
    uvaSupported = cuda.device_get_attribute(cuda.DeviceAttribute.UNIFIED_ADDRESSING, device)

    kernelString = """\
    struct testStruct {
        int value;
    };

    extern "C" __global__
    void structCustom(struct testStruct src, struct testStruct *dst)
    {
        dst->value = src.value;
    }
    """

    module = common_nvrtc(kernelString, device)

    kernel = cuda.module_get_function(module, "structCustom")

    stream = cuda.stream_create(0)

    # structCustom kernel
    class testStruct(ctypes.Structure):
        _fields_ = [("value", ctypes.c_int)]

    err, pStruct_host = cudart.cudaHostAlloc(ctypes.sizeof(testStruct), cudart.cudaHostAllocMapped)
    ASSERT_DRV(err)

    # Get device pointer if UVM is not enabled
    if uvaSupported:
        kernelValues = (testStruct(5), pStruct_host)
    else:
        err, pStruct_device = cudart.cudaHostGetDevicePointer(pStruct_host, 0)
        ASSERT_DRV(err)
        kernelValues = (testStruct(5), pStruct_device)
    kernelTypes = (None, ctypes.c_void_p)

    cuda.launch_kernel(
        kernel,
        1,
        1,
        1,  # grid dim
        1,
        1,
        1,  # block dim
        0,
        stream,  # shared mem and stream
        (kernelValues, kernelTypes),
        0,
    )  # arguments

    # Validate kernel values
    cuda.stream_synchronize(stream)
    struct_shared = testStruct.from_address(pStruct_host)
    assert kernelValues[0].value == struct_shared.value

    (err,) = cudart.cudaFreeHost(pStruct_host)
    ASSERT_DRV(err)
    cuda.stream_destroy_v2(stream)
    cuda.module_unload(module)


@pytest.mark.parametrize("pass_by_address", [False, True], ids=["by-address", "not-by-address"])
def test_kernelParams_buffer_protocol(pass_by_address, device):
    uvaSupported = cuda.device_get_attribute(cuda.DeviceAttribute.UNIFIED_ADDRESSING, device)

    kernelString = """\
    struct testStruct {
        int value;
    };
    extern "C" __global__
    void testkernel(int i, int *pi,
                    float f, float *pf,
                    struct testStruct s, struct testStruct *ps)
    {
        *pi = i;
        *pf = f;
        ps->value = s.value;
    }
    """

    module = common_nvrtc(kernelString, device)

    kernel = cuda.module_get_function(module, "testkernel")

    stream = cuda.stream_create(0)

    # testkernel kernel
    class testStruct(ctypes.Structure):
        _fields_ = [("value", ctypes.c_int)]

    err, pInt_host = cudart.cudaHostAlloc(ctypes.sizeof(ctypes.c_int), cudart.cudaHostAllocMapped)
    ASSERT_DRV(err)
    err, pFloat_host = cudart.cudaHostAlloc(ctypes.sizeof(ctypes.c_float), cudart.cudaHostAllocMapped)
    ASSERT_DRV(err)
    err, pStruct_host = cudart.cudaHostAlloc(ctypes.sizeof(testStruct), cudart.cudaHostAllocMapped)
    ASSERT_DRV(err)

    # Get device pointer if UVM is not enabled
    if uvaSupported:
        kernelValues = (
            ctypes.c_int(1),
            ctypes.c_void_p(pInt_host),
            ctypes.c_float(123.456),
            ctypes.c_void_p(pFloat_host),
            testStruct(5),
            ctypes.c_void_p(pStruct_host),
        )
    else:
        err, pInt_device = cudart.cudaHostGetDevicePointer(pInt_host, 0)
        ASSERT_DRV(err)
        err, pFloat_device = cudart.cudaHostGetDevicePointer(pFloat_host, 0)
        ASSERT_DRV(err)
        err, pStruct_device = cudart.cudaHostGetDevicePointer(pStruct_host, 0)
        ASSERT_DRV(err)
        kernelValues = (
            ctypes.c_int(1),
            ctypes.c_void_p(pInt_device),
            ctypes.c_float(123.456),
            ctypes.c_void_p(pFloat_device),
            testStruct(5),
            ctypes.c_void_p(pStruct_device),
        )

    packagedParams = (ctypes.c_void_p * len(kernelValues))()
    for idx in range(len(packagedParams)):
        packagedParams[idx] = ctypes.addressof(kernelValues[idx])
    cuda.launch_kernel(
        kernel,
        1,
        1,
        1,  # grid dim
        1,
        1,
        1,  # block dim
        0,
        stream,  # shared mem and stream
        ctypes.addressof(packagedParams) if pass_by_address else packagedParams,
        0,
    )  # arguments

    # Validate kernel values
    cuda.stream_synchronize(stream)
    assert kernelValues[0].value == ctypes.c_int.from_address(pInt_host).value
    assert kernelValues[2].value == ctypes.c_float.from_address(pFloat_host).value
    assert kernelValues[4].value == testStruct.from_address(pStruct_host).value

    (err,) = cudart.cudaFreeHost(pStruct_host)
    ASSERT_DRV(err)
    cuda.stream_destroy_v2(stream)
    cuda.module_unload(module)


def test_kernelParams_buffer_protocol_numpy(device):
    uvaSupported = cuda.device_get_attribute(cuda.DeviceAttribute.UNIFIED_ADDRESSING, device)

    kernelString = """\
    struct testStruct {
        int value;
    };
    extern "C" __global__
    void testkernel(int i, int *pi,
                    float f, float *pf,
                    struct testStruct s, struct testStruct *ps)
    {
        *pi = i;
        *pf = f;
        ps->value = s.value;
    }
    """

    module = common_nvrtc(kernelString, device)

    kernel = cuda.module_get_function(module, "testkernel")

    stream = cuda.stream_create(0)

    # testkernel kernel
    testStruct = np.dtype([("value", np.int32)])

    err, pInt_host = cudart.cudaHostAlloc(np.dtype(np.int32).itemsize, cudart.cudaHostAllocMapped)
    ASSERT_DRV(err)
    err, pFloat_host = cudart.cudaHostAlloc(np.dtype(np.float32).itemsize, cudart.cudaHostAllocMapped)
    ASSERT_DRV(err)
    err, pStruct_host = cudart.cudaHostAlloc(testStruct.itemsize, cudart.cudaHostAllocMapped)
    ASSERT_DRV(err)

    # Get device pointer if UVM is not enabled
    if uvaSupported:
        kernelValues = (
            np.array(1, dtype=np.uint32),
            np.array([pInt_host], dtype=np.uint64),
            np.array(123.456, dtype=np.float32),
            np.array([pFloat_host], dtype=np.uint64),
            np.array([5], testStruct),
            np.array([pStruct_host], dtype=np.uint64),
        )
    else:
        err, pInt_device = cudart.cudaHostGetDevicePointer(pInt_host, 0)
        ASSERT_DRV(err)
        err, pFloat_device = cudart.cudaHostGetDevicePointer(pFloat_host, 0)
        ASSERT_DRV(err)
        err, pStruct_device = cudart.cudaHostGetDevicePointer(pStruct_host, 0)
        ASSERT_DRV(err)
        kernelValues = (
            np.array(1, dtype=np.int32),
            np.array([pInt_device], dtype=np.uint64),
            np.array(123.456, dtype=np.float32),
            np.array([pFloat_device], dtype=np.uint64),
            np.array([5], testStruct),
            np.array([pStruct_device], dtype=np.uint64),
        )

    packagedParams = np.array([arg.ctypes.data for arg in kernelValues], dtype=np.uint64)
    cuda.launch_kernel(
        kernel,
        1,
        1,
        1,  # grid dim
        1,
        1,
        1,  # block dim
        0,
        stream,  # shared mem and stream
        packagedParams,
        0,
    )  # arguments

    # Validate kernel values
    cuda.stream_synchronize(stream)

    class numpy_address_wrapper:
        def __init__(self, address, typestr):
            self.__array_interface__ = {"data": (address, False), "typestr": typestr, "shape": (1,)}

    assert kernelValues[0] == np.array(numpy_address_wrapper(pInt_host, "<i4"))
    assert kernelValues[2] == np.array(numpy_address_wrapper(pFloat_host, "<f4"))
    assert kernelValues[4]["value"] == np.array(numpy_address_wrapper(pStruct_host, "<i4"), dtype=testStruct)["value"]

    (err,) = cudart.cudaFreeHost(pStruct_host)
    ASSERT_DRV(err)
    cuda.stream_destroy_v2(stream)
    cuda.module_unload(module)


def test_kernelParams_c_int_out_of_range_raises(device):
    # #363: an out-of-range Python int for a c_int / c_byte kernel argument must
    # raise instead of being silently truncated to fit the declared width.
    kernelString = """\
    extern "C" __global__ void take_int(int i) {}
    """
    module = common_nvrtc(kernelString, device)
    kernel = cuda.module_get_function(module, "take_int")
    stream = cuda.stream_create(0)

    # An in-range value still packs and launches fine.
    cuda.launch_kernel(kernel, 1, 1, 1, 1, 1, 1, 0, stream, ((5,), (ctypes.c_int,)), 0)

    # Out-of-range values now raise OverflowError during packing (previously the
    # high bits were silently dropped, so the kernel saw a different value).
    with pytest.raises(OverflowError):
        cuda.launch_kernel(kernel, 1, 1, 1, 1, 1, 1, 0, stream, ((2**32 + 5,), (ctypes.c_int,)), 0)
    with pytest.raises(OverflowError):
        cuda.launch_kernel(kernel, 1, 1, 1, 1, 1, 1, 0, stream, ((200,), (ctypes.c_byte,)), 0)

    cuda.stream_synchronize(stream)
    cuda.stream_destroy_v2(stream)
    cuda.module_unload(module)
