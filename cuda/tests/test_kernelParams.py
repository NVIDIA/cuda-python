# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
import pytest
from cuda import cuda, cudart, nvrtc
import numpy as np
import ctypes

def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError('Cuda Error: {}'.format(err))
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError('Cudart Error: {}'.format(err))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError('Nvrtc Error: {}'.format(err))
    else:
        raise RuntimeError('Unknown error type: {}'.format(err))

def common_nvrtc(allKernelStrings, dev):
    err, major = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev)
    ASSERT_DRV(err)
    err, minor = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev)
    ASSERT_DRV(err)
    err, _, nvrtc_minor = nvrtc.nvrtcVersion()
    ASSERT_DRV(err)
    use_cubin = (nvrtc_minor >= 1)
    prefix = 'sm' if use_cubin else 'compute'
    arch_arg = bytes(f'--gpu-architecture={prefix}_{major}{minor}', 'ascii')

    err, prog = nvrtc.nvrtcCreateProgram(str.encode(allKernelStrings), b'allKernelStrings.cu', 0, [], [])
    ASSERT_DRV(err)
    opts = [b'--fmad=false', arch_arg]
    err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)

    err_log, logSize = nvrtc.nvrtcGetProgramLogSize(prog)
    ASSERT_DRV(err_log)
    log = b' ' * logSize
    err_log, = nvrtc.nvrtcGetProgramLog(prog, log)
    ASSERT_DRV(err_log)
    result = log.decode()
    if len(result) > 1:
        print(result)
    ASSERT_DRV(err)

    if use_cubin:
        err, dataSize = nvrtc.nvrtcGetCUBINSize(prog)
        ASSERT_DRV(err)
        data = b' ' * dataSize
        err, = nvrtc.nvrtcGetCUBIN(prog, data)
        ASSERT_DRV(err)
    else:
        err, dataSize = nvrtc.nvrtcGetPTXSize(prog)
        ASSERT_DRV(err)
        data = b' ' * dataSize
        err, = nvrtc.nvrtcGetPTX(prog, data)
        ASSERT_DRV(err)

    err, module = cuda.cuModuleLoadData(np.char.array(data))
    ASSERT_DRV(err)

    return module

def test_kernelParams_empty():
    err, = cuda.cuInit(0)
    ASSERT_DRV(err)
    err, cuDevice = cuda.cuDeviceGet(0)
    ASSERT_DRV(err)
    err, context = cuda.cuCtxCreate(0, cuDevice)
    ASSERT_DRV(err)

    kernelString = '''\
    static __device__ bool isDone;
    extern "C" __global__
    void empty_kernel()
    {
        isDone = true;
        if (isDone) return;
    }
    '''

    module = common_nvrtc(kernelString, cuDevice)

    # cudaStructs kernel
    err, kernel = cuda.cuModuleGetFunction(module, b'empty_kernel')
    ASSERT_DRV(err)

    err, stream = cuda.cuStreamCreate(0)
    ASSERT_DRV(err)

    err, = cuda.cuLaunchKernel(kernel,
                               1, 1, 1,   # grid dim
                               1, 1, 1,   # block dim
                               0, stream, # shared mem and stream
                               ((), ()), 0) # arguments
    ASSERT_DRV(err)
    err, = cuda.cuLaunchKernel(kernel,
                               1, 1, 1,   # grid dim
                               1, 1, 1,   # block dim
                               0, stream, # shared mem and stream
                               None, 0) # arguments
    ASSERT_DRV(err)

    # Retrieve global and validate
    isDone_host = ctypes.c_bool()
    err, isDonePtr_device, isDonePtr_device_size = cuda.cuModuleGetGlobal(module, b'isDone')
    ASSERT_DRV(err)
    assert(isDonePtr_device_size == ctypes.sizeof(ctypes.c_bool))
    err, = cuda.cuMemcpyDtoHAsync(isDone_host, isDonePtr_device, ctypes.sizeof(ctypes.c_bool), stream)
    ASSERT_DRV(err)
    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)
    assert(isDone_host.value == True)

    err, = cuda.cuStreamDestroy(stream)
    ASSERT_DRV(err)
    err, = cuda.cuModuleUnload(module)
    ASSERT_DRV(err)
    err, = cuda.cuCtxDestroy(context)
    ASSERT_DRV(err)

def kernelParams_basic(use_ctypes_as_values):
    err, = cuda.cuInit(0)
    ASSERT_DRV(err)
    err, cuDevice = cuda.cuDeviceGet(0)
    ASSERT_DRV(err)
    err, context = cuda.cuCtxCreate(0, cuDevice)
    ASSERT_DRV(err)

    if use_ctypes_as_values:
        assertValues_host = (ctypes.c_bool(True),
                             ctypes.c_char(b'Z'), ctypes.c_wchar('Ā'),
                             ctypes.c_byte(-127), ctypes.c_ubyte(255),
                             ctypes.c_short(1), ctypes.c_ushort(1),
                             ctypes.c_int(2), ctypes.c_uint(2),
                             ctypes.c_long(3), ctypes.c_ulong(3),
                             ctypes.c_longlong(4), ctypes.c_ulonglong(4),
                             ctypes.c_size_t(5),
                             ctypes.c_float(float(123.456)), ctypes.c_float(float(123.456)),
                             ctypes.c_void_p(0xdeadbeef))
    else:
        assertValues_host = (True,
                             b'Z', 'Ā',
                             -127, 255,
                             90, 72,
                             85, 82,
                             66, 65,
                             86, 90,
                             33,
                             float(123.456), float(123.456),
                             0xdeadbeef)
    assertTypes_host = (ctypes.c_bool,
                        ctypes.c_char, ctypes.c_wchar,
                        ctypes.c_byte, ctypes.c_ubyte,
                        ctypes.c_short, ctypes.c_ushort,
                        ctypes.c_int, ctypes.c_uint,
                        ctypes.c_long, ctypes.c_ulong,
                        ctypes.c_longlong, ctypes.c_ulonglong,
                        ctypes.c_size_t,
                        ctypes.c_float, ctypes.c_double,
                        ctypes.c_void_p)

    basicKernelString = '''\
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
    '''
    idx = 0
    while '{}' in basicKernelString:
        val = assertValues_host[idx].value if use_ctypes_as_values else assertValues_host[idx]
        if assertTypes_host[idx] == ctypes.c_float:
            basicKernelString = basicKernelString.replace('{}', str(float(val)) + 'f', 1)
        elif assertTypes_host[idx] == ctypes.c_double:
            basicKernelString = basicKernelString.replace('{}', str(float(val)), 1)
        elif assertTypes_host[idx] == ctypes.c_char:
            basicKernelString = basicKernelString.replace('{}', str(val)[1:], 1)
        elif assertTypes_host[idx] == ctypes.c_wchar:
            basicKernelString = basicKernelString.replace('{}', str(ord(val)), 1)
        else:
            basicKernelString = basicKernelString.replace('{}', str(int(val)), 1)
        idx += 1

    module = common_nvrtc(basicKernelString, cuDevice)

    err, kernel = cuda.cuModuleGetFunction(module, b'basic')
    ASSERT_DRV(err)

    err, stream = cuda.cuStreamCreate(0)
    ASSERT_DRV(err)

    # Prepare kernel
    err, pb = cuda.cuMemAlloc(ctypes.sizeof(ctypes.c_bool))
    ASSERT_DRV(err)
    err, pc = cuda.cuMemAlloc(ctypes.sizeof(ctypes.c_char))
    ASSERT_DRV(err)
    err, pwc = cuda.cuMemAlloc(ctypes.sizeof(ctypes.c_wchar))
    ASSERT_DRV(err)
    err, pbyte = cuda.cuMemAlloc(ctypes.sizeof(ctypes.c_byte))
    ASSERT_DRV(err)
    err, pubyte = cuda.cuMemAlloc(ctypes.sizeof(ctypes.c_ubyte))
    ASSERT_DRV(err)
    err, ps = cuda.cuMemAlloc(ctypes.sizeof(ctypes.c_short))
    ASSERT_DRV(err)
    err, pus = cuda.cuMemAlloc(ctypes.sizeof(ctypes.c_ushort))
    ASSERT_DRV(err)
    err, pi = cuda.cuMemAlloc(ctypes.sizeof(ctypes.c_int))
    ASSERT_DRV(err)
    err, pui = cuda.cuMemAlloc(ctypes.sizeof(ctypes.c_uint))
    ASSERT_DRV(err)
    err, pl = cuda.cuMemAlloc(ctypes.sizeof(ctypes.c_long))
    ASSERT_DRV(err)
    err, pul = cuda.cuMemAlloc(ctypes.sizeof(ctypes.c_ulong))
    ASSERT_DRV(err)
    err, pll = cuda.cuMemAlloc(ctypes.sizeof(ctypes.c_longlong))
    ASSERT_DRV(err)
    err, pull = cuda.cuMemAlloc(ctypes.sizeof(ctypes.c_ulonglong))
    ASSERT_DRV(err)
    err, psize = cuda.cuMemAlloc(ctypes.sizeof(ctypes.c_size_t))
    ASSERT_DRV(err)
    err, pf = cuda.cuMemAlloc(ctypes.sizeof(ctypes.c_float))
    ASSERT_DRV(err)
    err, pd = cuda.cuMemAlloc(ctypes.sizeof(ctypes.c_double))
    ASSERT_DRV(err)

    assertValues_device = (pb,
                           pc, pwc,
                           pbyte, pubyte,
                           ps, pus,
                           pi, pui,
                           pl, pul,
                           pll, pull,
                           psize,
                           pf, pd)
    assertTypes_device = (None,
                          None, None,
                          None, None,
                          None, None,
                          None, None,
                          None, None,
                          None, None,
                          None,
                          None, None)

    basicKernelValues = assertValues_host + assertValues_device
    basicKernelTypes = assertTypes_host + assertTypes_device
    err, = cuda.cuLaunchKernel(kernel,
                               1, 1, 1,   # grid dim
                               1, 1, 1,   # block dim
                               0, stream, # shared mem and stream
                               (basicKernelValues, basicKernelTypes), 0) # arguments
    ASSERT_DRV(err)

    # Retrieve each dptr
    host_params = tuple([valueType() for valueType in assertTypes_host[:-1]])
    for i in range(len(host_params)):
        err, = cuda.cuMemcpyDtoHAsync(host_params[i], assertValues_device[i], ctypes.sizeof(assertTypes_host[i]), stream)
        ASSERT_DRV(err)

    # Validate retrieved values
    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)
    for i in range(len(host_params)):
        val = basicKernelValues[i].value if use_ctypes_as_values else basicKernelValues[i]
        if basicKernelTypes[i] == ctypes.c_float:
            if use_ctypes_as_values:
                assert(val == host_params[i].value)
            else:
                assert(val == (int(host_params[i].value * 1000) / 1000))
        else:
            assert(val == host_params[i].value)

    err, = cuda.cuMemFree(pb)
    ASSERT_DRV(err)
    err, = cuda.cuMemFree(pc)
    ASSERT_DRV(err)
    err, = cuda.cuMemFree(pwc)
    ASSERT_DRV(err)
    err, = cuda.cuMemFree(pbyte)
    ASSERT_DRV(err)
    err, = cuda.cuMemFree(pubyte)
    ASSERT_DRV(err)
    err, = cuda.cuMemFree(ps)
    ASSERT_DRV(err)
    err, = cuda.cuMemFree(pus)
    ASSERT_DRV(err)
    err, = cuda.cuMemFree(pi)
    ASSERT_DRV(err)
    err, = cuda.cuMemFree(pui)
    ASSERT_DRV(err)
    err, = cuda.cuMemFree(pl)
    ASSERT_DRV(err)
    err, = cuda.cuMemFree(pul)
    ASSERT_DRV(err)
    err, = cuda.cuMemFree(pll)
    ASSERT_DRV(err)
    err, = cuda.cuMemFree(pull)
    ASSERT_DRV(err)
    err, = cuda.cuMemFree(psize)
    ASSERT_DRV(err)
    err, = cuda.cuMemFree(pf)
    ASSERT_DRV(err)
    err, = cuda.cuMemFree(pd)
    ASSERT_DRV(err)
    err, = cuda.cuStreamDestroy(stream)
    ASSERT_DRV(err)
    err, = cuda.cuModuleUnload(module)
    ASSERT_DRV(err)
    err, = cuda.cuCtxDestroy(context)
    ASSERT_DRV(err)

def test_kernelParams_basic():
    # Kernel is given basic Python primative values as value input
    kernelParams_basic(use_ctypes_as_values = False)

def test_kernelParams_basic_ctypes():
    # Kernel is given basic c_type instances as primative value input
    kernelParams_basic(use_ctypes_as_values = True)

def test_kernelParams_types_cuda():
    err, = cuda.cuInit(0)
    ASSERT_DRV(err)
    err, cuDevice = cuda.cuDeviceGet(0)
    ASSERT_DRV(err)
    err, context = cuda.cuCtxCreate(0, cuDevice)
    ASSERT_DRV(err)
    err, uvaSupported = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, cuDevice)
    ASSERT_DRV(err)

    err, perr = cudart.cudaMalloc(ctypes.sizeof(ctypes.c_int))
    ASSERT_DRV(err)
    err, pSurface_host = cudart.cudaHostAlloc(cudart.sizeof(cudart.cudaSurfaceObject_t), cudart.cudaHostAllocMapped)
    ASSERT_DRV(err)
    err, pDim3_host = cudart.cudaHostAlloc(cudart.sizeof(cudart.dim3), cudart.cudaHostAllocMapped)
    ASSERT_DRV(err)

    # Get device pointer if UVM is not enabled
    if uvaSupported:
        kernelValues = (cudart.cudaError_t.cudaErrorUnknown, perr,                                         # enums
                        cudart.cudaSurfaceObject_t(248), cudart.cudaSurfaceObject_t(_ptr=pSurface_host),   # typedef of primative
                        cudart.dim3(), cudart.dim3(_ptr=pDim3_host))                                       # struct
    else:
        err, pSurface_device = cudart.cudaHostGetDevicePointer(pSurface_host, 0)
        ASSERT_DRV(err)
        err, pDim3_device = cudart.cudaHostGetDevicePointer(pDim3_host, 0)
        ASSERT_DRV(err)
        kernelValues = (cudart.cudaError_t.cudaErrorUnknown, perr,                                         # enums
                        cudart.cudaSurfaceObject_t(248), cudart.cudaSurfaceObject_t(_ptr=pSurface_device), # typedef of primative
                        cudart.dim3(), cudart.dim3(_ptr=pDim3_device))                                     # struct
    kernelTypes = (None, ctypes.c_void_p,
                   None, ctypes.c_void_p,
                   None, ctypes.c_void_p)
    kernelValues[4].x = 1
    kernelValues[4].y = 2
    kernelValues[4].z = 3

    kernelString = '''\
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
    '''

    module = common_nvrtc(kernelString, cuDevice)

    # cudaStructs kernel
    err, kernel = cuda.cuModuleGetFunction(module, b'structsCuda')
    ASSERT_DRV(err)

    err, stream = cuda.cuStreamCreate(0)
    ASSERT_DRV(err)

    err, = cuda.cuLaunchKernel(kernel,
                               1, 1, 1,   # grid dim
                               1, 1, 1,   # block dim
                               0, stream, # shared mem and stream
                               (kernelValues, kernelTypes), 0) # arguments
    ASSERT_DRV(err)

    # Retrieve each dptr
    host_err = ctypes.c_int()
    err, = cudart.cudaMemcpyAsync(ctypes.addressof(host_err), perr, ctypes.sizeof(ctypes.c_int()), cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    ASSERT_DRV(err)

    # Validate kernel values
    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)
    cuda_err = cudart.cudaError_t(host_err.value)

    if uvaSupported:
        assert(kernelValues[0] == cuda_err)
        assert(int(kernelValues[2]) == int(kernelValues[3]))
        assert(kernelValues[4].x == kernelValues[5].x)
        assert(kernelValues[4].y == kernelValues[5].y)
        assert(kernelValues[4].z == kernelValues[5].z)
    else:
        surface_host = cudart.cudaSurfaceObject_t(_ptr=pSurface_host)
        dim3_host = cudart.dim3(_ptr=pDim3_host)
        assert(kernelValues[0] == cuda_err)
        assert(int(kernelValues[2]) == int(surface_host))
        assert(kernelValues[4].x == dim3_host.x)
        assert(kernelValues[4].y == dim3_host.y)
        assert(kernelValues[4].z == dim3_host.z)

    err, = cudart.cudaFree(perr)
    ASSERT_DRV(err)
    err, = cudart.cudaFreeHost(pSurface_host)
    ASSERT_DRV(err)
    err, = cudart.cudaFreeHost(pDim3_host)
    ASSERT_DRV(err)
    err, = cuda.cuStreamDestroy(stream)
    ASSERT_DRV(err)
    err, = cuda.cuModuleUnload(module)
    ASSERT_DRV(err)
    err, = cuda.cuCtxDestroy(context)
    ASSERT_DRV(err)

def test_kernelParams_struct_custom():
    err, = cuda.cuInit(0)
    ASSERT_DRV(err)
    err, cuDevice = cuda.cuDeviceGet(0)
    ASSERT_DRV(err)
    err, context = cuda.cuCtxCreate(0, cuDevice)
    ASSERT_DRV(err)
    err, uvaSupported = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, cuDevice)
    ASSERT_DRV(err)

    kernelString = '''\
    struct testStruct {
        int value;
    };

    extern "C" __global__
    void structCustom(struct testStruct src, struct testStruct *dst)
    {
        dst->value = src.value;
    }
    '''

    module = common_nvrtc(kernelString, cuDevice)

    err, kernel = cuda.cuModuleGetFunction(module, b'structCustom')
    ASSERT_DRV(err)

    err, stream = cuda.cuStreamCreate(0)
    ASSERT_DRV(err)

    # structCustom kernel
    class testStruct(ctypes.Structure):
        _fields_ = [('value',ctypes.c_int)]

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

    err, = cuda.cuLaunchKernel(kernel,
                               1, 1, 1,   # grid dim
                               1, 1, 1,   # block dim
                               0, stream, # shared mem and stream
                               (kernelValues, kernelTypes), 0) # arguments
    ASSERT_DRV(err)

    # Validate kernel values
    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)
    struct_shared = testStruct.from_address(pStruct_host)
    assert(kernelValues[0].value == struct_shared.value)

    err, = cudart.cudaFreeHost(pStruct_host)
    ASSERT_DRV(err)
    err, = cuda.cuStreamDestroy(stream)
    ASSERT_DRV(err)
    err, = cuda.cuModuleUnload(module)
    ASSERT_DRV(err)
    err, = cuda.cuCtxDestroy(context)
    ASSERT_DRV(err)

def kernelParams_buffer_protocol_ctypes_common(pass_by_address):
    err, = cuda.cuInit(0)
    ASSERT_DRV(err)
    err, cuDevice = cuda.cuDeviceGet(0)
    ASSERT_DRV(err)
    err, context = cuda.cuCtxCreate(0, cuDevice)
    ASSERT_DRV(err)
    err, uvaSupported = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, cuDevice)
    ASSERT_DRV(err)

    kernelString = '''\
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
    '''

    module = common_nvrtc(kernelString, cuDevice)

    err, kernel = cuda.cuModuleGetFunction(module, b'testkernel')
    ASSERT_DRV(err)

    err, stream = cuda.cuStreamCreate(0)
    ASSERT_DRV(err)

    # testkernel kernel
    class testStruct(ctypes.Structure):
        _fields_ = [('value',ctypes.c_int)]

    err, pInt_host = cudart.cudaHostAlloc(ctypes.sizeof(ctypes.c_int), cudart.cudaHostAllocMapped)
    ASSERT_DRV(err)
    err, pFloat_host = cudart.cudaHostAlloc(ctypes.sizeof(ctypes.c_float), cudart.cudaHostAllocMapped)
    ASSERT_DRV(err)
    err, pStruct_host = cudart.cudaHostAlloc(ctypes.sizeof(testStruct), cudart.cudaHostAllocMapped)
    ASSERT_DRV(err)

    # Get device pointer if UVM is not enabled
    if uvaSupported:
        kernelValues = (ctypes.c_int(1), ctypes.c_void_p(pInt_host),
                        ctypes.c_float(float(123.456)), ctypes.c_void_p(pFloat_host),
                        testStruct(5), ctypes.c_void_p(pStruct_host))
    else:
        err, pInt_device = cudart.cudaHostGetDevicePointer(pInt_host, 0)
        ASSERT_DRV(err)
        err, pFloat_device = cudart.cudaHostGetDevicePointer(pFloat_host, 0)
        ASSERT_DRV(err)
        err, pStruct_device = cudart.cudaHostGetDevicePointer(pStruct_host, 0)
        ASSERT_DRV(err)
        kernelValues = (ctypes.c_int(1), ctypes.c_void_p(pInt_device),
                        ctypes.c_float(float(123.456)), ctypes.c_void_p(pFloat_device),
                        testStruct(5), ctypes.c_void_p(pStruct_device))

    packagedParams = (ctypes.c_void_p*len(kernelValues))()
    for idx in range(len(packagedParams)):
        packagedParams[idx] = ctypes.addressof(kernelValues[idx])
    err, = cuda.cuLaunchKernel(kernel,
                               1, 1, 1,   # grid dim
                               1, 1, 1,   # block dim
                               0, stream, # shared mem and stream
                               ctypes.addressof(packagedParams) if pass_by_address else packagedParams, 0) # arguments
    ASSERT_DRV(err)

    # Validate kernel values
    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)
    assert(kernelValues[0].value == ctypes.c_int.from_address(pInt_host).value)
    assert(kernelValues[2].value == ctypes.c_float.from_address(pFloat_host).value)
    assert(kernelValues[4].value == testStruct.from_address(pStruct_host).value)

    err, = cudart.cudaFreeHost(pStruct_host)
    ASSERT_DRV(err)
    err, = cuda.cuStreamDestroy(stream)
    ASSERT_DRV(err)
    err, = cuda.cuModuleUnload(module)
    ASSERT_DRV(err)
    err, = cuda.cuCtxDestroy(context)
    ASSERT_DRV(err)

def test_kernelParams_buffer_protocol_ctypes():
    kernelParams_buffer_protocol_ctypes_common(pass_by_address=True)
    kernelParams_buffer_protocol_ctypes_common(pass_by_address=False)

def test_kernelParams_buffer_protocol_numpy():
    err, = cuda.cuInit(0)
    ASSERT_DRV(err)
    err, cuDevice = cuda.cuDeviceGet(0)
    ASSERT_DRV(err)
    err, context = cuda.cuCtxCreate(0, cuDevice)
    ASSERT_DRV(err)
    err, uvaSupported = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, cuDevice)
    ASSERT_DRV(err)

    kernelString = '''\
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
    '''

    module = common_nvrtc(kernelString, cuDevice)

    err, kernel = cuda.cuModuleGetFunction(module, b'testkernel')
    ASSERT_DRV(err)

    err, stream = cuda.cuStreamCreate(0)
    ASSERT_DRV(err)

    # testkernel kernel
    testStruct = np.dtype([('value', np.int32)])

    err, pInt_host = cudart.cudaHostAlloc(np.dtype(np.int32).itemsize, cudart.cudaHostAllocMapped)
    ASSERT_DRV(err)
    err, pFloat_host = cudart.cudaHostAlloc(np.dtype(np.float32).itemsize, cudart.cudaHostAllocMapped)
    ASSERT_DRV(err)
    err, pStruct_host = cudart.cudaHostAlloc(testStruct.itemsize, cudart.cudaHostAllocMapped)
    ASSERT_DRV(err)

    # Get device pointer if UVM is not enabled
    if uvaSupported:
        kernelValues = (np.array(1, dtype=np.uint32), np.array([pInt_host], dtype=np.uint64),
                        np.array(float(123.456), dtype=np.float32), np.array([pFloat_host], dtype=np.uint64),
                        np.array([5], testStruct), np.array([pStruct_host], dtype=np.uint64))
    else:
        err, pInt_device = cudart.cudaHostGetDevicePointer(pInt_host, 0)
        ASSERT_DRV(err)
        err, pFloat_device = cudart.cudaHostGetDevicePointer(pFloat_host, 0)
        ASSERT_DRV(err)
        err, pStruct_device = cudart.cudaHostGetDevicePointer(pStruct_host, 0)
        ASSERT_DRV(err)
        kernelValues = (np.array(1, dtype=np.int32), np.array([pInt_device], dtype=np.uint64),
                        np.array(float(123.456), dtype=np.float32), np.array([pFloat_device], dtype=np.uint64),
                        np.array([5], testStruct), np.array([pStruct_device], dtype=np.uint64))

    packagedParams = np.array([arg.ctypes.data for arg in kernelValues], dtype=np.uint64)
    err, = cuda.cuLaunchKernel(kernel,
                               1, 1, 1,   # grid dim
                               1, 1, 1,   # block dim
                               0, stream, # shared mem and stream
                               packagedParams, 0) # arguments
    ASSERT_DRV(err)

    # Validate kernel values
    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)

    class numpy_address_wrapper():
        def __init__(self, address, typestr):
            self.__array_interface__ = {'data': (address, False),
                                        'typestr': typestr,
                                        'shape': (1,)}

    assert(kernelValues[0] == np.array(numpy_address_wrapper(pInt_host, '<i4')))
    assert(kernelValues[2] == np.array(numpy_address_wrapper(pFloat_host, '<f4')))
    assert(kernelValues[4]['value'] == np.array(numpy_address_wrapper(pStruct_host, '<i4'), dtype=testStruct)['value'])

    err, = cudart.cudaFreeHost(pStruct_host)
    ASSERT_DRV(err)
    err, = cuda.cuStreamDestroy(stream)
    ASSERT_DRV(err)
    err, = cuda.cuModuleUnload(module)
    ASSERT_DRV(err)
    err, = cuda.cuCtxDestroy(context)
    ASSERT_DRV(err)
