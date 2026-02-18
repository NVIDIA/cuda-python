# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import time

import numpy as np
from common import common
from common.helper_cuda import checkCudaErrors
from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart

isoPropagator = """\
extern "C"
__global__ void injectSource(float *__restrict__ in, float *__restrict__ src, int it)
{
    if (threadIdx.x == 0)
        in[0] = src[it];
}

extern "C"
__global__ void createVelocity(float *__restrict__ vel, float vmult,  int nz,  int nx, int stride)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;

  int idx_out = iy * nx + ix;
  for (int iz = 0; iz < nz ; iz++) {
        vel[idx_out] = 3.0f * 3.0f * vmult;
        idx_out += stride;
    }
}

extern "C"
__global__ void createSource(float *__restrict__ x, float dt, float freq, int nt)
{
    int istart = (int) (60.0f/dt); // start max at 30 ms
    float pi2 = 2.0f * 3.141592654f;
    float agauss = 0.5f * freq;

    for ( int i=threadIdx.x; i < nt; ++ i) {
        float arg = 1.0e-3 * fabsf(i - istart) * agauss;
        x[i] = 1000.0f * expf(-2.0f * arg * arg) * cosf(pi2 * arg);
    }
}

extern "C"
__global__ void fwd_3D_orderX2k(float *g_curr_1, float *g_prev_1, float *g_vsq_1,
                                int nz,  int dimx, int stride);

#define radius 4
#define diameter (2*radius+1)
#define BDIMX 32
#define BDIMY 16

inline __device__ void advance(float2 *field, const int num_points) {
    #pragma unroll
    for (int i = 0; i < num_points; i++)
        field[i] = field[i + 1];
}

__global__ void fwd_3D_orderX2k(float *g_curr_1, float *g_prev_1, float *g_vsq_1,
                                int nz,  int nx, int stride) {
    stride = stride / 2;
    nx = nx / 2;
    const float c_coeff[5]  = {-3.0f * 2.847222222f,
                                1.600000f,
                               -0.200000f,
                                0.025396825f,
                               -0.001785f};

    float2 *g_prev = (float2 *)g_prev_1;
    float2 *g_curr = (float2 *)g_curr_1;
    float2 *g_vsq = (float2 *)g_vsq_1;
    __shared__ float s_data[BDIMY + 2 * radius][2 * BDIMX + 2 * (radius + (radius % 2))];

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int offset = -radius * stride;

    int idx_out = iy * nx + ix;
    int idx_in = idx_out + offset;

    float2 local_input[diameter], tmp1, tmp2;

    int tx = 2 * threadIdx.x + radius + (radius % 2);
    int ty = threadIdx.y + radius;

    #pragma unroll
    for (int i = 1; i < diameter; i++) {
        local_input[i] = g_curr[idx_in];
        idx_in += stride;
    }

    for (int iz = 0; iz < nz ; iz++) {
        advance(local_input, diameter - 1);
        local_input[diameter - 1] = g_curr[idx_in];

        // update the data slice in smem
        s_data[ty][tx] = local_input[radius].x;
        s_data[ty][tx + 1] = local_input[radius].y;

        // halo above/below
        if (threadIdx.y < radius) {
            tmp1 = (g_curr[idx_out - radius * nx]);
            s_data[threadIdx.y][tx] = tmp1.x;
            s_data[threadIdx.y][tx + 1] = tmp1.y;
        }

        if (threadIdx.y >= radius && threadIdx.y < 2 * radius) {
            tmp1 = (g_curr[idx_out + (BDIMY - radius) * nx]);
            s_data[threadIdx.y + BDIMY][tx] = tmp1.x;
            s_data[threadIdx.y + BDIMY][tx + 1] = tmp1.y;
        }

        // halo left/right
        if (threadIdx.x < (radius + 1) / 2) {
            tmp1 = (g_curr[idx_out - (radius + 1) / 2]);
            s_data[ty][tx - radius - (radius % 2)] = tmp1.x;
            s_data[ty][tx - radius - (radius % 2) + 1] = tmp1.y;

            tmp2 = (g_curr[idx_out + BDIMX]);
            s_data[ty][tx + 2 * BDIMX] = tmp2.x;
            s_data[ty][tx + 2 * BDIMX + 1] = tmp2.y;
        }
        __syncthreads();

        // compute the output values
        float2 temp, div;

        temp.x = 2.f * local_input[radius].x -  g_prev[idx_out].x;
        temp.y = 2.f * local_input[radius].y -  g_prev[idx_out].y;

        div.x = c_coeff[0] * local_input[radius].x;
        div.y = c_coeff[0] * local_input[radius].y;

        #pragma unroll
        for (int d = 1; d <= radius; d++) {
            div.x += c_coeff[d] * (local_input[radius + d].x + local_input[radius - d].x + s_data[ty - d][tx] +
                                   s_data[ty + d][tx] + s_data[ty][tx - d] + s_data[ty][tx + d]);
            div.y += c_coeff[d] * (local_input[radius + d].y + local_input[radius - d].y + s_data[ty - d][tx + 1] +
                                   s_data[ty + d][tx + 1] + s_data[ty][tx - d + 1] + s_data[ty][tx + d + 1]);
        }

        g_prev[idx_out].x =  temp.x + div.x * g_vsq[idx_out].x;
        g_prev[idx_out].y =  temp.y + div.y * g_vsq[idx_out].y;

        __syncthreads();

        idx_out += stride;
        idx_in += stride;
    }
}
"""

display_graph = False
verbose_prints = False


def align_nx(nx, blk, nops):
    n_align = (int)((nx - 1) / blk) + 1
    n_align *= blk
    n_align += 2 * nops
    n_align = (int)((n_align - 1) / 64) + 1
    n_align *= 64
    return (int)(n_align)


def align_ny(ny, blk, nops):
    n_align = (int)((ny - 1) / blk) + 1
    n_align *= blk
    n_align += 2 * nops
    return (int)(n_align)


#
# this class contains the input params
#
class params:
    def __init__(self):
        self.BDIMX = 32  # tiles x y for fd operators
        self.BDIMY = 16
        self.FD_ORDER = 4
        self.lead = 64 - self.FD_ORDER
        self.nx = align_nx(700, 2 * self.BDIMX, self.FD_ORDER)
        self.ny = align_ny(600, self.BDIMY, self.FD_ORDER)
        self.blkx = (int)((self.nx - 2 * self.FD_ORDER) / (2 * self.BDIMX))
        self.blky = (int)((self.ny - 2 * self.FD_ORDER) / self.BDIMY)

        self.nz = 200
        self.delta = 25.0
        self.dt = 0.3 * 1000.0 * self.delta / 4500.0
        self.tmax_propag = 1000.0
        self.nt = int(self.tmax_propag / self.dt)
        self.freqMax = 3.5 * 1000.0 / (4.0 * self.delta)
        print(
            "dt= ",
            self.dt,
            " delta= ",
            self.delta,
            " nt= ",
            self.nt,
            " freq max= ",
            self.freqMax,
        )


#
# this class contains all the kernels to be used bu propagator
#
class cudaKernels:
    def __init__(self, cntx):
        checkCudaErrors(cuda.cuInit(0))
        checkCudaErrors(cuda.cuCtxSetCurrent(cntx))
        dev = checkCudaErrors(cuda.cuCtxGetDevice())

        self.kernelHelper = common.KernelHelper(isoPropagator, int(dev))

        # kernel to create a source fnction with some max frequency
        self.creatSource = self.kernelHelper.getFunction(b"createSource")
        # create a velocity to try things: just a sphere on the middle 4500 m/s and 2500 m/s all around
        self.createVelocity = self.kernelHelper.getFunction(b"createVelocity")

        # kernel to propagate the wavefield by 1 step in time
        self.fdPropag = self.kernelHelper.getFunction(b"fwd_3D_orderX2k")

        # kernel to propagate the wavefield by 1 step in time
        self.injectSource = self.kernelHelper.getFunction(b"injectSource")


#
# this class contains: propagator, source creation, velocity creation
# injection of data and domain exchange
#
class propagator:
    def __init__(self, params, _dev):
        print("init object for device ", _dev)
        self.dev = _dev

        checkCudaErrors(cuda.cuInit(0))
        self.cuDevice = checkCudaErrors(cuda.cuDeviceGet(_dev))
        self.context = checkCudaErrors(cuda.cuCtxCreate(None, 0, self.cuDevice))
        self.waveOut = 0
        self.waveIn = 0
        self.streamCenter = checkCudaErrors(cuda.cuStreamCreate(0))
        self.streamHalo = checkCudaErrors(cuda.cuStreamCreate(0))
        self.params = params

    def __del__(self):
        checkCudaErrors(cuda.cuCtxSetCurrent(self.context))
        checkCudaErrors(cuda.cuStreamDestroy(self.streamHalo))
        checkCudaErrors(cuda.cuStreamDestroy(self.streamCenter))
        if self.waveIn != 0:
            checkCudaErrors(cuda.cuMemFree(self.waveIn))
        if self.waveOut != 0:
            checkCudaErrors(cuda.cuMemFree(self.waveOut))
        checkCudaErrors(cuda.cuCtxDestroy(self.context))

    #
    # swap waveIn with waveOut
    #
    def swap(self):
        if verbose_prints:
            print("swap in out ", int(self.waveIn), " ", int(self.waveOut))
        i = int(self.waveIn)
        j = int(self.waveOut)
        a = i
        i = j
        j = a
        self.waveIn = cuda.CUdeviceptr(i)
        self.waveOut = cuda.CUdeviceptr(j)

    #
    # allocate the device memory
    #
    def allocate(self):
        nel = self.params.nx * self.params.ny * self.params.nz
        n = np.array(nel, dtype=np.uint32)

        bufferSize = n * np.dtype(np.float32).itemsize
        checkCudaErrors(cuda.cuCtxSetCurrent(self.context))

        self.velocity = checkCudaErrors(cuda.cuMemAlloc(bufferSize))
        checkCudaErrors(cuda.cuMemsetD32(self.velocity, 0, n))

        nel += self.params.lead
        n = np.array(nel, dtype=np.uint32)  ## we need to align at the beginning of the tile

        bufferSize = n * np.dtype(np.float32).itemsize
        self.waveIn = checkCudaErrors(cuda.cuMemAlloc(bufferSize))
        checkCudaErrors(cuda.cuMemsetD32(self.waveIn, 0, n))

        self.waveOut = checkCudaErrors(cuda.cuMemAlloc(bufferSize))
        checkCudaErrors(cuda.cuMemsetD32(self.waveOut, 0, n))

        n = np.array(self.params.nt, dtype=np.uint32)
        bufferSize = n * np.dtype(np.float32).itemsize
        self.source = checkCudaErrors(cuda.cuMemAlloc(bufferSize))
        checkCudaErrors(cuda.cuMemsetD32(self.source, 0, n))

    #
    # create source data
    #
    def createSource(self, kernel):
        print("creating source on device ", self.dev)

        buf = np.array([int(self.source)], dtype=np.uint64)
        nt = np.array(self.params.nt, dtype=np.uint32)
        dt = np.array(self.params.dt, dtype=np.float32)
        freq = np.array(self.params.freqMax, dtype=np.float32)

        args = [buf, dt, freq, nt]
        argsp = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
        checkCudaErrors(cuda.cuCtxSetCurrent(self.context))
        checkCudaErrors(
            cuda.cuLaunchKernel(
                kernel.creatSource,
                1,
                1,
                1,  # grid dim
                1024,
                1,
                1,  # block dim
                0,
                self.streamHalo,  # shared mem and stream
                argsp.ctypes.data,
                0,
            )
        )  # arguments
        checkCudaErrors(cuda.cuStreamSynchronize(self.streamHalo))

    #
    # inject source function: ony on the domain 0
    #
    def injectSource(self, kernel, iter):
        checkCudaErrors(cuda.cuCtxSetCurrent(self.context))

        if self.dev != 0:
            return

        wavein = np.array([int(self.waveIn)], dtype=np.uint64)
        src = np.array([int(self.source)], dtype=np.uint64)
        offset_sourceInject = (
            self.params.lead
            + (int)(self.params.nz / 2) * self.params.nx * self.params.ny
            + (int)(self.params.ny / 2) * self.params.nx
            + (int)(self.params.nx / 2)
        )
        offset_sourceInject *= np.dtype(np.float32).itemsize

        np_it = np.array(iter, dtype=np.uint32)

        args = [wavein + offset_sourceInject, src, np_it]
        argsp = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
        checkCudaErrors(
            cuda.cuLaunchKernel(
                kernel.injectSource,
                1,
                1,
                1,  # grid dim
                1,
                1,
                1,  # block dim
                0,
                self.streamHalo,  # shared mem and stream
                argsp.ctypes.data,
                0,
            )
        )  # arguments

    #
    # create velocity
    #
    def createVelocity(self, kernel):
        print("running create velocity on device ", self.dev)

        offset_velocity = (
            self.params.FD_ORDER * self.params.nx * self.params.ny
            + self.params.FD_ORDER * self.params.nx
            + self.params.FD_ORDER
        )
        offset_velocity *= np.dtype(np.float32).itemsize

        vel = np.array([int(self.velocity)], dtype=np.uint64)
        dx_dt2 = (self.params.dt * self.params.dt) / (self.params.delta * self.params.delta)

        stride = self.params.nx * self.params.ny
        np_dx_dt2 = np.array(dx_dt2, dtype=np.float32)
        np_nz = np.array((self.params.nz - 2 * self.params.FD_ORDER), dtype=np.uint32)
        np_nx = np.array(self.params.nx, dtype=np.uint32)
        np_stride = np.array(stride, dtype=np.uint32)

        args = [vel + offset_velocity, np_dx_dt2, np_nz, np_nx, np_stride]
        argsp = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

        checkCudaErrors(cuda.cuCtxSetCurrent(self.context))

        # do halo up
        checkCudaErrors(
            cuda.cuLaunchKernel(
                kernel.createVelocity,
                self.params.blkx,
                self.params.blky,
                1,  # grid dim
                2 * self.params.BDIMX,
                self.params.BDIMY,
                1,  # block dim
                0,
                self.streamHalo,  # shared mem and stream
                argsp.ctypes.data,
                0,
            )
        )  # arguments
        checkCudaErrors(cuda.cuStreamSynchronize(self.streamHalo))

    #
    # execute the center part of propagation
    #
    def executeCenter(self, kernel):
        if verbose_prints:
            print("running center on device ", self.dev)
        checkCudaErrors(cuda.cuCtxSetCurrent(self.context))
        offset_velocity = (
            2 * self.params.FD_ORDER * self.params.nx * self.params.ny
            + self.params.FD_ORDER * self.params.nx
            + self.params.FD_ORDER
        )

        offset_wave = self.params.lead + offset_velocity

        offset_wave *= np.dtype(np.float32).itemsize
        offset_velocity *= np.dtype(np.float32).itemsize

        wavein = np.array([int(self.waveIn)], dtype=np.uint64)
        waveout = np.array([int(self.waveOut)], dtype=np.uint64)

        vel = np.array([int(self.velocity)], dtype=np.uint64)
        stride = self.params.nx * self.params.ny
        np_nz = np.array(self.params.nz - 4 * self.params.FD_ORDER, dtype=np.uint32)
        np_nx = np.array(self.params.nx, dtype=np.uint32)
        np_stride = np.array(stride, dtype=np.uint32)

        args = [
            wavein + offset_wave,
            waveout + offset_wave,
            vel + offset_velocity,
            np_nz,
            np_nx,
            np_stride,
        ]
        argsp = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

        # do center propagation from 2 * fd_order to nz - 2 * fd_order
        checkCudaErrors(
            cuda.cuLaunchKernel(
                kernel.fdPropag,
                self.params.blkx,
                self.params.blky,
                1,  # grid dim
                self.params.BDIMX,
                self.params.BDIMY,
                1,  # block dim
                0,
                self.streamCenter,  # shared mem and stream
                argsp.ctypes.data,
                0,
            )
        )  # arguments

    #
    # execute the halo part of propagation
    #
    def executeHalo(self, kernel):
        if verbose_prints:
            print("running halos on device ", self.dev)
        checkCudaErrors(cuda.cuCtxSetCurrent(self.context))

        offset_velocity = (
            self.params.FD_ORDER * self.params.nx * self.params.ny
            + self.params.FD_ORDER * self.params.nx
            + self.params.FD_ORDER
        )

        offset_wave = self.params.lead + offset_velocity

        offset_wave *= np.dtype(np.float32).itemsize
        offset_velocity *= np.dtype(np.float32).itemsize

        wavein = np.array([int(self.waveIn)], dtype=np.uint64)
        waveout = np.array([int(self.waveOut)], dtype=np.uint64)

        vel = np.array([int(self.velocity)], dtype=np.uint64)
        stride = self.params.nx * self.params.ny
        np_nz = np.array(self.params.FD_ORDER, dtype=np.uint32)
        np_nx = np.array(self.params.nx, dtype=np.uint32)
        np_stride = np.array(stride, dtype=np.uint32)

        args = [
            wavein + offset_wave,
            waveout + offset_wave,
            vel + offset_velocity,
            np_nz,
            np_nx,
            np_stride,
        ]
        argsp = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

        # do halo up
        checkCudaErrors(
            cuda.cuLaunchKernel(
                kernel.fdPropag,
                self.params.blkx,
                self.params.blky,
                1,  # grid dim
                self.params.BDIMX,
                self.params.BDIMY,
                1,  # block dim
                0,
                self.streamHalo,  # shared mem and stream
                argsp.ctypes.data,
                0,
            )
        )  # arguments

        # do halo down
        offset_velocity = (
            (self.params.nz - 2 * self.params.FD_ORDER) * self.params.nx * self.params.ny
            + self.params.FD_ORDER * self.params.nx
            + self.params.FD_ORDER
        )
        offset_wave = self.params.lead + offset_velocity

        offset_wave *= np.dtype(np.float32).itemsize
        offset_velocity *= np.dtype(np.float32).itemsize

        args = [
            wavein + offset_wave,
            waveout + offset_wave,
            vel + offset_velocity,
            np_nz,
            np_nx,
            np_stride,
        ]
        argsp = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
        checkCudaErrors(
            cuda.cuLaunchKernel(
                kernel.fdPropag,
                self.params.blkx,
                self.params.blky,
                1,  # grid dim
                self.params.BDIMX,
                self.params.BDIMY,
                1,  # block dim
                0,
                self.streamHalo,  # shared mem and stream
                argsp.ctypes.data,
                0,
            )
        )  # arguments

    #
    # exchange the halos
    #
    def exchangeHalo(self, propag):
        if verbose_prints:
            print("exchange  halos on device ", self.dev, "with dev ", propag.dev)
        checkCudaErrors(cuda.cuCtxSetCurrent(self.context))

        #
        # the following variables don't change
        #
        nstride = self.params.nx * self.params.ny

        devS = self.context
        devD = propag.context

        n_exch = self.params.FD_ORDER * nstride
        n_exch *= np.dtype(np.float32).itemsize

        if self.dev < propag.dev:
            # exchange up
            offsetS = self.params.lead + (self.params.nz - 2 * self.params.FD_ORDER) * nstride
            offsetD = propag.params.lead

            offsetS *= np.dtype(np.float32).itemsize
            offsetD *= np.dtype(np.float32).itemsize

            waveD = cuda.CUdeviceptr(int(propag.waveOut) + offsetD)
            waveS = cuda.CUdeviceptr(int(self.waveOut) + offsetS)

            checkCudaErrors(cuda.cuMemcpyPeerAsync(waveD, devD, waveS, devS, n_exch, self.streamHalo))
        else:
            # exchange down
            offsetS = self.params.lead + self.params.FD_ORDER * nstride
            offsetD = propag.params.lead + (propag.params.nz - propag.params.FD_ORDER) * nstride

            offsetS *= np.dtype(np.float32).itemsize
            offsetD *= np.dtype(np.float32).itemsize

            waveD = cuda.CUdeviceptr(int(propag.waveOut) + offsetD)
            waveS = cuda.CUdeviceptr(int(self.waveOut) + offsetS)

            checkCudaErrors(cuda.cuMemcpyPeerAsync(waveD, devD, waveS, devS, n_exch, self.streamHalo))

    #
    # sync stream
    #
    def syncStream(self, stream):
        checkCudaErrors(cuda.cuCtxSetCurrent(self.context))
        checkCudaErrors(cuda.cuStreamSynchronize(stream))


def main():
    checkCudaErrors(cuda.cuInit(0))

    # Number of GPUs
    print("Checking for multiple GPUs...")
    gpu_n = checkCudaErrors(cuda.cuDeviceGetCount())
    print(f"CUDA-capable device count: {gpu_n}")

    if gpu_n < 2:
        print("Two or more GPUs with Peer-to-Peer access capability are required")
        return

    prop = [checkCudaErrors(cudart.cudaGetDeviceProperties(i)) for i in range(gpu_n)]
    # Check possibility for peer access
    print("\nChecking GPU(s) for support of peer to peer memory access...")

    p2pCapableGPUs = [-1, -1]
    for i in range(gpu_n):
        p2pCapableGPUs[0] = i
        for j in range(gpu_n):
            if i == j:
                continue
            i_access_j = checkCudaErrors(cudart.cudaDeviceCanAccessPeer(i, j))
            j_access_i = checkCudaErrors(cudart.cudaDeviceCanAccessPeer(j, i))
            print(
                "> Peer access from {} (GPU{}) -> {} (GPU{}) : {}\n".format(
                    prop[i].name, i, prop[j].name, j, "Yes" if i_access_j else "No"
                )
            )
            print(
                "> Peer access from {} (GPU{}) -> {} (GPU{}) : {}\n".format(
                    prop[j].name, j, prop[i].name, i, "Yes" if i_access_j else "No"
                )
            )
            if i_access_j and j_access_i:
                p2pCapableGPUs[1] = j
                break
        if p2pCapableGPUs[1] != -1:
            break

    if p2pCapableGPUs[0] == -1 or p2pCapableGPUs[1] == -1:
        print("Two or more GPUs with Peer-to-Peer access capability are required.")
        print("Peer to Peer access is not available amongst GPUs in the system, waiving test.")
        return

    # Use first pair of p2p capable GPUs detected
    gpuid = [p2pCapableGPUs[0], p2pCapableGPUs[1]]

    #
    # init device
    #
    pars = params()

    #
    # create propagators
    #
    propags = []
    kerns = []

    #
    # create kernels and propagators that are going to be used on device
    #
    for i in gpuid:
        p = propagator(pars, i)
        k = cudaKernels(p.context)
        propags.append(p)
        kerns.append(k)

    # allocate resources in device
    for propag, kern in zip(propags, kerns):
        propag.allocate()
        propag.createSource(kern)
        propag.createVelocity(kern)

    #
    # loop over time iterations
    #
    start = time.time()
    for it in range(pars.nt):
        for propag in propags:
            propag.syncStream(propag.streamHalo)

        for propag, kern in zip(propags, kerns):
            propag.injectSource(kern, it)

        for propag, kern in zip(propags, kerns):
            propag.executeHalo(kern)

        for propag in propags:
            propag.syncStream(propag.streamHalo)

        propags[1].exchangeHalo(propags[0])

        propags[0].exchangeHalo(propags[1])

        for propag, kern in zip(propags, kerns):
            propag.executeCenter(kern)

        for propag in propags:
            propag.syncStream(propag.streamCenter)

        for propag in propags:
            propag.swap()

    end = time.time()
    npoints = (pars.nz - 2 * pars.FD_ORDER) * (pars.blkx * 2 * pars.BDIMX) * (pars.blky * pars.BDIMY)

    nops = 1.0e-9 * pars.nt * npoints / (end - start)

    print("this code generates ", nops, " GPoints/sec / device ")

    #
    # get the result out of gpu
    #
    nz = 2 * (int)(pars.nz - 2 * pars.FD_ORDER)
    print(" nz= ", nz, " nx= ", pars.nx)
    hOut = np.zeros((nz, pars.nx), dtype="float32")

    istart = 0
    for propag in propags:
        checkCudaErrors(cuda.cuCtxSetCurrent(propag.context))
        offset = pars.lead + pars.FD_ORDER * pars.nx * pars.ny + (int)(pars.ny / 2) * pars.nx

        for j in range(pars.nz - 2 * pars.FD_ORDER):
            ptr = cuda.CUdeviceptr(int(propag.waveOut) + offset * 4)

            checkCudaErrors(
                cuda.cuMemcpyDtoH(
                    hOut[istart].ctypes.data,
                    ptr,
                    pars.nx * np.dtype(np.float32).itemsize,
                )
            )
            offset += pars.nx * pars.ny
            istart += 1

    #
    #  delete kernels and propagatrs
    #
    for propag in propags:
        del propag

    if display_graph:
        nrows = nz
        ncols = pars.nx
        dbz = hOut
        dbz = np.reshape(dbz, (nrows, ncols))

        ##
        ## those are to plot results
        ##
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        title = "test fd kernels up to " + str(pars.tmax_propag) + " ms "
        plt.title(title, fontsize=20)
        im = ax.imshow(
            dbz,
            interpolation="bilinear",
            cmap=plt.get_cmap("Greys"),
            aspect="auto",
            origin="upper",
            extent=[1, pars.nx, nz, 1],
            vmax=abs(dbz).max(),
            vmin=-abs(dbz).max(),
        )

        fig.colorbar(im, ax=ax)

        plt.show()

    print("Done")


if __name__ == "__main__":
    display_graph = True
    verbose_prints = True
    main()
