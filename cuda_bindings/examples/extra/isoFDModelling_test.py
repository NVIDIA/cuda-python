# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import time

import numpy as np
from common import common
from common.helper_cuda import check_cuda_errors

from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart

iso_propagator = """\
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
class Params:
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
class CudaKernels:
    def __init__(self, cntx):
        check_cuda_errors(cuda.cuInit(0))
        check_cuda_errors(cuda.cuCtxSetCurrent(cntx))
        dev = check_cuda_errors(cuda.cuCtxGetDevice())

        self.kernel_helper = common.KernelHelper(iso_propagator, int(dev))

        # kernel to create a source fnction with some max frequency
        self.creatSource = self.kernel_helper.get_function(b"createSource")
        # create a velocity to try things: just a sphere on the middle 4500 m/s and 2500 m/s all around
        self.create_velocity = self.kernel_helper.get_function(b"createVelocity")

        # kernel to propagate the wavefield by 1 step in time
        self.fdPropag = self.kernel_helper.get_function(b"fwd_3D_orderX2k")

        # kernel to propagate the wavefield by 1 step in time
        self.inject_source = self.kernel_helper.get_function(b"injectSource")


#
# this class contains: propagator, source creation, velocity creation
# injection of data and domain exchange
#
class Propagator:
    def __init__(self, params, _dev):
        print("init object for device ", _dev)
        self.dev = _dev

        check_cuda_errors(cuda.cuInit(0))
        self.cu_device = check_cuda_errors(cuda.cuDeviceGet(_dev))
        self.context = check_cuda_errors(cuda.cuCtxCreate(None, 0, self.cu_device))
        self.waveOut = 0
        self.waveIn = 0
        self.streamCenter = check_cuda_errors(cuda.cuStreamCreate(0))
        self.streamHalo = check_cuda_errors(cuda.cuStreamCreate(0))
        self.Params = params

    def __del__(self):
        check_cuda_errors(cuda.cuCtxSetCurrent(self.context))
        check_cuda_errors(cuda.cuStreamDestroy(self.streamHalo))
        check_cuda_errors(cuda.cuStreamDestroy(self.streamCenter))
        if self.waveIn != 0:
            check_cuda_errors(cuda.cuMemFree(self.waveIn))
        if self.waveOut != 0:
            check_cuda_errors(cuda.cuMemFree(self.waveOut))
        check_cuda_errors(cuda.cuCtxDestroy(self.context))

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
        nel = self.Params.nx * self.Params.ny * self.Params.nz
        n = np.array(nel, dtype=np.uint32)

        buffer_size = n * np.dtype(np.float32).itemsize
        check_cuda_errors(cuda.cuCtxSetCurrent(self.context))

        self.velocity = check_cuda_errors(cuda.cuMemAlloc(buffer_size))
        check_cuda_errors(cuda.cuMemsetD32(self.velocity, 0, n))

        nel += self.Params.lead
        n = np.array(nel, dtype=np.uint32)  ## we need to align at the beginning of the tile

        buffer_size = n * np.dtype(np.float32).itemsize
        self.waveIn = check_cuda_errors(cuda.cuMemAlloc(buffer_size))
        check_cuda_errors(cuda.cuMemsetD32(self.waveIn, 0, n))

        self.waveOut = check_cuda_errors(cuda.cuMemAlloc(buffer_size))
        check_cuda_errors(cuda.cuMemsetD32(self.waveOut, 0, n))

        n = np.array(self.Params.nt, dtype=np.uint32)
        buffer_size = n * np.dtype(np.float32).itemsize
        self.source = check_cuda_errors(cuda.cuMemAlloc(buffer_size))
        check_cuda_errors(cuda.cuMemsetD32(self.source, 0, n))

    #
    # create source data
    #
    def create_source(self, kernel):
        print("creating source on device ", self.dev)

        buf = np.array([int(self.source)], dtype=np.uint64)
        nt = np.array(self.Params.nt, dtype=np.uint32)
        dt = np.array(self.Params.dt, dtype=np.float32)
        freq = np.array(self.Params.freqMax, dtype=np.float32)

        args = [buf, dt, freq, nt]
        argsp = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
        check_cuda_errors(cuda.cuCtxSetCurrent(self.context))
        check_cuda_errors(
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
        check_cuda_errors(cuda.cuStreamSynchronize(self.streamHalo))

    #
    # inject source function: ony on the domain 0
    #
    def inject_source(self, kernel, iter):
        check_cuda_errors(cuda.cuCtxSetCurrent(self.context))

        if self.dev != 0:
            return

        wavein = np.array([int(self.waveIn)], dtype=np.uint64)
        src = np.array([int(self.source)], dtype=np.uint64)
        offset_source_inject = (
            self.Params.lead
            + (int)(self.Params.nz / 2) * self.Params.nx * self.Params.ny
            + (int)(self.Params.ny / 2) * self.Params.nx
            + (int)(self.Params.nx / 2)
        )
        offset_source_inject *= np.dtype(np.float32).itemsize

        np_it = np.array(iter, dtype=np.uint32)

        args = [wavein + offset_source_inject, src, np_it]
        argsp = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
        check_cuda_errors(
            cuda.cuLaunchKernel(
                kernel.inject_source,
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
    def create_velocity(self, kernel):
        print("running create velocity on device ", self.dev)

        offset_velocity = (
            self.Params.FD_ORDER * self.Params.nx * self.Params.ny
            + self.Params.FD_ORDER * self.Params.nx
            + self.Params.FD_ORDER
        )
        offset_velocity *= np.dtype(np.float32).itemsize

        vel = np.array([int(self.velocity)], dtype=np.uint64)
        dx_dt2 = (self.Params.dt * self.Params.dt) / (self.Params.delta * self.Params.delta)

        stride = self.Params.nx * self.Params.ny
        np_dx_dt2 = np.array(dx_dt2, dtype=np.float32)
        np_nz = np.array((self.Params.nz - 2 * self.Params.FD_ORDER), dtype=np.uint32)
        np_nx = np.array(self.Params.nx, dtype=np.uint32)
        np_stride = np.array(stride, dtype=np.uint32)

        args = [vel + offset_velocity, np_dx_dt2, np_nz, np_nx, np_stride]
        argsp = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

        check_cuda_errors(cuda.cuCtxSetCurrent(self.context))

        # do halo up
        check_cuda_errors(
            cuda.cuLaunchKernel(
                kernel.create_velocity,
                self.Params.blkx,
                self.Params.blky,
                1,  # grid dim
                2 * self.Params.BDIMX,
                self.Params.BDIMY,
                1,  # block dim
                0,
                self.streamHalo,  # shared mem and stream
                argsp.ctypes.data,
                0,
            )
        )  # arguments
        check_cuda_errors(cuda.cuStreamSynchronize(self.streamHalo))

    #
    # execute the center part of propagation
    #
    def execute_center(self, kernel):
        if verbose_prints:
            print("running center on device ", self.dev)
        check_cuda_errors(cuda.cuCtxSetCurrent(self.context))
        offset_velocity = (
            2 * self.Params.FD_ORDER * self.Params.nx * self.Params.ny
            + self.Params.FD_ORDER * self.Params.nx
            + self.Params.FD_ORDER
        )

        offset_wave = self.Params.lead + offset_velocity

        offset_wave *= np.dtype(np.float32).itemsize
        offset_velocity *= np.dtype(np.float32).itemsize

        wavein = np.array([int(self.waveIn)], dtype=np.uint64)
        waveout = np.array([int(self.waveOut)], dtype=np.uint64)

        vel = np.array([int(self.velocity)], dtype=np.uint64)
        stride = self.Params.nx * self.Params.ny
        np_nz = np.array(self.Params.nz - 4 * self.Params.FD_ORDER, dtype=np.uint32)
        np_nx = np.array(self.Params.nx, dtype=np.uint32)
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
        check_cuda_errors(
            cuda.cuLaunchKernel(
                kernel.fdPropag,
                self.Params.blkx,
                self.Params.blky,
                1,  # grid dim
                self.Params.BDIMX,
                self.Params.BDIMY,
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
    def execute_halo(self, kernel):
        if verbose_prints:
            print("running halos on device ", self.dev)
        check_cuda_errors(cuda.cuCtxSetCurrent(self.context))

        offset_velocity = (
            self.Params.FD_ORDER * self.Params.nx * self.Params.ny
            + self.Params.FD_ORDER * self.Params.nx
            + self.Params.FD_ORDER
        )

        offset_wave = self.Params.lead + offset_velocity

        offset_wave *= np.dtype(np.float32).itemsize
        offset_velocity *= np.dtype(np.float32).itemsize

        wavein = np.array([int(self.waveIn)], dtype=np.uint64)
        waveout = np.array([int(self.waveOut)], dtype=np.uint64)

        vel = np.array([int(self.velocity)], dtype=np.uint64)
        stride = self.Params.nx * self.Params.ny
        np_nz = np.array(self.Params.FD_ORDER, dtype=np.uint32)
        np_nx = np.array(self.Params.nx, dtype=np.uint32)
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
        check_cuda_errors(
            cuda.cuLaunchKernel(
                kernel.fdPropag,
                self.Params.blkx,
                self.Params.blky,
                1,  # grid dim
                self.Params.BDIMX,
                self.Params.BDIMY,
                1,  # block dim
                0,
                self.streamHalo,  # shared mem and stream
                argsp.ctypes.data,
                0,
            )
        )  # arguments

        # do halo down
        offset_velocity = (
            (self.Params.nz - 2 * self.Params.FD_ORDER) * self.Params.nx * self.Params.ny
            + self.Params.FD_ORDER * self.Params.nx
            + self.Params.FD_ORDER
        )
        offset_wave = self.Params.lead + offset_velocity

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
        check_cuda_errors(
            cuda.cuLaunchKernel(
                kernel.fdPropag,
                self.Params.blkx,
                self.Params.blky,
                1,  # grid dim
                self.Params.BDIMX,
                self.Params.BDIMY,
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
    def exchange_halo(self, propag):
        if verbose_prints:
            print("exchange  halos on device ", self.dev, "with dev ", propag.dev)
        check_cuda_errors(cuda.cuCtxSetCurrent(self.context))

        #
        # the following variables don't change
        #
        nstride = self.Params.nx * self.Params.ny

        dev_s = self.context
        dev_d = propag.context

        n_exch = self.Params.FD_ORDER * nstride
        n_exch *= np.dtype(np.float32).itemsize

        if self.dev < propag.dev:
            # exchange up
            offset_s = self.Params.lead + (self.Params.nz - 2 * self.Params.FD_ORDER) * nstride
            offset_d = propag.Params.lead

            offset_s *= np.dtype(np.float32).itemsize
            offset_d *= np.dtype(np.float32).itemsize

            wave_d = cuda.CUdeviceptr(int(propag.waveOut) + offset_d)
            wave_s = cuda.CUdeviceptr(int(self.waveOut) + offset_s)

            check_cuda_errors(cuda.cuMemcpyPeerAsync(wave_d, dev_d, wave_s, dev_s, n_exch, self.streamHalo))
        else:
            # exchange down
            offset_s = self.Params.lead + self.Params.FD_ORDER * nstride
            offset_d = propag.Params.lead + (propag.Params.nz - propag.Params.FD_ORDER) * nstride

            offset_s *= np.dtype(np.float32).itemsize
            offset_d *= np.dtype(np.float32).itemsize

            wave_d = cuda.CUdeviceptr(int(propag.waveOut) + offset_d)
            wave_s = cuda.CUdeviceptr(int(self.waveOut) + offset_s)

            check_cuda_errors(cuda.cuMemcpyPeerAsync(wave_d, dev_d, wave_s, dev_s, n_exch, self.streamHalo))

    #
    # sync stream
    #
    def sync_stream(self, stream):
        check_cuda_errors(cuda.cuCtxSetCurrent(self.context))
        check_cuda_errors(cuda.cuStreamSynchronize(stream))


def main():
    check_cuda_errors(cuda.cuInit(0))

    # Number of GPUs
    print("Checking for multiple GPUs...")
    gpu_n = check_cuda_errors(cuda.cuDeviceGetCount())
    print(f"CUDA-capable device count: {gpu_n}")

    if gpu_n < 2:
        print("Two or more GPUs with Peer-to-Peer access capability are required")
        return

    prop = [check_cuda_errors(cudart.cudaGetDeviceProperties(i)) for i in range(gpu_n)]
    # Check possibility for peer access
    print("\nChecking GPU(s) for support of peer to peer memory access...")

    p2p_capable_gp_us = [-1, -1]
    for i in range(gpu_n):
        p2p_capable_gp_us[0] = i
        for j in range(gpu_n):
            if i == j:
                continue
            i_access_j = check_cuda_errors(cudart.cudaDeviceCanAccessPeer(i, j))
            j_access_i = check_cuda_errors(cudart.cudaDeviceCanAccessPeer(j, i))
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
                p2p_capable_gp_us[1] = j
                break
        if p2p_capable_gp_us[1] != -1:
            break

    if p2p_capable_gp_us[0] == -1 or p2p_capable_gp_us[1] == -1:
        print("Two or more GPUs with Peer-to-Peer access capability are required.")
        print("Peer to Peer access is not available amongst GPUs in the system, waiving test.")
        return

    # Use first pair of p2p capable GPUs detected
    gpuid = [p2p_capable_gp_us[0], p2p_capable_gp_us[1]]

    #
    # init device
    #
    pars = Params()

    #
    # create propagators
    #
    propags = []
    kerns = []

    #
    # create kernels and propagators that are going to be used on device
    #
    for i in gpuid:
        p = Propagator(pars, i)
        k = CudaKernels(p.context)
        propags.append(p)
        kerns.append(k)

    # allocate resources in device
    for propag, kern in zip(propags, kerns):
        propag.allocate()
        propag.create_source(kern)
        propag.create_velocity(kern)

    #
    # loop over time iterations
    #
    start = time.time()
    for it in range(pars.nt):
        for propag in propags:
            propag.sync_stream(propag.streamHalo)

        for propag, kern in zip(propags, kerns):
            propag.inject_source(kern, it)

        for propag, kern in zip(propags, kerns):
            propag.execute_halo(kern)

        for propag in propags:
            propag.sync_stream(propag.streamHalo)

        propags[1].exchange_halo(propags[0])

        propags[0].exchange_halo(propags[1])

        for propag, kern in zip(propags, kerns):
            propag.execute_center(kern)

        for propag in propags:
            propag.sync_stream(propag.streamCenter)

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
    h_out = np.zeros((nz, pars.nx), dtype="float32")

    istart = 0
    for propag in propags:
        check_cuda_errors(cuda.cuCtxSetCurrent(propag.context))
        offset = pars.lead + pars.FD_ORDER * pars.nx * pars.ny + (int)(pars.ny / 2) * pars.nx

        for j in range(pars.nz - 2 * pars.FD_ORDER):
            ptr = cuda.CUdeviceptr(int(propag.waveOut) + offset * 4)

            check_cuda_errors(
                cuda.cuMemcpyDtoH(
                    h_out[istart].ctypes.data,
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
        dbz = h_out
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
