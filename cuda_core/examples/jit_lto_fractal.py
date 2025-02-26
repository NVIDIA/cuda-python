# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

# ################################################################################
#
# This demo aims to illustrate a couple takeaways:
#
#   1. How to use the JIT LTO feature provided by the Linker class to link multiple objects together
#   2. That linking allows for libraries to modify workflows dynamically at runtime
#
# This demo mimics a relationship between a library and a user. The user's sole responsability is to
# provide device code that generates art. Where as the library is responsible for all steps involved in
# setting up the device, launch configurations and arguments as well as linking the provided device code.
#
# Two algorithms are implemented:
#   1. A Mandelbrot set
#   2. A Julia set
#
# The user can choose which algorithm to use at runtime and generate the resulting image.
#
# ################################################################################

import argparse
import sys

import cupy as cp

from cuda.core.experimental import Device, LaunchConfig, Linker, LinkerOptions, Program, ProgramOptions, launch


# ################################################################################
#
# This Mocklibrary is responsible for all steps involved launching the device code.
#
# The user is responsible for providing the device code that will be linked into the library's workflow.
# The provided device code must contain a function with the signature `void generate_art(float* Data)`
class MockLibrary:
    def __init__(self):
        # For this mock library, the main workflow is intentially kept simple by limiting itself to only calling the
        # externally defined generate_art function. More involved libraries have the option of applying pre and post
        # processing steps before calling user-defined device code. Conversely, these responsabilities can be reversed
        # such that the library owns the bulk of the workflow while allowing users to provide customized pre/post
        # processing steps.
        code_main = r"""
        extern __device__ void generate_art(float* Data);

        extern "C"
        __global__
        void main_workflow(float* Data) {
            // Preprocessing steps can be called here
            // ...

            // Call the user-defined device code
            generate_art(Data);

            // Postprocessing steps can be called here
            // ...
        }
        """

        # Most of the launch configurations can be preemptively done before the user provides their device code
        # Therefore lets compile our main workflow device code now, and link the remaining pieces at a later time
        self.arch = "".join(f"{i}" for i in Device().compute_capability)
        self.program_options = ProgramOptions(std="c++11", arch=f"sm_{self.arch}", relocatable_device_code=True)
        self.main_object_code = Program(code_main, "c++", options=self.program_options).compile("ptx")

        # Setup device state
        self.dev = Device()
        self.dev.set_current()
        self.stream = self.dev.create_stream()

        # Setup buffer to store our results
        self.width = 1024
        self.height = 512
        self.buffer = cp.empty(self.width * self.height * 4, dtype=cp.float32)

        # Setup the launch configuration such that each thread will be generating one pixel, and subdivide
        # the problem into 16x16 chunks.
        self.grid = (self.width / 16, self.height / 16, 1)
        self.block = (16, 16, 1)
        self.config = LaunchConfig(grid=self.grid, block=self.block, stream=self.stream)

    def link_and_run_code(self, user_code):
        # First, user-defined code is compiled into a PTX object code
        user_object_code = Program(user_code, "c++", options=self.program_options).compile("ptx")

        # Then a Linker is created to link the main object code with the user-defined code
        linker_options = LinkerOptions(arch=f"sm_{self.arch}")
        linker = Linker(self.main_object_code, user_object_code, options=linker_options)
        linked_code = linker.link("cubin")

        # Now we're ready to retrieve the main device function and execute our library's workflow
        kernel = linked_code.get_kernel("main_workflow")
        launch(kernel, self.config, self.buffer.data.ptr)
        self.stream.sync()

        # Return the result buffer as a CUPY array
        return self.buffer.get().astype(float).reshape(self.height, self.width, 4)


# Now lets proceed with code from the user's perspective!
#
# ################################################################################

# Simple implementation of Mandelbrot set from Wikipedia
# http://en.wikipedia.org/wiki/Mandelbrot_set
#
# Note that this kernel is meant to be a simple, straight-forward
# implementation, and so may not represent optimized GPU code.
code_mandelbrot = r"""
__device__
void generate_art(float* Data) {
    // Which pixel am I?
    unsigned DataX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned DataY = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned Width = gridDim.x * blockDim.x;
    unsigned Height = gridDim.y * blockDim.y;

    float R, G, B, A;

    // Scale coordinates to (-2.5, 1) and (-1, 1)

    float NormX = (float)DataX / (float)Width;
    NormX *= 3.5f;
    NormX -= 2.5f;

    float NormY = (float)DataY / (float)Height;
    NormY *= 2.0f;
    NormY -= 1.0f;

    float X0 = NormX;
    float Y0 = NormY;

    float X = 0.0f;
    float Y = 0.0f;

    unsigned Iter = 0;
    unsigned MaxIter = 1000;

    // Iterate
    while(X*X + Y*Y < 4.0f && Iter < MaxIter) {
        float XTemp = X*X - Y*Y + X0;
        Y = 2.0f*X*Y + Y0;

        X = XTemp;

        Iter++;
    }

    unsigned ColorG = Iter % 50;
    unsigned ColorB = Iter % 25;

    R = 0.0f;
    G = (float)ColorG / 50.0f;
    B = (float)ColorB / 25.0f;
    A = 1.0f;

    Data[DataY*Width*4+DataX*4+0] = R;
    Data[DataY*Width*4+DataX*4+1] = G;
    Data[DataY*Width*4+DataX*4+2] = B;
    Data[DataY*Width*4+DataX*4+3] = A;
}
"""

# Simple implementation of Julia set from Wikipedia
# http://en.wikipedia.org/wiki/Julia_set
#
# Note that this kernel is meant to be a simple, straight-forward
# implementation, and so may not represent optimized GPU code.
code_julia = r"""
__device__
void generate_art(float* Data) {
    // Which pixel am I?
    unsigned DataX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned DataY = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned Width = gridDim.x * blockDim.x;
    unsigned Height = gridDim.y * blockDim.y;

    float R, G, B, A;

    // Scale coordinates to (-2, 2) for both x and y
    // Scale coordinates to (-2.5, 1) and (-1, 1)
    float X = (float)DataX / (float)Width;
    X *= 4.0f;
    X -= 2.0f;

    float Y = (float)DataY / (float)Height;
    Y *= 2.0f;
    Y -= 1.0f;

    // Julia set uses a fixed constant C
    float Cx = -0.8f;  // Try different values for different patterns
    float Cy = 0.156f;   // Try different values for different patterns

    unsigned Iter = 0;
    unsigned MaxIter = 1000;

    // Iterate
    while(X*X + Y*Y < 4.0f && Iter < MaxIter) {
        float XTemp = X*X - Y*Y + Cx;
        Y = 2.0f*X*Y + Cy;
        X = XTemp;
        Iter++;
    }

    unsigned ColorG = Iter % 50;
    unsigned ColorB = Iter % 25;

    R = 0.0f;
    G = (float)ColorG / 50.0f;
    B = (float)ColorB / 25.0f;
    A = 1.0f;

    Data[DataY*Width*4+DataX*4+0] = R;
    Data[DataY*Width*4+DataX*4+1] = G;
    Data[DataY*Width*4+DataX*4+2] = B;
    Data[DataY*Width*4+DataX*4+3] = A;
}
"""

# Parse command line arguments
# Two different kernels are implemented with unique algorithms, and the user can choose which one should be used
# Both kernels fulfill the signature required by the MockLibrary: `void generate_art(float* Data)`
parser = argparse.ArgumentParser()
parser.add_argument(
    "--target",
    "-t",
    type=str,
    default="all",
    choices=["mandelbrot", "julia", "all"],
    help="Type of visualization to generate (mandelbrot, julia, or all)",
)
parser.add_argument(
    "--display",
    "-d",
    action="store_true",
    help="Display the generated images",
)
args = parser.parse_args()

if args.display:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("this example requires matplotlib installed in order to display the image", file=sys.stderr)
        sys.exit(0)

result_to_display = []
lib = MockLibrary()

# Process mandelbrot option
if args.target in ("mandelbrot", "all"):
    # The library will compile and link their main kernel with the provided Mandelbrot kernel
    result = lib.link_and_run_code(code_mandelbrot)
    result_to_display.append((result, "Mandelbrot"))

# Process julia option
if args.target in ("julia", "all"):
    # Likewise, the same library can be configured to instead use the provided Julia kernel
    result = lib.link_and_run_code(code_julia)
    result_to_display.append((result, "Julia"))

# Display the generated images if requested
if args.display:
    fig = plt.figure()
    for i, (image, title) in enumerate(result_to_display):
        axs = fig.add_subplot(len(result_to_display), 1, i + 1)
        axs.imshow(image)
        axs.set_title(title)
        axs.axis("off")
    plt.show()

print("done!")
