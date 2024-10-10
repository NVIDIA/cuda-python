from cuda.py._compiler import Compiler

def test_compiler_initialization():
    code = "__device__ int test_func() { return 0; }"
    compiler = Compiler(code, "c++")
    assert compiler._handle is not None
    assert compiler._backend == "nvrtc"