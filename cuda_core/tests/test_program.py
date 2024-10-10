from cuda.core._program import Program

def test_program_initialization():
    code = "__device__ int test_func() { return 0; }"
    program = Program(code, "c++")
    assert program._handle is not None
    assert program._backend == "nvrtc"