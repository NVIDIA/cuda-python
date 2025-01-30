import sys

assert len(sys.argv) == 3, "filename.ll numba|cuda-python"


def show_library_path(libname):
    with open("/proc/self/maps") as maps_file:
        for line in maps_file:
            if libname in line:
                print("/proc/self/maps:", line.strip(), flush=True)


def show_info(ir_version):
    show_library_path("libnvvm.so")
    print("ir_version:", ir_version, flush=True)
    print()


filename_ll = sys.argv[1]
with open(filename_ll) as f:
    nvvmir = f.read()


if sys.argv[2] == "numba":
    from numba.cuda.cudadrv import nvvm

    show_info(nvvm.NVVM().get_ir_version())

    ptx = nvvm.compile_ir(nvvmir).decode("utf8")
    print(ptx)


else:
    from cuda.bindings import nvvm

    show_info(nvvm.ir_version())

    prog = nvvm.create_program()
    nvvm.add_module_to_program(prog, nvvmir, len(nvvmir), filename_ll)
    try:
        nvvm.compile_program(prog, 0, [])
    except Exception as e:
        print("EXCEPTION:", e, flush=True)
    log_size = nvvm.get_program_log_size(prog)
    buffer = bytearray(log_size)
    nvvm.get_program_log(prog, buffer)
    print(buffer, flush=True)
