import datetime
import os

from cuda.bindings import driver, nvrtc, runtime
from cuda.core.experimental import _utils

err, _DRIVER_VERSION = driver.cuDriverGetVersion()
assert err == driver.CUresult.CUDA_SUCCESS

_BINDING_VERSION = _utils.get_binding_version()

_IMPORT_TIME = str(datetime.datetime.now())


def test_driver_error_info():
    label = f"{_IMPORT_TIME} {os.name=} {_BINDING_VERSION=} {_DRIVER_VERSION=}"
    print(f"\n{label} ENTRY", flush=True)
    expl_dict = _utils._DRIVER_CU_RESULT_EXPLANATIONS
    valid_codes = set()
    for code in range(1000):
        try:
            error = driver.CUresult(code)
        except ValueError:
            if _BINDING_VERSION >= (12, 0):
                assert code not in expl_dict
        else:
            assert code in expl_dict
            valid_codes.add(code)
            print(code)
            print(error, flush=True)
            if code not in (226, 721, 916):  # These trigger SegFaults
                name, desc, expl = _utils._driver_error_info(error)
                print(name)
                print(desc)
                print(expl, flush=True)
            print(flush=True)
    if _BINDING_VERSION >= (12, 0):
        extra_expl_codes = sorted(set(expl_dict.keys()) - valid_codes)
        assert not extra_expl_codes
    missing_expl_codes = sorted(valid_codes - set(expl_dict.keys()))
    assert not missing_expl_codes
    print(f"{label} DONE\n", flush=True)


def test_runtime_error_info():
    expl_dict = _utils._RUNTIME_CUDA_ERROR_T_EXPLANATIONS
    valid_codes = set()
    for code in range(100000):
        try:
            error = runtime.cudaError_t(code)
        except ValueError:
            if _BINDING_VERSION >= (12, 0):
                assert code not in expl_dict
        else:
            assert code in expl_dict
            valid_codes.add(code)
            print(code)
            print(error)
            if True:
                name, desc, expl = _utils._runtime_error_info(error)
                print(name)
                print(desc)
                print(expl)
            print()
    if _BINDING_VERSION >= (12, 0):
        extra_expl_codes = sorted(set(expl_dict.keys()) - valid_codes)
        assert not extra_expl_codes
    missing_expl_codes = sorted(valid_codes - set(expl_dict.keys()))
    assert not missing_expl_codes


def test_nvrtc_error_info():
    for code in range(100):
        try:
            error = nvrtc.nvrtcResult(code)
        except ValueError:
            pass
        else:
            print(code)
            print(error)
            if True:
                desc = _utils._nvrtc_error_info(error)
                print(desc)
            print()
