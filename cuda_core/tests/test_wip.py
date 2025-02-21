from cuda.bindings import driver, nvrtc, runtime
from cuda.core.experimental import _utils


def test_driver_error_info():
    expl_dict = _utils._DRIVER_CU_RESULT_EXPLANATIONS
    valid_codes = set()
    for code in range(1000):
        try:
            error = driver.CUresult(code)
        except ValueError:
            assert code not in expl_dict
        else:
            assert code in expl_dict
            valid_codes.add(code)
            print(code)
            print(error)
            if code not in (226, 721, 916):  # These trigger SegFaults
                name, desc, expl = _utils._driver_error_info(error)
                print(name)
                print(desc)
                print(expl)
            print()
    stray_expl_codes = sorted(set(expl_dict.keys()) - valid_codes)
    assert not stray_expl_codes
    missing_expl_codes = sorted(valid_codes - set(expl_dict.keys()))
    assert not missing_expl_codes


def test_runtime_error_info():
    expl_dict = _utils._RUNTIME_CUDA_ERROR_T_EXPLANATIONS
    valid_codes = set()
    for code in range(100000):
        try:
            error = runtime.cudaError_t(code)
        except ValueError:
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
    stray_expl_codes = sorted(set(expl_dict.keys()) - valid_codes)
    assert not stray_expl_codes
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
