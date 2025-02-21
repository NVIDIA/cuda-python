from cuda.bindings import driver
from cuda.core.experimental import _utils


def test_all():
    for code in range(1000):
        try:
            error = driver.CUresult(code)
        except ValueError:
            pass
        else:
            print(code)
            print(error)
            if code not in (226, 721, 916):  # These trigger SegFaults
                name, desc, expl = _utils._driver_error_info(error)
                print(name)
                print(desc)
                print(expl)
            print()
