Setting a ``cuda_bindings`` class instance to itself
(``obj = cuda_bindings.driver.Class(); obj[0] = obj``) will no longer crash or use garbage
values. (#2821)
