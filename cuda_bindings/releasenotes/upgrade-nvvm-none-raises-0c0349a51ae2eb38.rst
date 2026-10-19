Passing a ``None`` to ``nvvm.add_module_to_program`` or ``nvvm.lazy_add_module_to_program`` will
now raise a ``nvvm.nvvmError`` (``INVALID_INPUT``) rather than a ``ValueError``. (#2818)
