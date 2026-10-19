The ``StridedMemoryView`` DLPack C exchange API now preserves the underlying
Python exception when an export or import fails, as required by the DLPack
contract, instead of returning ``-1`` with no exception set. (#2939)
