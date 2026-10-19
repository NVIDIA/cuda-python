:attr:`Stream.device` and related queries on a stream whose context is not
current now restore the caller's context even when the device query fails. (#2759)
