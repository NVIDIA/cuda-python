from dataclasses import dataclass

from cuda import cuda, cudart
from cuda.py._utils import handle_return


@dataclass
class ContextOptions:
    pass  # TODO


class Context:

    __slots__ = ("_handle", "_id")

    def __init__(self):
        raise NotImplementedError("TODO")

    @staticmethod
    def _from_ctx(obj, dev_id):
        assert isinstance(obj, cuda.CUcontext)
        ctx = Context.__new__(Context)
        ctx._handle = obj
        ctx._id = dev_id
        return ctx
