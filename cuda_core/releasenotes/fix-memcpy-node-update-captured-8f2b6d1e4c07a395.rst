:meth:`~graph.MemcpyNode.update` and the executable view of a memcpy node
now accept nodes recorded by stream capture. The driver stores the operands
of a captured :meth:`Buffer.copy_from` or :meth:`Buffer.copy_to` as unified
addresses, and ``update`` rejected that descriptor with a message about
multidimensional copies. Such an operand stays unified when it is replaced,
because the driver does not accept a change of an operand's memory type in
an instantiated graph. Replacing a device operand with another device
buffer of the same kind can therefore also be applied with
:meth:`~graph.Graph.update`. Replacing it with host memory, or with memory
from a different allocator, always takes effect in a new instantiation;
whether an executable update accepts it depends on how the operands were
allocated and on the driver. The driver compares the memory class of
pool-backed and virtual-memory operands and rejects the change for them,
while operands from plain ``cuMemAlloc`` allocations accept it. The node's
``repr`` now shows ``U`` for unified operands.
(`#2649 <https://github.com/NVIDIA/cuda-python/issues/2649>`__) (#3029, #3035)
