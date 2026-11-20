# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""PyTorch view factories, storage ownership, and consumer-stream ordering."""

import weakref
from contextlib import closing

import numpy as np
import pytest

from cuda.core import LaunchConfig, Program, ProgramOptions, Stream, StreamOptions, launch
from cuda.core.graph import KernelNode
from cuda.core.utils import StridedMemoryView

torch = pytest.importorskip("torch", minversion="2.3")


@pytest.mark.agent_authored(model="gpt-6-sol")
@pytest.mark.parametrize("transpose", [False, True], ids=["offset-slice", "offset-transpose"])
def test_torch_array_interface_factory_preserves_storage_and_owner(transpose):
    """The CPU array-interface factory preserves strided tensor storage and its owner."""
    tensor = torch.arange(42, dtype=torch.int32).reshape(6, 7)[1:5, 1:7:2]
    if transpose:
        tensor = tensor.t()
    expected = tensor.numpy().copy()
    owner = weakref.ref(tensor)
    view = StridedMemoryView.from_array_interface(tensor)
    assert view.ptr == tensor.data_ptr()
    assert view.shape == tuple(tensor.shape)
    assert view.strides == tensor.stride()
    assert view.dtype == np.dtype(np.int32)
    assert view.device_id == -1
    assert view.is_device_accessible is False
    assert view.exporting_obj is tensor

    del tensor
    assert owner() is not None
    result = np.from_dlpack(view)
    assert result.ctypes.data == view.ptr
    np.testing.assert_array_equal(result, expected)
    owner().add_(11)
    np.testing.assert_array_equal(result, expected + 11)


@pytest.mark.agent_authored(model="gpt-6-sol")
@pytest.mark.skipif(torch.version.cuda is None, reason="CUDA-enabled PyTorch is required")
@pytest.mark.parametrize("transpose", [False, True], ids=["offset-slice", "offset-transpose"])
def test_torch_cuda_array_interface_factory_preserves_storage_and_owner(init_cuda, transpose):
    """The CUDA array-interface factory retains strided tensor storage after its name is deleted."""
    tensor = torch.arange(42, dtype=torch.int32, device=f"cuda:{init_cuda.device_id}").reshape(6, 7)[1:5, 1:7:2]
    if transpose:
        tensor = tensor.t()
    expected = np.arange(42, dtype=np.int32).reshape(6, 7)[1:5, 1:7:2]
    if transpose:
        expected = expected.T
    owner = weakref.ref(tensor)
    current = torch.cuda.current_stream(init_cuda.device_id)
    view = StridedMemoryView.from_cuda_array_interface(tensor, stream_ptr=current.cuda_stream)
    assert view.ptr == tensor.data_ptr()
    assert view.shape == tuple(tensor.shape)
    assert view.strides == tensor.stride()
    assert view.dtype == np.dtype(np.int32)
    assert view.device_id == init_cuda.device_id
    assert view.is_device_accessible is True
    assert view.exporting_obj is tensor

    del tensor
    assert owner() is not None
    with torch.cuda.device(init_cuda.device_id):
        result = torch.from_dlpack(view)
        assert result.data_ptr() == view.ptr
        result.add_(11)
        np.testing.assert_array_equal(result.cpu().numpy(), expected + 11)


@pytest.mark.agent_authored(model="gpt-6-sol")
@pytest.mark.skipif(torch.version.cuda is None, reason="CUDA-enabled PyTorch is required")
@pytest.mark.parametrize("factory", ["from_dlpack", "from_cuda_array_interface", "from_any_interface"])
def test_torch_view_factory_orders_producer_before_consumer(init_cuda, factory):
    """A factory orders the consumer kernel after the PyTorch producer writes."""
    code = r"""
    extern "C" __global__ void write_value(int* dst) {
        dst[0] = 73;
    }
    extern "C" __global__ void read_value(const int* src, int* dst) {
        dst[0] = src[0];
    }
    """
    config = LaunchConfig(grid=1, block=1)
    tensor = torch.zeros(1, dtype=torch.int32, device=f"cuda:{init_cuda.device_id}")
    output = torch.zeros_like(tensor)
    source_view = StridedMemoryView.from_dlpack(tensor, stream_ptr=-1)
    output_view = StridedMemoryView.from_dlpack(output, stream_ptr=-1)
    producer = torch.cuda.Stream(device=init_cuda.device_id)
    torch.cuda.current_stream(init_cuda.device_id).synchronize()

    with (
        closing(Program(code, "c++", ProgramOptions(arch=f"sm_{init_cuda.arch}"))) as program,
        closing(init_cuda.create_stream(options=StreamOptions(nonblocking=True))) as consumer,
        closing(Stream.from_handle(producer.cuda_stream)) as producer_stream,
        closing(init_cuda.create_event()) as fork,
        closing(init_cuda.create_event()) as join,
        closing(producer_stream.create_graph_builder()) as builder,
    ):
        module = program.compile("cubin")
        write = module.get_kernel("write_value")
        read = module.get_kernel("read_value")
        builder.begin_building()
        try:
            # Fork before the write so this event adds no write-to-read dependency.
            producer_stream.record(fork)
            consumer.wait(fork)
            try:
                launch(producer_stream, config, write, source_view.ptr)
                with torch.cuda.stream(producer):
                    view = getattr(StridedMemoryView, factory)(tensor, stream_ptr=int(consumer.handle))
                launch(consumer, config, read, view.ptr, output_view.ptr)
            finally:
                consumer.record(join)
                producer_stream.wait(join)
        finally:
            builder.end_building()

        definition = builder.graph_definition
        kernels = {node.kernel.handle: node for node in definition.nodes() if isinstance(node, KernelNode)}
        assert kernels.keys() == {write.handle, read.handle}
        pending = list(kernels[read.handle].pred)
        ancestors = set()
        while pending:
            node = pending.pop()
            if node not in ancestors:
                ancestors.add(node)
                pending.extend(node.pred)
        assert kernels[write.handle] in ancestors, "consumer kernel must wait for producer"

        with closing(builder.complete()) as graph:
            graph.launch(consumer)
            consumer.sync()
        assert output.cpu().item() == 73


@pytest.mark.agent_authored(model="gpt-6-sol")
def test_torch_view_rejects_unsupported_itemsize():
    """A tensor dtype absent from the bridge's itemsize map is rejected."""
    tensor = torch.empty(4, dtype=torch.float8_e4m3fn)
    with pytest.raises(TypeError, match="Unsupported AOTI dtype code"):
        StridedMemoryView.from_dlpack(tensor)


@pytest.mark.agent_authored(model="gpt-6-sol")
def test_torch_view_reports_unsupported_numpy_dtype_lazily():
    """A complex-half tensor retains its layout but cannot expose a NumPy dtype."""
    # Reinterpret storage to avoid PyTorch's experimental complex-half creation warning.
    tensor = torch.empty(8, dtype=torch.float16).view(torch.complex32)
    view = StridedMemoryView.from_dlpack(tensor)
    assert view.ptr == tensor.data_ptr()
    assert view.shape == (4,)
    assert view.strides == (1,)
    assert view.exporting_obj is tensor
    with pytest.raises(TypeError, match="Unsupported AOTI dtype code"):
        _ = view.dtype
