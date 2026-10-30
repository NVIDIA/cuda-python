# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# distutils: include_dirs = ../../cuda/core

"""Deterministic raw-handle reuse tests against the real graph registry.

Parents are real empty CUDA graphs. Stack-local addresses stand in for reused
child and owner handles only inside the metadata API; they are never passed to
CUDA. Distinct addresses per invocation also keep parallel workers independent.
The parents retain ownership of their real graphs and destroy them normally.
"""

from libc.stdint cimport uintptr_t
from cuda.bindings.cydriver cimport CUgraph, CUgraphNode
from cuda.core._rt cimport (
    GraphHandle,
    as_cu,
    create_child_graph_handle,
    invalidate_child_graph_state,
)
from cuda.core.graph._graph_definition cimport GraphDefinition

import pytest


@pytest.mark.agent_authored(model="gpt-6")
def test_child_graph_registry_rejects_foreign_parent(init_cuda):
    """Reusing both raw addresses must not alias an independent parent tree."""
    cdef uintptr_t tokens[2]
    cdef CUgraph raw_child = <CUgraph>&tokens[0]
    cdef CUgraphNode owner = <CUgraphNode>&tokens[1]
    cdef GraphDefinition old_parent = GraphDefinition()
    cdef GraphDefinition new_parent = GraphDefinition()
    cdef GraphHandle old_child = create_child_graph_handle(
        raw_child, old_parent._h_graph, owner
    )
    cdef GraphHandle same_child = create_child_graph_handle(
        raw_child, old_parent._h_graph, owner
    )
    assert same_child.get() == old_child.get()

    cdef GraphHandle new_child = create_child_graph_handle(
        raw_child, new_parent._h_graph, owner
    )
    assert new_child.get() != old_child.get()
    assert as_cu(new_child) == raw_child

    invalidate_child_graph_state(old_parent._h_graph, owner)
    assert as_cu(old_child) == NULL
    assert as_cu(same_child) == NULL
    assert as_cu(new_child) == raw_child

    same_child = create_child_graph_handle(raw_child, new_parent._h_graph, owner)
    assert same_child.get() == new_child.get()


@pytest.mark.agent_authored(model="gpt-6")
def test_child_graph_registry_rejects_foreign_owner_in_same_parent(init_cuda):
    """Retiring an old owner preserves the replacement within one hierarchy."""
    cdef uintptr_t tokens[3]
    cdef CUgraph raw_child = <CUgraph>&tokens[0]
    cdef CUgraphNode old_owner = <CUgraphNode>&tokens[1]
    cdef CUgraphNode new_owner = <CUgraphNode>&tokens[2]
    cdef GraphDefinition parent = GraphDefinition()
    cdef GraphHandle old_child = create_child_graph_handle(
        raw_child, parent._h_graph, old_owner
    )
    cdef GraphHandle new_child = create_child_graph_handle(
        raw_child, parent._h_graph, new_owner
    )
    assert new_child.get() != old_child.get()

    invalidate_child_graph_state(parent._h_graph, old_owner)
    assert as_cu(old_child) == NULL
    assert as_cu(new_child) == raw_child

    cdef GraphHandle same_child = create_child_graph_handle(
        raw_child, parent._h_graph, new_owner
    )
    assert same_child.get() == new_child.get()


@pytest.mark.agent_authored(model="gpt-6")
def test_graph_hierarchy_deletion_preserves_replacement_mapping(init_cuda):
    """An old tree's destructor must not erase another tree's newer entry."""
    cdef uintptr_t tokens[3]
    cdef CUgraph raw_child = <CUgraph>&tokens[0]
    cdef CUgraphNode old_owner = <CUgraphNode>&tokens[1]
    cdef CUgraphNode new_owner = <CUgraphNode>&tokens[2]
    cdef GraphDefinition old_parent = GraphDefinition()
    cdef GraphDefinition new_parent = GraphDefinition()
    cdef GraphHandle old_child = create_child_graph_handle(
        raw_child, old_parent._h_graph, old_owner
    )
    cdef GraphHandle new_child = create_child_graph_handle(
        raw_child, new_parent._h_graph, new_owner
    )
    assert new_child.get() != old_child.get()

    # Drop every owner of the old hierarchy so its real CUDA root is destroyed.
    old_child.reset()
    old_parent = None
    assert as_cu(new_child) == raw_child

    cdef GraphHandle same_child = create_child_graph_handle(
        raw_child, new_parent._h_graph, new_owner
    )
    assert same_child.get() == new_child.get()
