# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import importlib
import inspect
import sys
import threading
from contextlib import ExitStack, contextmanager, nullcontext
from typing import Any, Callable

import _pytest.outcomes
import pytest

_WORKER_CONTEXT_ATTR = "_cuda_python_run_parallel_worker_context"
_PATCHED_ATTR = "_cuda_python_patched_run_parallel_worker_context"
_ORIGINAL_ATTR = "_cuda_python_original_wrap_function_parallel"


def install_run_parallel_worker_context_patch() -> bool:
    """Patch pytest-run-parallel to run optional per-worker context managers.

    Returns True when pytest-run-parallel is importable and patched, and False
    when the plugin is not installed in the active environment.
    """
    try:
        plugin = importlib.import_module("pytest_run_parallel.plugin")
    except ModuleNotFoundError as exc:
        if exc.name == "pytest_run_parallel" or exc.name.startswith("pytest_run_parallel."):
            return False
        raise

    wrap_function_parallel = getattr(plugin, "wrap_function_parallel", None)
    if wrap_function_parallel is None:
        raise RuntimeError("pytest-run-parallel does not expose wrap_function_parallel")

    if getattr(wrap_function_parallel, _PATCHED_ATTR, False):
        return True

    _validate_wrap_function_parallel(wrap_function_parallel)
    patched = _make_patched_wrap_function_parallel()
    setattr(patched, _PATCHED_ATTR, True)
    setattr(patched, _ORIGINAL_ATTR, wrap_function_parallel)
    plugin.wrap_function_parallel = patched
    return True


def mark_item_for_worker_context(item: Any, context_factory: Callable[..., Any]) -> bool:
    """Attach a worker context factory to a pytest item function.

    The factory is called in each pytest-run-parallel worker thread as:
    ``factory(thread_index=..., iteration_index=..., kwargs=...)``.
    It may mutate ``kwargs`` before yielding.
    """
    obj = getattr(item, "obj", None)
    if obj is None:
        return False

    try:
        set_worker_context_factory(obj, context_factory)
    except (AttributeError, TypeError):
        original = obj

        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return original(*args, **kwargs)

        set_worker_context_factory(wrapper, context_factory)
        item.obj = wrapper

    return True


def set_worker_context_factory(func: Callable[..., Any], context_factory: Callable[..., Any]) -> Callable[..., Any]:
    """Attach or compose a pytest-run-parallel worker context factory."""
    existing = getattr(func, _WORKER_CONTEXT_ATTR, None)
    if existing is None:
        setattr(func, _WORKER_CONTEXT_ATTR, context_factory)
    elif existing is not context_factory:
        setattr(func, _WORKER_CONTEXT_ATTR, _compose_context_factories(existing, context_factory))
    return func


def _validate_wrap_function_parallel(wrap_function_parallel: Callable[..., Any]) -> None:
    parameters = tuple(inspect.signature(wrap_function_parallel).parameters)
    expected = ("fn", "n_workers", "n_iterations")
    if parameters != expected:
        raise RuntimeError(
            f"Unsupported pytest-run-parallel wrap_function_parallel signature: expected {expected}, got {parameters}"
        )


def _make_patched_wrap_function_parallel() -> Callable[..., Any]:
    def wrap_function_parallel(fn: Callable[..., Any], n_workers: int, n_iterations: int) -> Callable[..., Any]:
        context_factory = getattr(fn, _WORKER_CONTEXT_ATTR, None)

        @functools.wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> None:
            errors = []
            skip = None
            failed = None
            barrier = threading.Barrier(n_workers)
            original_switch = sys.getswitchinterval()
            new_switch = 1e-6
            for _ in range(3):
                try:
                    sys.setswitchinterval(new_switch)
                    break
                except ValueError:
                    new_switch *= 10
            else:
                sys.setswitchinterval(original_switch)

            try:

                def closure(*args: Any, **kwargs: Any) -> None:
                    nonlocal skip, failed

                    thread_index, args = args[0], args[1:]
                    worker_kwargs = dict(kwargs)
                    if n_workers > 1:
                        if "thread_index" in worker_kwargs:
                            worker_kwargs["thread_index"] = thread_index
                        if "tmp_path" in worker_kwargs:
                            worker_kwargs["tmp_path"] = worker_kwargs["tmp_path"] / f"thread_{thread_index!s}"
                            worker_kwargs["tmp_path"].mkdir(exist_ok=True)
                        if "tmpdir" in worker_kwargs:
                            worker_kwargs["tmpdir"] = worker_kwargs["tmpdir"].ensure(
                                f"thread_{thread_index!s}", dir=True
                            )

                    for i in range(n_iterations):
                        call_kwargs = dict(worker_kwargs)
                        if "iteration_index" in call_kwargs:
                            call_kwargs["iteration_index"] = i

                        barrier.wait()
                        try:
                            with _worker_context(context_factory, thread_index, i, call_kwargs):
                                fn(*args, **call_kwargs)
                        except Warning:
                            pass
                        except Exception as e:
                            errors.append(e)
                        except _pytest.outcomes.Skipped as s:
                            skip = s.msg
                        except _pytest.outcomes.Failed as f:
                            failed = f

                workers = []
                for i in range(n_workers):
                    workers.append(threading.Thread(target=closure, args=(i, *args), kwargs=kwargs))

                num_completed = 0
                try:
                    for worker in workers:
                        worker.start()
                        num_completed += 1
                finally:
                    if num_completed < len(workers):
                        barrier.abort()

                for worker in workers:
                    worker.join()

            finally:
                sys.setswitchinterval(original_switch)

            if skip is not None:
                pytest.skip(skip)
            elif failed is not None:
                raise failed
            elif errors:
                raise errors[0]

        return inner

    return wrap_function_parallel


@contextmanager
def _worker_context(context_factory, thread_index: int, iteration_index: int, kwargs: dict):
    if context_factory is None:
        with nullcontext():
            yield
        return

    context = context_factory(thread_index=thread_index, iteration_index=iteration_index, kwargs=kwargs)
    if context is None:
        context = nullcontext()
    with context:
        yield


def _compose_context_factories(first: Callable[..., Any], second: Callable[..., Any]) -> Callable[..., Any]:
    @contextmanager
    def combined(**kwargs: Any):
        with ExitStack() as stack:
            first_context = first(**kwargs)
            if first_context is not None:
                stack.enter_context(first_context)
            second_context = second(**kwargs)
            if second_context is not None:
                stack.enter_context(second_context)
            yield

    return combined
