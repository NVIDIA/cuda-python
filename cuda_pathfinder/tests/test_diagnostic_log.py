# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for opt-in diagnostic logging.

All tests run without a GPU and without any NVIDIA library installed: the
search cascade is driven with a libname that cannot resolve, and the
enabled/disabled behaviour is exercised by reimporting the module under a
patched environment.
"""

import importlib
import logging
import os
import sys
from unittest.mock import patch

import pytest

from cuda.pathfinder._dynamic_libs.lib_descriptor import LIB_DESCRIPTORS
from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError
from cuda.pathfinder._utils import diagnostic_log

ENV_VAR = diagnostic_log.ENV_VAR_NAME
LOGGER_NAME = diagnostic_log.LOGGER_NAME


def reload_with_env(value):
    """Reimport diagnostic_log with ENV_VAR set to *value* (None to unset).

    Returns the freshly imported module. The module is restored afterwards by
    the ``restore_diagnostic_log`` fixture.
    """
    env = dict(os.environ)
    env.pop(ENV_VAR, None)
    if value is not None:
        env[ENV_VAR] = value
    with patch.dict(os.environ, env, clear=True):
        return importlib.reload(diagnostic_log)


@pytest.fixture(autouse=True)
def restore_diagnostic_log():
    """Leave the module and the logger exactly as they were found."""
    logger = logging.getLogger(LOGGER_NAME)
    saved = (logger.level, list(logger.handlers), logger.propagate)
    yield
    logger.setLevel(saved[0])
    logger.handlers[:] = saved[1]
    logger.propagate = saved[2]
    reload_with_env(None)


# ---------------------------------------------------------------------------
# enable / disable
# ---------------------------------------------------------------------------


@pytest.mark.agent_authored(model="claude-opus-5")
def test_logger_is_none_by_default():
    """Unset env var means no logger object at all, not merely a quiet one."""
    module = reload_with_env(None)
    assert module.LOGGER is None


@pytest.mark.parametrize("value", ["", "   "])
@pytest.mark.agent_authored(model="claude-opus-5")
def test_empty_value_leaves_logging_disabled(value):
    module = reload_with_env(value)
    assert module.LOGGER is None


@pytest.mark.parametrize("value", ["DEBUG", "debug", " Debug ", "INFO", "WARNING", "ERROR", "CRITICAL"])
@pytest.mark.agent_authored(model="claude-opus-5")
def test_env_var_enables_logger(value):
    module = reload_with_env(value)
    assert module.LOGGER is not None
    assert module.LOGGER.name == LOGGER_NAME
    assert module.LOGGER.level == getattr(logging, value.strip().upper())


@pytest.mark.agent_authored(model="claude-opus-5")
def test_numeric_level_accepted():
    module = reload_with_env("10")
    assert module.LOGGER is not None
    assert module.LOGGER.level == logging.DEBUG


@pytest.mark.agent_authored(model="claude-opus-5")
def test_invalid_value_warns_once_and_stays_disabled():
    """An unusable value must not raise, and must not enable partial logging."""
    with pytest.warns(UserWarning, match=ENV_VAR) as record:
        module = reload_with_env("VERBOSE")
    assert module.LOGGER is None
    assert len(record) == 1


@pytest.mark.agent_authored(model="claude-opus-5")
def test_logging_not_imported_when_disabled():
    """The disabled path must not pay for `import logging`.

    Guard: this asserts the module does not import logging *itself*. pytest has
    already imported logging, so the check is on the module's own behaviour.
    """
    source = (diagnostic_log.__file__ or "").replace(".pyc", ".py")
    with open(source) as f:
        body = f.read()
    # Every `import logging` must sit inside a function, never at module scope.
    module_level_imports = [line for line in body.splitlines() if line.startswith(("import logging", "from logging"))]
    assert module_level_imports == [], f"logging imported at module scope: {module_level_imports}"


# ---------------------------------------------------------------------------
# no side effects on the root logger
# ---------------------------------------------------------------------------


@pytest.mark.agent_authored(model="claude-opus-5")
def test_does_not_configure_root_logger():
    root = logging.getLogger()
    before = (root.level, list(root.handlers))
    module = reload_with_env("DEBUG")
    assert module.LOGGER is not None
    assert root.level == before[0]
    assert root.handlers == before[1]


@pytest.mark.agent_authored(model="claude-opus-5")
def test_attaches_only_a_null_handler():
    module = reload_with_env("DEBUG")
    assert module.LOGGER is not None
    assert any(isinstance(h, logging.NullHandler) for h in module.LOGGER.handlers)
    assert all(isinstance(h, logging.NullHandler) for h in module.LOGGER.handlers)


# ---------------------------------------------------------------------------
# the search cascade emits records
# ---------------------------------------------------------------------------


def _reload_dependents():
    """Reload the modules that captured LOGGER at import time."""
    for name in (
        "cuda.pathfinder._dynamic_libs.search_steps",
        "cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib",
    ):
        if name in sys.modules:
            importlib.reload(sys.modules[name])


@pytest.fixture
def enabled_cascade():
    """Enable logging and rebind it into the search modules."""
    reload_with_env("DEBUG")
    _reload_dependents()
    yield
    reload_with_env(None)
    _reload_dependents()


def _unresolvable_context():
    """A SearchContext for a library that cannot be found anywhere."""
    from cuda.pathfinder._dynamic_libs import search_steps

    ctx = search_steps.SearchContext(LIB_DESCRIPTORS["cudart"])
    ctx.error_messages.append("no candidate in site-packages")
    ctx.attachments.append("tried: /nonexistent/one\ntried: /nonexistent/two")
    return search_steps, ctx


@pytest.mark.agent_authored(model="claude-opus-5")
def test_failed_lookup_logs_full_candidate_list(enabled_cascade, caplog):
    """The failure path must carry the candidate list as structured fields."""
    search_steps, ctx = _unresolvable_context()
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME), pytest.raises(DynamicLibNotFoundError):
        ctx.raise_not_found()

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors, "expected an ERROR record on the failure path"
    record = errors[0]
    assert record.pathfinder_libname == "cudart"
    assert record.pathfinder_error_messages == ["no candidate in site-packages"]
    assert "/nonexistent/one" in "\n".join(record.pathfinder_attachments)


@pytest.mark.agent_authored(model="claude-opus-5")
def test_step_outcomes_are_logged(enabled_cascade, caplog):
    """Each find step reports whether it matched, with the step name attached."""
    search_steps, ctx = _unresolvable_context()

    def miss(_ctx):
        return None

    def hit(_ctx):
        return search_steps.FindResult("/opt/cuda/lib64/libcudart.so.12", "conda")

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        result = search_steps.run_find_steps(ctx, (miss, hit))

    assert result is not None
    by_step = {r.pathfinder_step: r for r in caplog.records if hasattr(r, "pathfinder_step")}
    assert by_step["miss"].pathfinder_matched is False
    assert by_step["hit"].pathfinder_matched is True
    assert by_step["hit"].pathfinder_abs_path == "/opt/cuda/lib64/libcudart.so.12"
    assert by_step["hit"].pathfinder_found_via == "conda"


@pytest.mark.agent_authored(model="claude-opus-5")
def test_successful_resolution_logs_resolved_path(enabled_cascade, caplog):
    """A resolved load emits the absolute path and how it was found."""
    from cuda.pathfinder._dynamic_libs import load_nvidia_dynamic_lib as mod
    from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL

    loaded = LoadedDL(
        abs_path="/usr/lib/x86_64-linux-gnu/libcudart.so.12",
        was_already_loaded_from_elsewhere=False,
        _handle_uint=1234,
        found_via="site-packages",
    )
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        returned = mod._log_resolved(loaded, "cudart")

    assert returned is loaded
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert infos, "expected an INFO record on the success path"
    record = infos[0]
    assert record.pathfinder_abs_path == "/usr/lib/x86_64-linux-gnu/libcudart.so.12"
    assert record.pathfinder_found_via == "site-packages"
    assert record.pathfinder_was_already_loaded is False


@pytest.mark.agent_authored(model="claude-opus-5")
def test_silent_when_disabled(caplog):
    """With the env var unset, the same call sites emit nothing."""
    reload_with_env(None)
    _reload_dependents()
    from cuda.pathfinder._dynamic_libs import load_nvidia_dynamic_lib as mod
    from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL

    loaded = LoadedDL("/x/libcudart.so.12", False, 1, "conda")
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        mod._log_resolved(loaded, "cudart")
        search_steps, ctx = _unresolvable_context()
        with pytest.raises(DynamicLibNotFoundError):
            ctx.raise_not_found()

    assert [r for r in caplog.records if r.name == LOGGER_NAME] == []


@pytest.mark.agent_authored(model="claude-opus-5")
def test_search_extra_carries_libname():
    extra = diagnostic_log.search_extra("cudart", pathfinder_found_via="conda")
    assert extra == {"pathfinder_libname": "cudart", "pathfinder_found_via": "conda"}
