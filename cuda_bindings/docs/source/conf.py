# SPDX-FileCopyrightText: Copyright (c) 2012-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import inspect
import os
import sys
from pathlib import Path

sys.path.insert(0, str((Path(__file__).parents[3] / "cuda_python" / "docs" / "exts").absolute()))


# -- Project information -----------------------------------------------------

project = "cuda.bindings"
copyright = "2021-2025, NVIDIA"
author = "NVIDIA"

# The full version, including alpha/beta/rc tags
release = os.environ["SPHINX_CUDA_BINDINGS_VER"]


def _github_examples_ref():
    if int(os.environ.get("BUILD_PREVIEW", 0)) or int(os.environ.get("BUILD_LATEST", 0)):
        return "main"
    return f"v{release}"


GITHUB_EXAMPLES_REF = _github_examples_ref()


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_nb",
    "sphinx_copybutton",
    "release_toc",
    "release_date",
    "enum_documenter",
]

nb_execution_mode = "off"
numfig = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Include object entries (methods, attributes, etc.) in the table of contents
# This enables the "On This Page" sidebar to show class methods and properties
# Requires Sphinx 5.1+
toc_object_entries = True
toc_object_entries_show_parents = "domain"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_baseurl = "docs"
html_theme = "nvidia_sphinx_theme"
html_theme_options = {
    "switcher": {
        "json_url": "https://nvidia.github.io/cuda-python/cuda-bindings/nv-versions.json",
        "version_match": release,
    },
    # Add light/dark mode and documentation version switcher
    "navbar_center": [
        "version-switcher",
        "navbar-nav",
    ],
    # Use custom secondary sidebar that includes autodoc entries
    "secondary_sidebar_items": ["page-toc"],
    # Show more TOC levels by default
    "show_toc_level": 3,
}
if os.environ.get("CI"):
    if int(os.environ.get("BUILD_PREVIEW", 0)):
        PR_NUMBER = f"{os.environ['PR_NUMBER']}"
        PR_TEXT = f'<a href="https://github.com/NVIDIA/cuda-python/pull/{PR_NUMBER}">PR {PR_NUMBER}</a>'
        html_theme_options["announcement"] = f"<em>Warning</em>: This documentation is only a preview for {PR_TEXT}!"
    elif int(os.environ.get("BUILD_LATEST", 0)):
        html_theme_options["announcement"] = (
            "<em>Warning</em>: This documentation is built from the development branch!"
        )

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []  # ["_static"] does not exist in our environment

# skip cmdline prompts
copybutton_exclude = ".linenos, .gp"

rst_epilog = f"""
.. |cuda_bindings_github_ref| replace:: {GITHUB_EXAMPLES_REF}
"""

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "nvvm": ("https://docs.nvidia.com/cuda/libnvvm-api/", None),
    "nvjitlink": ("https://docs.nvidia.com/cuda/nvjitlink/", None),
    "cufile": ("https://docs.nvidia.com/gpudirect-storage/api-reference-guide/", None),
}

def _sanitize_generated_docstring(lines):
    doc_lines = inspect.cleandoc("\n".join(lines)).splitlines()
    if not doc_lines:
        return

    if "(" in doc_lines[0] and ")" in doc_lines[0]:
        doc_lines = doc_lines[1:]
        while doc_lines and not doc_lines[0].strip():
            doc_lines.pop(0)

    if not doc_lines:
        lines[:] = []
        return

    lines[:] = [".. code-block:: text", ""]
    lines.extend(f"   {line}" if line else "   " for line in doc_lines)


def autodoc_process_docstring(app, what, name, obj, options, lines):
    if name.startswith("cuda.bindings."):
        _sanitize_generated_docstring(lines)


def rewrite_source(app, docname, source):
    text = source[0]

    if docname.startswith("release/"):
        text = text.replace(".. module:: cuda.bindings\n\n", "", 1)

    source[0] = text

suppress_warnings = [
    # for warnings about multiple possible targets, see NVIDIA/cuda-python#152
    "ref.python",
]


def setup(app):
    app.connect("autodoc-process-docstring", autodoc_process_docstring)
    app.connect("source-read", rewrite_source)
