# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os

# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "cuda.bindings"
copyright = "2021-2024, NVIDIA"
author = "NVIDIA"

# The full version, including alpha/beta/rc tags
release = os.environ["SPHINX_CUDA_BINDINGS_VER"]


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "myst_nb", "enum_tools.autoenum"]

jupyter_execute_notebooks = "force"
numfig = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_baseurl = "docs"
html_theme = "furo"
# html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "light_logo": "logo-light-mode.png",
    "dark_logo": "logo-dark-mode.png",
    # For pydata_sphinx_theme:
    # "logo": {
    #    "image_light": "_static/logo-light-mode.png",
    #    "image_dark": "_static/logo-dark-mode.png",
    # },
    # "switcher": {
    #    "json_url": "https://nvidia.github.io/cuda-python/cuda-bindings/versions.json",
    #    "version_match": release,
    # },
    ## Add light/dark mode and documentation version switcher
    # "navbar_end": [
    #    "search-button",
    #    "theme-switcher",
    #    "version-switcher",
    #    "navbar-icon-links",
    # ],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

suppress_warnings = [
    # for warnings about multiple possible targets, see NVIDIA/cuda-python#152
    "ref.python",
]
