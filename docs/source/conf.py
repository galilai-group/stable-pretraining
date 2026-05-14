# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath("../stable_pretraining"))

from stable_pretraining.__about__ import (
    __version__,
)  # Import the version from __about__.py

project = "stable-pretraining"
copyright = "2024, stable-pretraining team"
author = "stable-pretraining team"

# The full version, including alpha/beta/rc tags
release = __version__  # Set release to the version from __about__.py

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
    "myst_parser",
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# myst-parser: register markdown headings (level 1-3) as cross-ref targets so
# in-document anchor links like ``[Foo](#foo)`` resolve. Without this, MyST
# emits ``myst.xref_missing`` warnings even though the rendered HTML *does*
# contain the auto-generated id.
myst_heading_anchors = 3

copybutton_exclude = ".linenos, .gp"

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
    "omegaconf": ("https://omegaconf.readthedocs.io/en/latest/", None),
}

templates_path = ["_templates"]
# Exclude the auto-generated ``sphinx-apidoc`` output that the CI step
# ``sphinx-apidoc -o docs/source stable_pretraining/`` drops at the source
# root. The hand-written ``api/*.rst`` already covers every module, and
# parsing both sets produces dozens of "duplicate object description"
# warnings (each symbol gets indexed twice).
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "stable_pretraining*.rst",
    "modules.rst",
]

# References that we don't want to chase. Lightning + omegaconf intersphinx
# resolves most class names; the remainder are plain English nouns that
# napoleon mistakenly treats as type names ("optional", "callable",
# "Dictionary containing ..."), or built-ins it can't link.
nitpick_ignore_regex = [
    (r"py:.*", r"^optional$"),
    (r"py:.*", r"^callable$"),
    (r"py:.*", r"^Dictionary containing.*"),
]

sphinx_gallery_conf = {
    "examples_dirs": ["../../examples/"],
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "filename_pattern": "/demo_",
    "run_stale_examples": True,
    "ignore_pattern": r"__init__\.py",
    "reference_url": {
        # The module you locally document uses None
        "sphinx_gallery": None
    },
    # directory where function/class granular galleries are stored
    "backreferences_dir": "gen_modules/backreferences",
    # Modules for which function/class level galleries are created. In
    # this case sphinx_gallery and numpy in a tuple of strings.
    "doc_module": ("stable_pretraining",),
    # objects to exclude from implicit backreferences. The default option
    # is an empty set, i.e. exclude nothing.
    "exclude_implicit_doc": {},
    "nested_sections": False,
}

# how to define macros: https://docs.mathjax.org/en/latest/input/tex/macros.html
mathjax3_config = {
    "tex": {"equationNumbers": {"autoNumber": "AMS", "useLabelIds": True}}
}

# bibliography
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "alpha"

math_numfig = True
numfig = True
numfig_secnum_depth = 3

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = []
# html_favicon =
# html_logo =
# Options accepted by sphinx_book_theme. The earlier list also held a
# bunch of sphinx_rtd_theme options (analytics_anonymize_ip, logo_only,
# prev_next_buttons_location, …) which the book theme silently rejected
# with a warning per option — removed.
html_theme_options = {
    "collapse_navigation": True,
    "navigation_depth": 4,
}

# Separator substitution : Writing |sep| in the rst file will display a horizontal line.
rst_prolog = """
.. |sep| raw:: html

   <hr />
"""
