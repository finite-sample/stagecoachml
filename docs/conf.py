"""Sphinx configuration — fleet standard via py-canon, plus this repo's extras."""

from py_canon.sphinx import configure

configure(globals())

# Repo-specific additions layered on the fleet standard. The docs render an
# executable notebook (myst-nb), a "try it in your browser" button
# (sphinx-design), and a JupyterLite deployment of the notebook.
#
# myst_nb *is* myst_parser plus notebook support and registers it itself;
# leaving both in the list makes the second setup() fail outright, so the
# fleet-standard entry is swapped out rather than added to.
# `configure()` writes `extensions` into this module's namespace, so it is read
# back through globals() rather than as a bare name a linter cannot resolve.
extensions = [e for e in globals()["extensions"] if e != "myst_parser"] + [
    "sphinx.ext.githubpages",
    "myst_nb",
    "sphinx_design",
    "jupyterlite_sphinx",
]

# myst-nb supersedes myst-parser as the markdown parser, so .md must be routed
# to it or the two extensions fight over the suffix.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
    "html_image",
    "smartquotes",
    "replacements",
    "strikethrough",
    "dollarmath",
    "amsmath",
]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    # jupyterlite-sphinx copies the notebook into the Lite deployment; leaving
    # it in the toctree as well would build it twice and warn about a document
    # not included in any toctree.
    "notebooks/**",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}
autodoc_typehints_description_target = "documented"

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Notebooks are committed with outputs; executing them at build time would make
# the docs job depend on training models.
nb_execution_mode = "off"

jupyterlite_config = "jupyter_lite_config.json"
# Named as a file, not as the directory: since jupyterlite-sphinx 0.23 a
# directory keeps its name inside the Lite filesystem, which would move the
# notebook to notebooks/quickstart_interactive.ipynb and break the
# ?path=quickstart_interactive.ipynb links already published on PyPI.
jupyterlite_contents = ["notebooks/quickstart_interactive.ipynb"]
jupyterlite_bind_ipynb_suffix = False
