# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path

sys.path.append(str(Path('_ext').resolve()))

project = 'Qudi Lab on a molecule modules'
copyright = '2025, Molecular Quantum Nanophotonics Lab @ LENS'
author = 'Molecular Quantum Nanophotonics Lab @ LENS'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['autodoc2', 'plotstatemachine', 'sphinx.ext.linkcode', "myst_nb", "sphinx_exercise"]
nb_render_image_options = {
    'width':"60%"
}
nb_render_markdown_format = "myst"

autodoc2_packages = [
    {
        "path": "../../src/qudi",
        "auto_mode": False,
    },
]

autodoc2_hidden_objects = ["dunder", "private"]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'press'
html_static_path = ['_static']
html_favicon = '_static/logo.png'

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return "https://github.com/lab-on-a-Molecule-Quantum-Nanophotonics/qudi-lab-on-a-molecule-modules/src/qudi/%s.py" % filename
