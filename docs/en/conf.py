# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import subprocess
import sys

project = 'FGVClib'
copyright = '2022, yyq'
author = 'yyq'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx_markdown_tables',
    'recommonmark'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']


# The master toctree document.
master_doc = 'index'

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

language='en'

#extensions = ['recommonmark']

# -- Options for EPUB output
epub_show_urls = 'footnote'


