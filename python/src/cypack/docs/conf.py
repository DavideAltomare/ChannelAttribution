# Configuration file for the Sphinx documentation builder.

from __future__ import annotations
import sys
import locale
from pathlib import Path

# --- Paths -------------------------------------------------------------------
DOCS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(DOCS_DIR))

# Locale is best-effort; don't fail if missing
try:
    locale.setlocale(locale.LC_TIME, "en_US.utf8")
except Exception:
    pass

# --- Project -----------------------------------------------------------------
project = "ChannelAttribution"
author = "Davide Altomare, David Loris"
copyright = "Davide Altomare and David Loris"
release = "2.2.4"

# --- Extensions --------------------------------------------------------------
# Use napoleon (NumPy/Google docstrings) for HTML; we'll swap to numpydoc for rinoh in setup().
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "rinoh.frontend.sphinx",
]

autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# --- Napoleon: map your custom sections (HTML build) -------------------------
napoleon_custom_sections = [
    ("User Interaction", "notes"),
    ("Environment Detection", "notes"),
    ("Network Behavior", "notes"),
    ("Packages & Proxies", "notes"),
    ("Output & Errors", "notes"),
    ("Security Notes", "notes"),
    ("Example", "examples"),  # napoleon expects "Examples"
]

# --- HTML --------------------------------------------------------------------
html_theme = "nature"
_static_dir = DOCS_DIR / "_static"
html_static_path = ["_static"] if _static_dir.is_dir() else []

# --- Rinoh / PDF -------------------------------------------------------------
# Dict form avoids the auto-conversion warning.
rinoh_documents = [
    {
        "doc": "index",
        "target": "channelattribution",
        "title": "ChannelAttribution Documentation",
        "author": "Davide Altomare",
        "domain_indices": ["genindex", "py-modindex"],
    }
]

# Optional knobs
rinoh_inline_elements = True
# rinoh_paper_size = "A4"  # uncomment if you want A4 explicitly

# --- Make rinoh happy: swap napoleon -> numpydoc only for rinoh --------------
def setup(app):
    def _on_config_inited(app, config):
        # Only adjust when building the PDF with rinoh
        if getattr(app, "builder", None) and app.builder.name == "rinoh":
            exts = list(config.extensions)

            # Disable napoleon (can emit nodes rinoh struggles with)
            if "sphinx.ext.napoleon" in exts:
                exts.remove("sphinx.ext.napoleon")

            # Enable numpydoc
            if "numpydoc" not in exts:
                exts.append("numpydoc")

            config.extensions = exts

            # Map custom sections for numpydoc so it doesn't warn
            try:
                from numpydoc.docscrape import NumpyDocString
                for _sec in [
                    "User Interaction",
                    "Environment Detection",
                    "Network Behavior",
                    "Packages & Proxies",
                    "Output & Errors",
                    "Security Notes",
                ]:
                    NumpyDocString._sections[_sec] = NumpyDocString._sections["Notes"]
                NumpyDocString._sections["Example"] = NumpyDocString._sections["Examples"]
            except Exception:
                # Don't break the build if numpydoc internals change
                pass

    app.connect("config-inited", _on_config_inited)
