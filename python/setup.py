# setup.py

import sys
from pathlib import Path

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize


# -------------------------
# Paths (relative to this file)
# -------------------------

# Cython + C++ core sources
pyx_file = "src/cypack/ChannelAttribution.pyx"
cpp_file = "src/cypack/functions.cpp"
armadillo_inc = "src/cypack/armadillo-9.860.2/include"


def numpy_include():
    # Build isolation: oldest-supported-numpy is installed via pyproject.toml
    import numpy as np
    return np.get_include()


# -------------------------
# Compile flags
# -------------------------

if sys.platform == "win32":
    extra_compile_args = ["/std:c++17", "/O2", "/DNOMINMAX"]
else:
    extra_compile_args = ["-std=c++17", "-O3"]

# -------------------------
# Extension: core module
# -------------------------
# IMPORTANT:
# - We now build the core as "ChannelAttribution._core"
# - The pure-Python package "ChannelAttribution" will live in src/ChannelAttribution
#   and import from ._core plus install_pro helpers.
# -------------------------

ext = Extension(
    name="ChannelAttribution._core",              # <- compiled submodule
    sources=[pyx_file, cpp_file],
    include_dirs=[armadillo_inc, numpy_include()],
    language="c++",
    extra_compile_args=extra_compile_args,
)

extensions = cythonize(
    [ext],
    compiler_directives={
        "language_level": "3",
        "embedsignature": True,
    },
)

# -------------------------
# Long description (optional)
# -------------------------

long_desc = ""
readme = Path(__file__).with_name("README.md")
if readme.exists():
    long_desc = readme.read_text(encoding="utf-8")

# -------------------------
# Setup
# -------------------------
# EXPECTED LAYOUT:
#   src/
#     ChannelAttribution/
#       __init__.py         (imports from ._core and from .install_pro)
#       install_pro.py      (contains install_pro / install_pro_222 in pure Python)
#     cypack/
#       ChannelAttribution.pyx
#       functions.cpp
#       armadillo-9.860.2/...
# -------------------------

setup(
    name="ChannelAttribution",
    version="2.2.4",  # bump version so wheels/env pick up the new layout
    package_dir={"": "src"},
    packages=find_packages("src"),
    ext_modules=extensions,
    install_requires=[
        "numpy>=1.22",
        "pandas>=1.5",
    ],
    long_description=long_desc or None,
    long_description_content_type=("text/markdown" if long_desc else None),
    zip_safe=False,
)
