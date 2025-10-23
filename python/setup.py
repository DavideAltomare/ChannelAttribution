# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import sys
from pathlib import Path

# --- RELATIVE paths (important for sdist/isolated wheel builds)
pyx_file = "src/cypack/ChannelAttribution.pyx"
cpp_file = "src/cypack/functions.cpp"
armadillo_inc = "src/cypack/armadillo-9.860.2/include"

def numpy_include():
    # Build isolation will install oldest-supported-numpy per pyproject [build-system].requires
    import numpy as np
    return np.get_include()

# Minimal, portable compile flags (no BLAS/LAPACK)
extra_compile_args = ["/std:c++17", "/O2", "/DNOMINMAX"] if sys.platform == "win32" else ["-std=c++17", "-O3"]

ext = Extension(
    name="ChannelAttribution",
    sources=[pyx_file, cpp_file],                   # <- relative
    include_dirs=[armadillo_inc, numpy_include()],  # <- relative + numpy
    language="c++",
    extra_compile_args=extra_compile_args,
)

extensions = cythonize(
    [ext],
    compiler_directives={"language_level": "3", "embedsignature": True},
)

# Optional long_description if README.md exists next to this file
long_desc = ""
readme = Path(__file__).with_name("README.md")
if readme.exists():
    long_desc = readme.read_text(encoding="utf-8")

setup(
    name="ChannelAttribution",
    version="2.2.2",
    ext_modules=extensions,
    install_requires=["numpy>=1.22", "pandas>=1.5"],
    long_description=long_desc or None,
    long_description_content_type=("text/markdown" if long_desc else None),
)
