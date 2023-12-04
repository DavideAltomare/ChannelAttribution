from setuptools import setup
from Cython.Distutils import build_ext, Extension

extensions = [Extension(name="ChannelAttribution", 
                        sources=["src/cypack/ChannelAttribution.pyx", "src/cypack/functions.cpp"],
						include_dirs=["src/cypack/armadillo-9.860.2/include"], 
						language='c++', 
						extra_compile_args=['-std=c++11'],
						cython_directives={"language_level":'3',"embedsignature": True})
]

setup(
	ext_modules = extensions,    
	cmdclass={'build_ext': build_ext},
	install_requires=['numpy', 'pandas']
)
