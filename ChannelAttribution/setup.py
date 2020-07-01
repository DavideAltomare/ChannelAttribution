
#pip uninstall ChannelAttribution
#cd "C:\Users\a458057\Google Drive\Projects\ChannelAttribution\Python\ChannelAttribution\ChannelAttribution"
#python setup.py build_ext --inplace
#python setup.py sdist bdist_wheel
#pip install "C:\Users\a458057\Google Drive\Projects\ChannelAttribution\Python\ChannelAttribution\ChannelAttribution\dist\ChannelAttribution-1.18.0.tar.gz"
#pip install "C:\Users\a458057\Google Drive\Projects\ChannelAttribution\Python\ChannelAttribution\ChannelAttribution"


#from  setuptools import setup, Extension, find_packages
#from setuptools.command.build_ext import build_ext
from  setuptools import setup
from Cython.Distutils import build_ext, Extension

extensions = [Extension(name="ChannelAttribution", 
                        sources=["src/cypack/ChannelAttribution.pyx", "src/cypack/functions.cpp"],
						include_dirs=["src/armadillo-9.860.2/include"], 
						language='c++', 
						extra_compile_args=['-std=c++11'],
						cython_directives={"language_level":'3',"embedsignature": True})
]

setup(
	ext_modules = extensions,    
	cmdclass={'build_ext': build_ext},
	install_requires=['numpy', 'pandas']
)
