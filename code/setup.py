# python setup.py build_ext --inplace --rpath=...
import os
from distutils.core import setup
from Cython.Distutils import build_ext
from Cython.Distutils import Extension

ext = [Extension('log_multi_gauss', ['log_multi_gauss.pyx'],
                 libraries=['gsl', 'gslcblas'],
                 library_dirs=[os.environ['gsldir'] + 'lib/'],
                 include_dirs=[os.environ['gsldir'] + 'include/', '.']),
       Extension('Estep', ['Estep.pyx'],
                 libraries=['gsl', 'gslcblas'],
                 library_dirs=[os.environ['gsldir'] + 'lib/'],
                 include_dirs=[os.environ['gsldir'] + 'include/', '.'])]

setup(cmdclass={'build_ext':build_ext}, ext_modules=ext)

