from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy
import sys

#if sys.platform == "linux2" :
#    include_gsl_dir = "/usr/local/include/"
#    lib_gsl_dir = "/usr/local/lib/"
#elif sys.platform == "win32":
#    include_gsl_dir = r"c:\msys\1.0\local\include"
#    lib_gsl_dir = r"c:\msys\1.0\local\lib"    

#ext = Extension("init.f_init", ["init/f_init.pyx"],
#    include_dirs=[numpy.get_include(),
#                  include_gsl_dir],
#    library_dirs=[lib_gsl_dir],
#    libraries=["gsl","gslcblas"]
#)


ext1 = Extension("postprocess.calc_spect", ["postprocess/calc_spect.pyx"])

#ext3 = Extension("postprocess.calc_gw_spect", ["postprocess/calc_gw_spect.pyx"])

#setup(ext_modules=[ext, ext1, ext3],
#    cmdclass = {'build_ext': build_ext})

setup(ext_modules=[ext1],
    cmdclass = {'build_ext': build_ext})
