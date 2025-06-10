# Copyright (c) Météo France (2017-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info
"""Define how to compile _fastimpl.pyx."""
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

extensions = [
    Extension(
        "_fastimpl",
        ["_fastimpl.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extensions, language_level=3),
)
