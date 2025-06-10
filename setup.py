from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy


def _get_version():
    with open('evaltools/__init__.py', 'r') as init_f:
        content = init_f.readlines()
    target_line = [ln for ln in content if '__version__ = ' in ln]
    version = target_line[-1].split()[-1].strip("'")
    return version


# with open('requirements.txt') as f:
#     requires = f.read().splitlines()

extensions = [
    Extension(
        "evaltools.scores._fastimpl",
        ["evaltools/scores/_fastimpl.pyx"],
    ),
]

setup(
    name='evaltools',
    packages=find_packages(),
    # install_requires=requires,
    # scripts=[],  # command line tools
    version=_get_version(),
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
)
