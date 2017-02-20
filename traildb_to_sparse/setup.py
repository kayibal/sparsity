import numpy as np
from distutils.core import setup, Extension

from Cython.Build import cythonize

ext = Extension("traildb_sparse",
                        ['traildb_sparse.pyx',
                         'traildb_coo.c',
                         'hashtable.c',
                         'linklist.c'],
                include_dirs=['/usr/local/include/', np.get_include()],
                libraries=["traildb"])

setup(
    ext_modules = cythonize([ext]),
)