import numpy as np
from distutils.core import setup, Extension

from Cython.Build import cythonize

ext = Extension("sparsity.traildb",
                        ['sparsity/traildb.pyx',
                         'sparsity/src/traildb_coo.c',
                         'sparsity/src/hashtable.c',
                         'sparsity/src/linklist.c'],
                include_dirs=['/usr/local/include/', np.get_include()],
                libraries=["traildb"])

setup(
    ext_modules = cythonize([ext]),
)