import numpy as np
from distutils.core import setup, Extension

from Cython.Build import cythonize

ext = Extension("sparsity._traildb",
                        ['sparsity/_traildb.pyx',
                         'sparsity/src/traildb_coo.c',
                         'sparsity/src/hashtable.c',
                         'sparsity/src/linklist.c'],
                include_dirs=['/usr/local/include/', np.get_include()],
                install_requires=[
                    'pandas>=0.19.2',
                    'scipy>=0.18.1',
                    'numpy>=1.12.0'
                ],
                libraries=["traildb"])

setup(
    ext_modules = cythonize([ext]),
)