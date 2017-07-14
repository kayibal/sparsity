import os
from distutils.core import setup, Extension
from setuptools import find_packages

try:
    import traildb
    import numpy as np
    from Cython.Build import cythonize
    ext = Extension("sparsity._traildb",
                            ['sparsity/_traildb.pyx',
                             'sparsity/src/traildb_coo.c',
                             'sparsity/src/hashtable.c',
                             'sparsity/src/linklist.c'],
                    include_dirs=['/usr/local/include/', np.get_include()],
                    libraries=["traildb"])
    ext_modules = cythonize([ext])
except (ImportError, OSError):
    ext_modules = None

root = os.path.dirname(__file__)
VERSION = open(os.path.join(root, 'sparsity', 'VERSION'))\
    .read().strip()

packages = find_packages()
packages.remove('sparsity.test')

setup(
    name='sparsity',
    version=VERSION,
    ext_modules = ext_modules,
    author='Alan Hoeng',
    author_email='alan.f.hoeng@gmail.com',
    packages=packages,
    package_data={'sparsity': ['VERSION']},
    install_requires=[
                        'pandas>=0.19.0',
                        'scipy>=0.18.1',
                        'numpy>=1.12.0'
                    ],
    test_requires=[
        'moto'
    ],
    zip_safe=False
)