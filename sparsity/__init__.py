import os

from sparsity.sparse_frame import SparseFrame, sparse_one_hot
root = os.path.dirname(__file__)
__version__ = open(os.path.join(root, 'VERSION')).read().strip()
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
