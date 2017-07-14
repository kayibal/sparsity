import os

from sparsity.sparse_frame import SparseFrame, sparse_one_hot
root = os.path.dirname(__file__)
__version__ = open(os.path.join(root, 'VERSION')).read().strip()