import os

import numpy as np
import pytest

import sparsity
from sparsity import traildb_to_coo
from sparsity.traildb import traildb_coo_repr_func

@pytest.fixture()
def testdb():
    return os.path.join(sparsity.__path__[0], 'test/tiny.tdb')

def test_coo_func(testdb):
    r_idx = np.zeros(9, dtype=np.uint64)
    c_idx = np.zeros(9, dtype=np.uint64)
    traildb_coo_repr_func(testdb.encode(), b"username", r_idx, c_idx)
    assert all(r_idx == np.arange(9))
    assert all(c_idx[:3] == 0)
    assert all(c_idx[3:6] == 1)
    assert all(c_idx[6:] == 2)

def test_db_to_coo(testdb):
    coo_matrix = traildb_to_coo(testdb, "action")
    pass
