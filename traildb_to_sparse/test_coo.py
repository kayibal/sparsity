import numpy as np
from traildb_sparse import traildb_coo_repr_func, traildb_to_coo

def test_coo_func():
    r_idx = np.zeros(9, dtype=np.uint64)
    c_idx = np.zeros(9, dtype=np.uint64)
    traildb_coo_repr_func(b"/Users/kayibal/Code/traildb-python/examples/tiny.tdb", b"username", r_idx, c_idx)
    assert all(r_idx == np.arange(9))
    assert all(c_idx[:3] == 0)
    assert all(c_idx[3:6] == 1)
    assert all(c_idx[6:] == 2)

def test_db_to_coo():
    coo_matrix = traildb_to_coo(b"/Users/kayibal/Code/traildb-python/examples/tiny.tdb", b"username")