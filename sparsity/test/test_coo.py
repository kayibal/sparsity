import numpy as np
import pytest
try:
    from sparsity._traildb import traildb_coo_repr_func
    from sparsity.io import traildb_to_coo
    trail_db = True
except (ImportError, OSError):
    trail_db = False

@pytest.mark.skipif(trail_db is False, reason="TrailDB is not installed")
def test_coo_func(testdb):
    r_idx = np.zeros(9, dtype=np.uint64)
    c_idx = np.zeros(9, dtype=np.uint64)
    uuids = np.zeros((9,16), dtype=np.uint8)
    timestamps = np.zeros(9, dtype=np.uint64)
    res = traildb_coo_repr_func(testdb.encode(), b"username", r_idx, c_idx,
                              uuids,
                          timestamps)
    assert all(r_idx == np.arange(9))
    assert all(c_idx[:3] == 0)
    assert all(c_idx[3:6] == 1)
    assert all(c_idx[6:] == 2)

# def test_db_to_coo(testdb):
#     res = traildb_to_coo(testdb, "action")
#     pass
