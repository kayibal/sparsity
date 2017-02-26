from traildb import TrailDB
import numpy
from scipy import sparse

from sparsity.traildb import traildb_coo_repr_func

def traildb_to_coo(db, fieldname):
    db_handle = TrailDB(db)
    num_events = db_handle.num_events
    del db_handle
    r_idx = numpy.zeros(num_events, dtype=numpy.uint64)
    c_idx = numpy.zeros(num_events, dtype=numpy.uint64)

    traildb_coo_repr_func(db.encode(), fieldname.encode(), r_idx, c_idx)
    return sparse.coo_matrix((numpy.ones(num_events), (r_idx, c_idx)))

from sparsity.sparse_frame import (SparseFrame, csr_one_hot_series,)
                                   #sparse_aggregate_cs)