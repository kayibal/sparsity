import numpy
from scipy import sparse

from sparsity._traildb import traildb_coo_repr_func

try:
    from traildb import TrailDB
except ImportError:
    traildb = False

def traildb_to_coo(db, fieldname):
    db_handle = TrailDB(db)
    num_events = db_handle.num_events
    del db_handle
    r_idx = numpy.zeros(num_events, dtype=numpy.uint64)
    c_idx = numpy.zeros(num_events, dtype=numpy.uint64)
    uuids = numpy.zeros((num_events,16), dtype=numpy.uint8)
    timestamps = numpy.zeros(num_events, dtype=numpy.uint64)

    traildb_coo_repr_func(db.encode(), fieldname.encode(), r_idx,
                          c_idx, uuids, timestamps)
    return uuids, timestamps, \
        sparse.coo_matrix((numpy.ones(num_events), (r_idx, c_idx)))