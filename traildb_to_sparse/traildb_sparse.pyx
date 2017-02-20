from traildb import TrailDB
import numpy
from scipy import sparse
cimport numpy as np

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example, but good practice)
np.import_array()

cdef extern from "<traildb.h>":
    pass
cdef extern from "stdint.h":
    ctypedef unsigned long long uint64_t
# cdefine the signature of our c function
cdef extern from "traildb_coo.h":
    void traildb_coo_repr (const char * path, const char * fieldname,
                           uint64_t * row_idx, uint64_t * col_idx)

# create the wrapper code, with numpy type annotations
def traildb_coo_repr_func(char * path, char * fieldname,
                          np.ndarray[np.uint64_t, ndim=1, mode="c"] row_idx not None,
                          np.ndarray[np.uint64_t, ndim=1, mode="c"] col_idx not None):
    traildb_coo_repr(path, fieldname,
                <uint64_t*> np.PyArray_DATA(row_idx),
                <uint64_t*> np.PyArray_DATA(col_idx))

def traildb_to_coo(db, fieldname):
    db_handle = TrailDB(db)
    num_events = db_handle.num_events
    del db_handle
    r_idx = numpy.zeros(num_events, dtype=numpy.uint64)
    c_idx = numpy.zeros(num_events, dtype=numpy.uint64)

    traildb_coo_repr_func(db, fieldname, r_idx, c_idx)
    return sparse.coo_matrix((numpy.ones(num_events), (r_idx, c_idx)))
