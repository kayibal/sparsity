cimport numpy as np
np.import_array()

cdef extern from "<traildb.h>":
    pass
cdef extern from "stdint.h":
    ctypedef unsigned long long uint64_t
# cdefine the signature of our c function
cdef extern from "src/traildb_coo.h":
    void traildb_coo_repr (const char * path, const char * fieldname,
                           uint64_t * row_idx, uint64_t * col_idx)

# create the wrapper code, with numpy type annotations
def traildb_coo_repr_func(char * path, char * fieldname,
                          np.ndarray[np.uint64_t, ndim=1, mode="c"] row_idx not None,
                          np.ndarray[np.uint64_t, ndim=1, mode="c"] col_idx not None,):
    traildb_coo_repr(path, fieldname,
                <uint64_t*> np.PyArray_DATA(row_idx),
                <uint64_t*> np.PyArray_DATA(col_idx))