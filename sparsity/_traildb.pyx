from libc.stdlib cimport malloc, free
cimport numpy as np
from cpython.bytes cimport PyBytes_FromString
np.import_array()

cdef extern from "stdint.h":
    ctypedef unsigned long long uint64_t
    ctypedef unsigned char uint8_t
# cdefine the signature of our c function
cdef extern from "src/traildb_coo.h":
    uint64_t traildb_coo_repr (const char * path, const char * fieldname,
                               uint64_t * row_idx, uint64_t * col_idx,
                               uint8_t * uids, uint64_t * timestamps,
                               char** col_names, uint64_t** str_lens)

# create the wrapper code, with numpy type annotations
def traildb_coo_repr_func(char * path, char * fieldname,
                          np.ndarray[np.uint64_t, ndim=1, mode="c"] row_idx not None,
                          np.ndarray[np.uint64_t, ndim=1, mode="c"] col_idx not None,
                          np.ndarray[np.uint8_t, ndim=2, mode="c"] uuids not None,
                          np.ndarray[np.uint64_t, ndim=1, mode="c"] timestamps not None):
    #cdef uint8_t** uuids = <uint8_t**>malloc(len(row_idx) * sizeof(uint8_t*))
    cdef uint8_t[:,:] cython_uuids_view = uuids
    cdef uint8_t *c_uuid_array = &cython_uuids_view[0, 0]
    cdef char* col_names;
    cdef uint64_t* lens;
    n_cols = traildb_coo_repr(path, fieldname,
                              <uint64_t*> np.PyArray_DATA(row_idx),
                              <uint64_t*> np.PyArray_DATA(col_idx),
                              c_uuid_array,
                              <uint64_t*> np.PyArray_DATA(timestamps),
                              <char**> &col_names,
                              <uint64_t**> &lens)

    cols_raw = PyBytes_FromString(col_names)
    cols = []
    start = 0
    end = 0
    for i in range(n_cols):
        end = start + lens[i]
        cols.append(cols_raw[start:end].decode())
        start = end
    free(col_names)
    free(lens)
    return cols