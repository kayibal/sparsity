//
//  traildb_coo.h
//  traildb_to_sparse
//
//  Created by Alan Höng on 19/02/2017.
//  Copyright © 2017 Alan Höng. All rights reserved.
//

#ifndef traildb_coo_h
#define traildb_coo_h
#include <stdio.h>
#include <traildb.h>
uint64_t traildb_coo_repr(const char* fname, const char* fieldname,
                      uint64_t* row_idx_array, uint64_t* col_idx_array,
                      uint8_t* uids, uint64_t* timestamps,
                      char** col_names, uint64_t** str_lens);
#endif /* traildb_coo_h */
