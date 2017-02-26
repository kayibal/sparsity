//
//  traildb_coo.c
//  traildb_to_sparse
//
//  Created by Alan Höng on 19/02/2017.
//  Copyright © 2017 Alan Höng. All rights reserved.
//
#include "hashtable.h"
#include <traildb.h>
#include "traildb_coo.h"

void traildb_coo_repr(const char* fname, const char* fieldname,
                      uint64_t* row_idx_array, uint64_t* col_idx_array,
                      uint64_t* uids, uint64_t* timestamps){
    int summed = 0;
    tdb_error err;
    const char * db_path = fname;
    tdb* db = tdb_init();
    
    printf("%s\n", db_path);
    if ((err = tdb_open(db, db_path))){
        printf("Opening TrailDB failed: %s\n", tdb_error_str(err));
        exit(1);
    }
    
    tdb_field oh_field;
    if (( err = tdb_get_field(db, fieldname, &oh_field))){
        printf("Could not find field: %s\n", tdb_error_str(err));
        exit(1);
    }
    
    uint64_t n_columns = tdb_lexicon_size(db, oh_field);
    hashtable_t *col_mapping = ht_create(n_columns, n_columns, free);
    
    uint64_t max_col_idx = 0;
    
    tdb_cursor *cursor = tdb_cursor_new(db);
    
    uint64_t i;
    uint64_t j;
    uint64_t row_idx = 0;
    uint64_t cidx;
    
    /* loop over all trails aka users */
    for (i = 0; i < tdb_num_trails(db); i++){
        const tdb_event *event;
        tdb_get_trail(cursor, i);
        
        /* loop over all events */
        while ((event = tdb_cursor_next(cursor))){
            for (j = 0; j < event->num_items; j++){
                if (oh_field == tdb_item_field(event->items[j])){
                    uint64_t len;
                    const char *val = tdb_get_item_value(db, event->items[j], &len);
                    if (ht_exists(col_mapping, val, len)){
                        cidx = *((uint64_t*) (ht_get(col_mapping, val, len, NULL)));
                    } else {
                        uint64_t *tmp = malloc(sizeof max_col_idx);
                        *tmp = max_col_idx;
                        ht_set(col_mapping, val, len, tmp, 1);
                        cidx = max_col_idx;
                        max_col_idx += 1;
                    }
                    if (summed <=0){
                        row_idx_array[row_idx] = row_idx;
                        col_idx_array[row_idx] = cidx;
                        row_idx += 1;
                    }
                    uids[row_idx] = (uint64_t)(*tdb_get_uuid(db, i));
                    timestamps[row_idx] = event->timestamp;
                    break;
                }
            }
        }
    }
    // TODO: return colnames
    tdb_cursor_free(cursor);
    ht_destroy(col_mapping);
    tdb_close(db);
}
