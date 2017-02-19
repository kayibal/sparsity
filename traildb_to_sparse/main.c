//
//  main.c
//  traildb_to_sparse
//
//  Created by Alan Höng on 19/02/2017.
//  Copyright © 2017 Alan Höng. All rights reserved.
//

#include <stdio.h>
#include <traildb.h>
#include "hashtable.h"
#include "linklist.h"

int main(int argc, const char * argv[]) {
    linked_list_t *mat_col_idx = list_create();
    linked_list_t *mat_row_idx = list_create();
    
    tdb_error err;
    const char *fields[] = {"username", "action"};
    const char * db_path = argv[1];
    tdb* db = tdb_init();
    
    printf("%s\n", db_path);
    if ((err = tdb_open(db, db_path))){
        printf("Opening TrailDB failed: %s\n", tdb_error_str(err));
        exit(1);
    }
    
    tdb_field *oh_field;
    if (( err = tdb_get_field(db, "username", oh_field))){
        printf("Could not find field: %s\n", tdb_error_str(err));
        exit(1);
    }
    
    uint64_t n_columns = tdb_lexicon_size(db, *oh_field);
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
                if (*oh_field == tdb_item_field(event->items[j])){
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
                    list_push_value(mat_row_idx, row_idx);
                    list_push_value(mat_col_idx, cidx);
                    row_idx += 1;
                    break;
                }
            }
        }
    }
    
    while (list_count(mat_row_idx)){
        //printf("%d\n", (int)(l_idx));
        uint64_t loc= (uint64_t)(list_pop_value(mat_row_idx));
        printf("mat_row_idx: %d\n", (int)(loc));
    }
    
    while (list_count(mat_col_idx)){
        //printf("%d\n", (int)(l_idx));
        uint64_t loc= (uint64_t)(list_pop_value(mat_col_idx));
        printf("mat_col_idx: %d\n", (int)(loc));
    }
    
    
    return 0;
}

