//
//  main.c
//  traildb_to_sparse
//
//  Created by Alan Höng on 19/02/2017.
//  Copyright © 2017 Alan Höng. All rights reserved.
//

#include <stdio.h>
#include <traildb.h>
#include "traildb_coo.h"

int main(int argc, const char * argv[]) {
    tdb_error err;
    const char * db_path = argv[1];
    tdb* db = tdb_init();
    
    printf("%s\n", db_path);
    if ((err = tdb_open(db, db_path))){
        printf("Opening TrailDB failed: %s\n", tdb_error_str(err));
        exit(1);
    }
    
    uint64_t num_events = tdb_num_events(db);
    
    tdb_close(db);
    
    uint64_t *row_idx_array = malloc(sizeof(uint64_t) * num_events);
    uint64_t *col_idx_array = malloc(sizeof(uint64_t) * num_events);
    
    traildb_coo_repr(db_path, "username", row_idx_array, col_idx_array);
    
    int i;
    for (i=0; i < num_events; i++){
        printf("%d:%d\n", (int)(row_idx_array[i]), (int)(col_idx_array[i]));
    }
    return 0;
}
