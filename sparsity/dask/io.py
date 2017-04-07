from glob import glob
from math import ceil
from tokenize import tokenize

import numpy as np
import pandas as pd
from dask.dataframe.io.io import sorted_division_locations

import sparsity as sp
from sparsity.dask.core import SparseFrame, _make_meta


def from_pandas(df, npartitions=None, chunksize=None):
    nrows = df.shape[0]

    if chunksize is None:
        chunksize = int(ceil(nrows / npartitions))
    else:
        npartitions = int(ceil(nrows / chunksize))

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    divisions, locations = sorted_division_locations(df.index,
                                                     chunksize=chunksize)
    name = 'from_pandas-{}'.format(tokenize(df, npartitions))
    dsk = dict(((name, i), sp.SparseFrame(df.iloc[start: stop]))
               for i, (start, stop) in enumerate(zip(locations[:-1],
                                                     locations[1:])))
    meta = _make_meta(df)
    return SparseFrame(dsk, name, meta, divisions)


def read_npz(path, sorted=False):
    """
    Read SparseFrame from npz archives

    Parameters
    ----------
    path: str
        path to load files from can contain '*' to
        reference multiple files
    sorted: bool
        if the files are sorted read the index for each file
        to obtain divions

    Returns
    -------
        dsf: dask.SparseFrame
    """
    dsk = {}
    name = 'read_npz-{}'.format(tokenize(path))
    _paths = sorted(list(glob(path)))
    archive = np.load(_paths[0])

    meta_idx, meta_cols = archive['index'], archive['columns']
    meta = sp.SparseFrame([],
                          index=meta_idx[:0],
                          columns=meta_cols)
    for i, p in enumerate(_paths):
        dsk[name, i] = (sp.SparseFrame.read_npz, p)

    if sorted:
        level = 0 if isinstance(meta_idx, pd.MultiIndex) else None
        divisions = _npz_read_divisions(_paths, level=level)
    else:
        divisions = [None] * (len(_paths) + 1)

    return SparseFrame(dsk, name, meta, divisions=divisions)


def _npz_read_divisions(paths, level=None):
    divisions = []
    assert len(paths) > 1
    for p in paths:
        archive = np.load(p)
        idx = archive['index']
        if level is not None:
            idx = idx.get_level_values(level)
        istart = idx[0]
        istop = idx[-1]
        divisions.append(istart)
    divisions.append(istop)

    for i in range(len(divisions) - 1):
        if not divisions[i] < divisions[i+1]:
            raise ValueError("Divisions are not sorted"
                             "Problematic File:\n"
                             "{file}, !{div1} < {div2}".format(
                file=paths[i], div1=divisions[i], div2=divisions[i+1]
            ))

    return divisions