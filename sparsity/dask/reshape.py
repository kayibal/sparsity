import sparsity as sp
from sparsity import sparse_one_hot
from sparsity.dask import SparseFrame
import numpy as np

from sparsity.io import _just_read_array


def one_hot_encode(ddf, column,
                   categories, index_col):
    """
    Sparse one hot encoding of dask.DataFrame

    Convert a dask.DataFrame into a series of SparseFrames. By one hot
    encoding a single column

    Parameters
    ----------
    ddf: dask.DataFrame
        e.g. the clickstream
    column: str
        column name to one hot encode in with SparseFrame
    categories: iterable
        possible category values
    index_col: str, iterable
        which columns to use as index

    Returns
    -------
        sparse_one_hot: dask.Series
    """
    idx_meta = ddf._meta.reset_index().set_index(index_col).index[:0] \
        if index_col else ddf._meta.index

    if isinstance(categories, str):
        columns = _just_read_array(categories)
    else:
        columns = categories

    meta = sp.SparseFrame(np.array([]), columns=columns,
                        index=idx_meta)

    dsf = ddf.map_partitions(sparse_one_hot,
                             column=column,
                             categories=categories,
                             index_col=index_col,
                             meta=object)

    return SparseFrame(dsf.dask, dsf._name, meta, dsf.divisions)