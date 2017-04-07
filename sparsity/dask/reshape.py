from sparsity import sparse_one_hot
from sparsity.dask import SparseFrame
import pandas as pd

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
    idx_meta_col = index_col[0] if isinstance(index_col, list) else index_col
    if idx_meta_col and idx_meta_col != 'index':
        idx_meta = ddf._meta.reset_index().set_index(idx_meta_col).index
    else:
        idx_meta = ddf._meta.index

    meta = pd.DataFrame([], columns=categories,
                        index=idx_meta)

    dsf = ddf.map_partitions(sparse_one_hot,
                             column=column,
                             categories=categories,
                             index_col=index_col,
                             meta=meta)

    return SparseFrame(dsf.dask, dsf._name, meta, dsf.divisions)