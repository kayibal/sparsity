import dask
import pytest

import sparsity as sp
import sparsity.dask as dsp
import pandas as pd
import numpy as np
import dask.dataframe as dd

from sparsity.dask.core import one_hot_encode

dask.context.set_options(get=dask.async.get_sync)

def test_from_pandas():
    dsf = dsp.from_pandas(pd.DataFrame(np.random.rand(10,2)),
                          npartitions=3)
    res = dsf.compute()

    assert isinstance(res, sp.SparseFrame)
    assert res.shape == (10,2)

def test_map_partitions():
    data = pd.DataFrame(np.random.rand(10, 2))
    dsf = dsp.from_pandas(data,
                          npartitions=3)
    dsf = dsf.map_partitions(lambda x: x, dsf._meta)

    res = dsf.compute()

    assert isinstance(res, sp.SparseFrame)
    assert res.shape == (10, 2)


@pytest.mark.parametrize('iindexer, correct_shape', [
    (slice('A', 'B'), (2, 2)),
    (slice('C', None), (8, 2)),
    (slice(None, 'C'), (3, 2)),
])
def test_loc(iindexer, correct_shape):
    df = pd.DataFrame(np.random.rand(10, 2),
                      index=list('ABCDEFGHIJ'))
    dsf = dsp.from_pandas(df, npartitions=2)
    res = dsf.loc[iindexer].compute()

    assert isinstance(res, sp.SparseFrame)
    assert res.shape == correct_shape


def test_repr():
    dsf = dsp.from_pandas(pd.DataFrame(np.random.rand(10, 2)),
                          npartitions=3)
    assert isinstance(dsf.__repr__(), str)


def test_one_hot(clickstream):
    ddf = dd.from_pandas(clickstream, npartitions=10)
    dsf = one_hot_encode(ddf, column='page_id',
                         categories=list('ABCDE'),
                         index_col=['index', 'id'])
    sf = dsf.compute()
    assert sf.shape == (100, 5)
    assert isinstance(sf.index, pd.MultiIndex)