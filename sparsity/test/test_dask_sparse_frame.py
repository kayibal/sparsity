import shutil
import tempfile
import os
from contextlib import contextmanager

import dask
import pytest

import sparsity as sp
import sparsity.dask as dsp
import pandas as pd
import numpy as np
import dask.dataframe as dd

from sparsity.dask.reshape import one_hot_encode

dask.context.set_options(get=dask.async.get_sync)


@contextmanager
def tmpdir(dir=None):
    dirname = tempfile.mkdtemp(dir=dir)

    try:
        yield dirname
    finally:
        if os.path.exists(dirname):
            shutil.rmtree(dirname, ignore_errors=True)



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


def test_read_npz():
    sf = sp.SparseFrame(np.identity(100))
    with tmpdir() as tmp:
        sf.iloc[:25].to_npz(os.path.join(tmp, '1'))
        sf.iloc[25:50].to_npz(os.path.join(tmp, '2'))
        sf.iloc[50:75].to_npz(os.path.join(tmp, '3'))
        sf.iloc[75:].to_npz(os.path.join(tmp, '4'))

        dsf = dsp.read_npz(os.path.join(tmp, '*.npz'))
        sf = dsf.compute()
    assert np.all(sf.data.toarray() == np.identity(100))