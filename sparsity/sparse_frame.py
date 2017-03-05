# coding=utf-8
from functools import partial

import pandas as pd
import numpy as np
import datetime as dt
import uuid
from functools import reduce

from pandas.core.common import _default_index
from pandas.indexes.base import _ensure_index
from scipy import sparse

from sparsity.io import traildb_to_coo
from sparsity.indexing import _CsrILocationIndexer, _CsrLocIndexer

class SparseFrame(object):
    """
    Simple sparse table based on scipy.sparse.csr_matrix
    """

    __slots__ = ["_index", "_columns", "_data", "shape", "_multi_index",
                 'ndim', 'iloc', 'loc']

    def __init__(self, data, index=None, columns=None, **kwargs):
        if len(data.shape) != 2:
            raise ValueError("Only two dimensional data supported")
        N,K = data.shape

        if index is None:
            self._index = _default_index(N)
        else:
            assert len(index) == N
            self._index = _ensure_index(index)


        if columns is None:
            self._columns = _default_index(K)
        else:
            assert len(columns) == K
            self._columns = _ensure_index(columns)

        if not sparse.isspmatrix_csr(data):
            self._data = sparse.csr_matrix(data, **kwargs)
        else:
            self._data = data

        self.shape = data.shape

        # register indexers
        self.ndim = 2
        self.iloc = _CsrILocationIndexer(self, 'iloc')
        self.loc = _CsrLocIndexer(self, 'loc')


    def _get_axis(self, axis):
        if axis == 0:
            return self._index
        if axis == 1:
            return self._columns

    @property
    def index(self):
        return self._index

    @property
    def columns(self):
        return self._columns

    def groupby(self, by=None, level=0):
        """
        simple groupby operation using sparse matrix multiplication. Expects result to be sparse aswell
        :param by: (optional) alternative index
        :return:
        """
        if by is not None and by is not "index":
            assert len(by) == self._data.shape[0]
            by = np.array(by)
        else:
            if level and isinstance(self._index, pd.MultiIndex):
                by = self._multi_index.get_level_values(level).values
            elif level:
                raise ValueError("Connot use level in a non MultiIndex Frame")
            else:
                by = self.index.values
        group_idx = by.argsort()
        gm = _create_group_matrix(by[group_idx])
        grouped_data = self._data[group_idx, :].T.dot(gm).T
        return SparseFrame(grouped_data, index=np.unique(by), columns=self._columns)

    def join(self, other, axis=0, level=None):
        """
        Can be used to stack two tables with identical inidizes
        :param other: another CSRTable or compatible datatype
        :param axis:
        :return:
        """
        if isinstance(self._index, pd.MultiIndex)\
            or isinstance(other._index, pd.MultiIndex):
            raise NotImplementedError()
        if not isinstance(other, SparseFrame):
            other = SparseFrame(other)
        if axis not in set([0, 1]):
            raise ValueError("axis mut be either 0 or 1")
        if axis == 0:
            if np.all(other._columns.values == self._columns.values):
                # take short path if join axes are identical
                data = sparse.vstack([self._data, other._data])
                index = np.hstack([self.index, other.index])
                res = SparseFrame(data, index=index, columns=self._columns)
            else:
                data, new_index = _matrix_join(self._data.T.tocsr(), other._data.T.tocsr(),
                                               np.asarray(self._columns), np.asarray(other._columns))
                res = SparseFrame(data.T.to_csr(),
                                  index=np.concatenate([self.index, other.index]),
                                  columns=new_index)
        elif axis == 1:
            if np.all(self.index.values == other.index.values):
                # take short path if join axes are identical
                data = sparse.hstack([self._data, other._data])
                columns = np.hstack([self._columns, other._columns])
                res = SparseFrame(data, index=self.index, columns=columns)
            else:
                data, new_index= _matrix_join(self._data, other._data,
                                   np.asarray(self.index), np.asarray(other.index))
                res = SparseFrame(data,
                                  index=new_index,
                                  columns=np.concatenate([self._columns, other._columns]))
        return res

    def sort_index(self):
        passive_sort_idx = np.argsort(self._index)
        data = self._data[passive_sort_idx]
        index = self._index[passive_sort_idx]
        return SparseFrame(data, index=index)

    def add(self, other):
        if isinstance(self._index, pd.MultiIndex)\
            or isinstance(other._index, pd.MultiIndex):
            raise NotImplementedError()
        assert np.all(self._columns == other.columns)
        data, new_idx = _aligned_csr_elop(self._data, other._data,
                                          _safe_index(self.index),
                                          _safe_index(other.index))
        # new_idx = self._index.join(other.index, how=how)
        res = SparseFrame(data, index=new_idx, columns = self._columns)
        return res


    def __sizeof__(self):
        return super().__sizeof__() + self.index.nbytes + \
               self._columns.nbytes + self._data.data.nbytes + \
               self._data.indptr.nbytes + self._data.indices.nbytes

    def _align_axis(self):
        raise NotImplementedError()

    def __repr__(self, *args, **kwargs):
        return self.head(5).to_string()

    def head(self, n=5):
        n = min(n, len(self._index))
        return pd.DataFrame(self._data[:n].todense(),
                            index=self._index[:n],
                            columns=self._columns)

    def _slice(self, sliceobj):
        return SparseFrame(self._data[sliceobj,:], index=self.index[sliceobj])

    @classmethod
    def concat(cls, tables, axis=0):
        func = partial(SparseFrame.join, axis=axis)
        return reduce(func, tables)

    def _ixs(self, key, axis=0):
        if axis != 0:
            raise NotImplementedError()
        new_idx = self.index[key]
        if not isinstance(new_idx, pd.Index):
            new_idx = [new_idx]
        return SparseFrame(self._data[key,:], index=new_idx)

    @classmethod
    def read_traildb(cls, file, field, ts_unit='s'):
        uuids, timestamps, coo = traildb_to_coo(file, field)
        uuids = np.asarray([uuid.UUID(bytes=x.tobytes()) for x in
                            uuids])
        index = pd.MultiIndex.from_arrays \
            ([pd.CategoricalIndex(uuids),pd.to_datetime(timestamps, unit=ts_unit,)],
             names=('uuid', 'timestamp'))
        return cls(coo.tocsr(), index=index)



def _aligned_csr_elop(a, b, a_idx, b_idx, op='_plus_'):
    if len(a_idx) < len(b_idx):
        # swap variables
        a, b = b, a
        a_idx, b_idx = b_idx, a_idx

    # align data
    sort_idx_a = np.argsort(a_idx)
    sort_idx_b = np.argsort(b_idx)
    a_idx = a_idx[sort_idx_a]
    b_idx = b_idx[sort_idx_b]
    a = a[sort_idx_a]
    b = b[sort_idx_b]
    intersection = np.intersect1d(a_idx, b_idx)

    intersect_idx_a = np.in1d(a_idx, intersection)
    intersect_idx_b = np.in1d(b_idx, intersection)

    # calc result & result index
    added = a[intersect_idx_a]._binopt(b[intersect_idx_b], op=op)
    res = sparse.vstack([a[~intersect_idx_a], added, b[~intersect_idx_b]])
    index = np.concatenate([a_idx[~intersect_idx_a],a_idx[intersect_idx_a],
                            b_idx[~intersect_idx_b]])
    return res, index


def _matrix_join(a,b, a_idx, b_idx, how='outer'):
    # align data
    sort_idx = np.argsort(a_idx)
    a = a[sort_idx]
    a_idx = a_idx[sort_idx]

    sort_idx = np.argsort(b_idx)
    b = b[sort_idx]
    b_idx = b_idx[sort_idx]

    intersection = np.intersect1d(a_idx, b_idx)
    only_a = np.setdiff1d(a_idx, intersection)
    only_b = np.setdiff1d(b_idx, intersection)

    # calc result & result index
    joined_data = []
    joined_idx = []

    if len(only_a) > 0:
        passive_idx_a = np.in1d(a_idx, only_a).nonzero()[0]
        only_left = a[passive_idx_a]
        only_left._shape = (len(passive_idx_a), a.shape[1] + b.shape[1])
        joined_data.append(only_left)
        joined_idx.append(a_idx[passive_idx_a])

    if len(intersection) > 0:
        passive_idx_a = np.in1d(a_idx, intersection).nonzero()[0]
        passive_idx_b = np.in1d(b_idx, intersection).nonzero()[0]
        joined_data.append(sparse.hstack([a[passive_idx_a], b[passive_idx_b]]))
        joined_idx.append(a_idx[passive_idx_a])

    if len(only_b) > 0:
        passive_idx_b = np.in1d(b_idx, only_b).nonzero()[0]
        only_right = b[passive_idx_b]
        filler = sparse.coo_matrix((len(passive_idx_b), a.shape[1]))
        only_right = sparse.hstack([filler, only_right])
        joined_data.append(only_right)
        joined_idx.append(b_idx[passive_idx_b])

    joined_data = sparse.vstack(joined_data)
    joined_idx = np.concatenate(joined_idx)
    return joined_data, joined_idx


def _create_group_matrix(group_idx, dtype='f8'):
    """create a matrix based on groupby index labels"""
    if not isinstance(group_idx, pd.Categorical):
        group_idx = pd.Categorical(group_idx, np.unique(group_idx))
    col_idx = group_idx.codes
    row_idx = np.arange(len(col_idx))
    data = np.ones(len(row_idx))
    return sparse.coo_matrix((data, (row_idx, col_idx)),
                             shape=(len(group_idx), len(group_idx.categories)),
                             dtype=dtype).tocsr()


def csr_one_hot_series(s, categories, dtype='f8'):
    """Transform a pandas.Series into a sparse matrix.
    Works by one-hot-encoding for the given categories
    """
    cat = pd.Categorical(s, np.asarray(categories))

    codes = cat.codes
    n_features = len(cat.categories)
    n_samples = codes.size
    mask = codes != -1
    if np.any(~mask):
        raise ValueError("unknown categorical features present %s "
                         "during transform." % np.unique(s[~mask]))
    row_indices = np.arange(n_samples, dtype=np.int32)
    col_indices = codes
    data = np.ones(row_indices.size)
    return sparse.coo_matrix((data, (row_indices, col_indices)),
                             shape=(n_samples, n_features),
                             dtype=dtype).tocsr()


def sparse_aggregate_cs(raw, slice_date, agg_bin, categories,
                        id_col="id", categorical_col="pageId", **kwargs):
    """aggregates clickstream data using sparse data structures"""
    start_date = slice_date - dt.timedelta(days=agg_bin[1])
    end_date = slice_date - dt.timedelta(days=agg_bin[0])

    sliced_cs = raw.loc[start_date:end_date]
    sparse_bagged= sliced_cs.map_partitions(_sparse_groupby_sum_cs,
                                            group_col=id_col,
                                            categorical_col=categorical_col,
                                            categories=categories, meta=SparseFrame).compute(**kwargs)
    data = SparseFrame.concat(sparse_bagged, axis=0)
    data = data.groupby()
    return data


def _sparse_groupby_sum_cs(cs, group_col, categorical_col, categories):
    """transform a dask partition into a bagged sparse matrix"""
    if isinstance(categories, str):
        categories = pd.read_hdf(categories, "/df")
    one_hot = csr_one_hot_series(cs[categorical_col], categories)
    table = SparseFrame(one_hot, columns=categories, index=cs[group_col])
    return table.groupby()

def _safe_index(index):
    if isinstance(index, pd.MultiIndex):
        return index.map(hash)
    else:
        return index.values



