# coding=utf-8
from functools import partial

import pandas as pd
import numpy as np
import datetime as dt
from functools import reduce

from scipy import sparse

from sparsity import traildb_to_coo


class SparseFrame(object):
    """
    Simple sparse table based on scipy.sparse.csr_matrix
    """

    __slots__ = ["index", "columns", "data", "shape"]

    def __init__(self, data, index=None, columns=None, **kwargs):
        if len(data.shape) != 2:
            raise ValueError("Only two dimensional data supported")

        if index is None:
            self.index = np.arange(data.shape[0])
        else:
            assert len(index) == data.shape[0]
            self.index = index

        if columns is None:
            self.columns = np.arange(data.shape[1])
        else:
            assert len(columns) == data.shape[1]
            self.columns = columns

        if not sparse.isspmatrix_csr(data):
            self.data = sparse.csr_matrix(data, **kwargs)
        else:
            self.data = data

        self.shape = data.shape

    def groupby(self, by=None):
        """
        simple groupby operation using sparse matrix multiplication. Expects result to be sparse aswell
        :param by: (optional) alternative index
        :return:
        """
        if by is not None and by is not "index":
            assert len(by) == self.data.shape[0]
            by = np.array(by)
        else:
            by = self.index
        group_idx = by.argsort()
        gm = _create_group_matrix(by[group_idx])
        grouped_data = self.data[group_idx, :].T.dot(gm).T
        return SparseFrame(grouped_data, index=np.unique(by), columns=self.columns)

    def join(self, other, axis=0):
        """
        Can be used to stack two tables with identical inidizes
        :param other: another CSRTable or compatible datatype
        :param axis:
        :return:
        """
        if not isinstance(other, SparseFrame):
            other = SparseFrame(other)
        if axis not in set([0, 1]):
            raise ValueError("axis mut be either 0 or 1")
        if axis == 0:
            if np.all(other.columns == self.columns):
                data = sparse.vstack([self.data, other.data])
                index = np.hstack([self.index, other.index])
                res = SparseFrame(data, index=index, columns=self.columns)
            else:
                data, new_index = _matrix_join(self.data.T.tocsr(), other.data.T.tocsr(),
                                               np.asarray(self.columns), np.asarray(other.columns))
                res = SparseFrame(data.T.to_csr(),
                                  index=np.concatenate([self.index,other.index]),
                                  columns=new_index)
        elif axis == 1:
            if np.all(self.index == other.index):
                data = sparse.hstack([self.data, other.data])
                columns = np.hstack([self.columns, other.columns])
                res = SparseFrame(data, index=self.index, columns=columns)
            else:
                data, new_index = _matrix_join(self.data, other.data,
                                           np.asarray(self.index), np.asarray(other.index))
                res = SparseFrame(data,
                                  index=new_index,
                                  columns=np.concatenate([self.columns,other.columns]))
        return res

    def sort_index(self):
        passive_sort_idx = np.argsort(self.index)
        data = self.data[passive_sort_idx]
        index = self.index[passive_sort_idx]
        return SparseFrame(data, index=index)

    def add(self, other):
        assert np.all(self.columns == other.columns)
        data, index = _aligned_csr_elop(self.data, other.data,
                                        np.asarray(self.index),
                                        np.asarray(other.index))
        res = SparseFrame(data, index=index, columns = self.columns)
        return res


    def __sizeof__(self):
        return super().__sizeof__() + self.index.nbytes + \
               self.columns.nbytes + self.data.data.nbytes + \
               self.data.indptr.nbytes + self.data.indices.nbytes

    def _align_axis(self):
        raise NotImplementedError()

    def __repr__(self, *args, **kwargs):
        return self.head(5).to_string()

    def head(self, n=5):
        n = min(n, len(self.index))
        return pd.DataFrame(self.data[:n].todense(), index=self.index[:n], columns=self.columns)

    @classmethod
    def concat(cls, tables, axis=0):
        tables = list(tables)
        if len(tables) <= 1:
            return tables[0]
        func = partial(SparseFrame.join, axis=axis)
        return reduce(func, tables)

    @classmethod
    def read_traildb(cls, file, field):
        coo = traildb_to_coo(file, field)
        return cls(coo.tocsr())



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
    res = sparse.vstack([added, a[~intersect_idx_a], b[~intersect_idx_b]])
    index = np.concatenate([a_idx[intersect_idx_a], a_idx[~intersect_idx_a], b_idx[~intersect_idx_b]])
    return res, index


def _matrix_join(a,b, a_idx, b_idx):
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
    if len(intersection) > 0:
        passive_idx_a = np.in1d(a_idx, intersection).nonzero()[0]
        passive_idx_b = np.in1d(b_idx, intersection).nonzero()[0]
        joined_data.append(sparse.hstack([a[passive_idx_a], b[passive_idx_b]]))
        joined_idx.append(a_idx[passive_idx_a])

    if len(only_a) > 0:
        passive_idx_a = np.in1d(a_idx, only_a).nonzero()[0]
        only_left = a[passive_idx_a]
        only_left._shape = (len(passive_idx_a), a.shape[1] + b.shape[1])
        joined_data.append(only_left)
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



