# coding=utf-8
import os
import datetime as dt
import pandas as pd

import dask.dataframe as dd
import numpy as np
import pytest
from dask.async import get_sync
from scipy import sparse

from sparsity import SparseFrame, sparse_one_hot

try:
    import traildb
except (ImportError, OSError):
    traildb = False


# 2017 starts with a sunday
@pytest.fixture()
def sampledata():
    def gendata(n):
        sample_data = pd.DataFrame(
            dict(date=pd.date_range("2017-01-01", periods=n)))
        sample_data["weekday"] = sample_data.date.dt.weekday_name
        sample_data["id"] = np.tile(np.arange(7), len(sample_data) // 7 + 1)[
                            :len(sample_data)]
        return sample_data

    return gendata


@pytest.fixture()
def sf_midx():
    midx = pd.MultiIndex.from_arrays(
        [pd.date_range("2016-10-01", periods=5),
         np.arange(5)]
    )
    cols = list('ABCDE')
    sf = SparseFrame(np.identity(5), index=midx, columns=cols)
    return sf


def test_empty_init():
    sf = SparseFrame(np.array([]), index=[], columns=['A', 'B'])
    assert sf.data.shape == (0, 2)


def test_groupby():
    shuffle_idx = np.random.permutation(np.arange(100))
    index = np.tile(np.arange(10), 10)
    data = np.vstack([np.identity(10) for _ in range(10)])
    t = SparseFrame(data[shuffle_idx, :], index=index[shuffle_idx])
    res = t.groupby_sum().data.todense()
    assert np.all(res == (np.identity(10) * 10))


def test_groupby_dense_random_data():
    shuffle_idx = np.random.permutation(np.arange(100))
    index = np.tile(np.arange(10), 10)
    single_tile = np.random.rand(10, 10)
    data = np.vstack([single_tile for _ in range(10)])
    t = SparseFrame(data[shuffle_idx, :], index=index[shuffle_idx])
    res = t.groupby_sum().data.todense()
    np.testing.assert_array_almost_equal(res, (single_tile * 10))


def test_simple_join():
    t = SparseFrame(np.identity(10))

    res1 = t.join(t, axis=0).data.todense()
    correct = np.vstack([np.identity(10), np.identity(10)])
    assert np.all(res1 == correct)

    res2 = t.join(t, axis=1).data.todense()
    correct = np.hstack([np.identity(10), np.identity(10)])
    assert np.all(res2 == correct)


def test_complex_join(complex_example):
    first, second, third = complex_example
    correct = pd.DataFrame(first.data.todense(),
                           index=first.index,
                           columns=map(str, range(len(first.columns)))) \
        .join(pd.DataFrame(second.data.todense(),
                           index=second.index,
                           columns=map(str, range(len(second.columns)))),
              how='left',
              rsuffix='_second') \
        .join(pd.DataFrame(third.data.todense(),
                           index=third.index,
                           columns=map(str, range(len(third.columns)))),
              how='left',
              rsuffix='_third') \
        .sort_index().fillna(0)

    res = first.join(second, axis=1).join(third, axis=1) \
        .sort_index().data.todense()
    assert np.all(correct.values == res)

    # res = right.join(left, axis=1).data.todense()
    # assert np.all(correct == res)


def test_mutually_exclusive_join():
    correct = np.vstack([np.hstack([np.identity(5), np.zeros((5, 5))]),
                         np.hstack([np.zeros((5, 5)), np.identity(5)])])

    left_ax1 = SparseFrame(np.identity(5), index=np.arange(5))
    right_ax1 = SparseFrame(np.identity(5), index=np.arange(5, 10))

    res_ax1 = left_ax1.join(right_ax1, axis=1)

    left_ax0 = SparseFrame(np.identity(5), columns=np.arange(5))
    right_ax0 = SparseFrame(np.identity(5), columns=np.arange(5, 10))

    with pytest.raises(NotImplementedError):  # FIXME: remove when repaired
        res_ax0 = left_ax0.join(right_ax0, axis=0)
        assert np.all(res_ax0.data.todense() == correct), \
            "Joining along axis 0 failed."

    assert np.all(res_ax1.data.todense() == correct), \
        "Joining along axis 1 failed."

def test__array___():
    correct = np.identity(5)
    sf = SparseFrame(correct, index=list('ABCDE'),
                     columns=list('ABCDE'))
    res = np.asarray(sf)
    assert np.all(res == correct)
    assert isinstance(res, np.ndarray)

    res = np.asarray(sf['A'])
    assert len(res.shape) == 1


def test_iloc():
    # name index and columns somehow so that their names are not integers
    sf = SparseFrame(np.identity(5), index=list('ABCDE'),
                     columns=list('ABCDE'))

    assert np.all(sf.iloc[:2].data.todense() == np.identity(5)[:2])
    assert np.all(sf.iloc[[3, 4]].data.todense() == np.identity(5)[[3, 4]])
    assert np.all(sf.iloc[3].data.todense() == np.identity(5)[3])
    assert sf.iloc[1:].shape == (4, 5)


def test_loc():
    sf = SparseFrame(np.identity(5), index=list("ABCDE"))

    # test single
    assert np.all(sf.loc['A'].data.todense() == np.matrix([[1, 0, 0, 0, 0]]))

    # test slices
    assert np.all(sf.loc[:'B'].data.todense() == np.identity(5)[:2])


    sf = SparseFrame(np.identity(5), pd.date_range("2016-10-01", periods=5))

    str_slice = slice('2016-10-01',"2016-10-03")
    assert np.all(sf.loc[str_slice].data.todense() ==
                  np.identity(5)[:3])

    ts_slice = slice(pd.Timestamp('2016-10-01'),pd.Timestamp("2016-10-03"))
    assert np.all(sf.loc[ts_slice].data.todense() ==
                  np.identity(5)[:3])

    dt_slice = slice(dt.date(2016,10,1), dt.date(2016,10,3))
    assert np.all(sf.loc[dt_slice].data.todense() ==
                  np.identity(5)[:3])


def test_loc_multi_index(sf_midx):

    assert sf_midx.loc['2016-10-01'].data[0, 0] == 1

    str_slice = slice('2016-10-01', "2016-10-03")
    assert np.all(sf_midx.loc[str_slice].data.todense() ==
                  np.identity(5)[:3])

    ts_slice = slice(pd.Timestamp('2016-10-01'), pd.Timestamp("2016-10-03"))
    assert np.all(sf_midx.loc[ts_slice].data.todense() ==
                  np.identity(5)[:3])

    dt_slice = slice(dt.date(2016, 10, 1), dt.date(2016, 10, 3))
    assert np.all(sf_midx.loc[dt_slice].data.todense() ==
                  np.identity(5)[:3])


def test_set_index(sf_midx):
    sf = sf_midx.set_index(level=1)
    assert np.all(sf.index.values == np.arange(5))

    sf = sf_midx.set_index(column='A')
    assert np.all(sf.index.values[1:] == 0)
    assert sf.index.values[0] == 1

    sf = sf_midx.set_index(idx=np.arange(5))
    assert np.all(sf.index.values == np.arange(5))

    # what if indices are actually ints, but don't start from 0?
    sf = SparseFrame(np.identity(5), index=[1, 2, 3, 4, 5])

    # test single
    assert np.all(sf.loc[1].data.todense() == np.matrix([[1, 0, 0, 0, 0]]))

    # test slices
    assert np.all(sf.loc[:2].data.todense() == np.identity(5)[:2])

    # assert np.all(sf.loc[[4, 5]].data.todense() == np.identity(5)[[3, 4]])


def test_new_column_assign_array():
    sf = SparseFrame(np.identity(5))
    sf[6] = np.ones(5)
    correct = np.hstack([np.identity(5), np.ones(5).reshape(-1, 1)])
    assert sf.shape == (5, 6)
    assert np.all(correct == sf.data.todense())


def test_new_column_assign_number():
    sf = SparseFrame(np.identity(5))
    sf[6] = 1
    correct = np.hstack([np.identity(5), np.ones(5).reshape(-1, 1)])
    assert sf.shape == (5, 6)
    assert np.all(correct == sf.data.todense())

def test_assign_array():
    sf = SparseFrame(np.identity(5), columns=list('ABCDE'))
    sf = sf.assign(**{'F': np.ones(5)})
    correct = np.hstack([np.identity(5), np.ones(5).reshape(-1, 1)])
    assert 'F' in set(sf.columns)
    assert sf.shape == (5, 6)
    assert np.all(correct == sf.data.todense())


def test_assign_number():
    sf = SparseFrame(np.identity(5), columns=list('ABCDE'))
    sf = sf.assign(**{'F': 1})
    correct = np.hstack([np.identity(5), np.ones(5).reshape(-1, 1)])
    assert 'F' in set(sf.columns)
    assert sf.shape == (5, 6)
    assert np.all(correct == sf.data.todense())


def test_existing_column_assign_array():
    sf = SparseFrame(np.identity(5))
    with pytest.raises(NotImplementedError):
        sf[0] = np.ones(5)
        correct = np.identity(5)
        correct[:, 0] = 1
        assert np.all(correct == sf.data.todense())


def test_existing_column_assign_number():
    sf = SparseFrame(np.identity(5))
    with pytest.raises(NotImplementedError):
        sf[0] = 1
        correct = np.identity(5)
        correct[:, 0] = 1
        assert np.all(correct == sf.data.todense())


@pytest.fixture()
def complex_example():
    first = np.identity(10)
    second = np.zeros((4, 10))
    third = np.zeros((4, 10))
    second[[0, 1, 2, 3], [2, 3, 4, 5]] = 10
    third[[0, 1, 2, 3], [6, 7, 8, 9]] = 20

    shuffle_idx = np.arange(10)
    np.random.shuffle(shuffle_idx)

    first = SparseFrame(first[shuffle_idx],
                        index=np.arange(10)[shuffle_idx])

    shuffle_idx = np.arange(4)
    np.random.shuffle(shuffle_idx)

    second = SparseFrame(second[shuffle_idx],
                         index=np.arange(2, 6)[shuffle_idx])

    shuffle_idx = np.arange(4)
    np.random.shuffle(shuffle_idx)

    third = SparseFrame(third[shuffle_idx],
                        index=np.arange(6, 10)[shuffle_idx])
    return first, second, third


def test_add_total_overlap(complex_example):
    first, second, third = complex_example
    correct = first.sort_index().data.todense()
    correct[2:6, :] += second.sort_index().data.todense()
    correct[6:, :] += third.sort_index().data.todense()

    res = first.add(second).add(third).sort_index()

    assert np.all(res.data.todense() == correct)


def test_simple_add_partial_overlap(complex_example):
    first = SparseFrame(np.ones((3, 5)), index=[0, 1, 2])
    second = SparseFrame(np.ones((3, 5)), index=[2, 3, 4])

    correct = np.ones((5,5))
    correct[2, :] += 1

    res = first.add(second)
    assert np.all(res.data.todense() == correct)
    assert np.all(res.index == range(5))


def test_add_partial_overlap(complex_example):
    first, second, third = complex_example
    third = third.sort_index()
    third._index = np.arange(8, 12)

    correct = first.sort_index().data.todense()
    correct[2:6, :] += second.sort_index().data.todense()
    correct[8:, :] += third.sort_index().data.todense()[:2, :]
    correct = np.vstack((correct, third.sort_index().data.todense()[2:, :]))

    res = first.add(second).add(third).sort_index()

    assert np.all(res.data.todense() == correct)


def test_add_no_overlap(complex_example):
    first, second, third = complex_example
    third = third.sort_index()
    third._index = np.arange(10, 14)

    correct = first.sort_index().data.todense()
    correct[2:6, :] += second.sort_index().data.todense()
    correct = np.vstack((correct, third.sort_index().data.todense()))

    res = first.add(second).add(third).sort_index()

    assert np.all(res.data.todense() == correct)


def test_csr_one_hot_series(sampledata):
    categories = ['Sunday', 'Monday', 'Tuesday', 'Wednesday',
                  'Thursday', 'Friday', 'Saturday']
    sparse_frame = sparse_one_hot(sampledata(49), 'weekday', categories)
    res = sparse_frame.groupby_sum(np.tile(np.arange(7), 7)).data.todense()
    assert np.all(res == np.identity(7) * 7)


def test_csr_one_hot_series_too_much_categories(sampledata):
    categories = ['Sunday', 'Monday', 'Tuesday', 'Wednesday',
                  'Thursday', 'Friday', 'Yesterday', 'Saturday', 'Birthday']
    sparse_frame = sparse_one_hot(sampledata(49), 'weekday', categories)
    res = sparse_frame.groupby_sum(np.tile(np.arange(7), 7)).data.todense()

    correct = np.identity(7) * 7
    correct = np.hstack((correct[:,:6], np.zeros((7, 1)),
                         correct[:, 6:], np.zeros((7, 1))))

    assert np.all(res == correct)


def test_csr_one_hot_series_too_little_categories(sampledata):
    categories = ['Sunday', 'Monday', 'Tuesday', 'Wednesday',
                  'Thursday', 'Friday']
    with pytest.raises(ValueError):
        sparse_one_hot(sampledata(49), 'weekday', categories)


@pytest.mark.skipif(traildb is False, reason="TrailDB not installed")
def test_read_traildb(testdb):
    res = SparseFrame.read_traildb(testdb, 'action')
    assert res.shape == (9, 3)


@pytest.mark.skipif(traildb is False, reason="TrailDB not installed")
def test_add_traildb(testdb):
    simple = SparseFrame.read_traildb(testdb, 'action')
    doubled = simple.add(simple)
    assert np.all(doubled.data.todense() == simple.data.todense() * 2)


def test_npz_io(complex_example):
    sf, second, third = complex_example
    sf.to_npz('/tmp/sparse.npz')
    loaded = SparseFrame.read_npz('/tmp/sparse.npz')
    assert np.all(loaded.data.todense() == sf.data.todense())
    assert np.all(loaded.index == sf.index)
    assert np.all(loaded.columns == sf.columns)
    os.remove('/tmp/sparse.npz')


def test_getitem():
    sf = SparseFrame(np.identity(10), columns=list('abcdefghij'))
    assert sf['a'].data.todense()[0] == 1
    assert sf['j'].data.todense()[9] == 1
    tmp = sf[['j', 'a']].data.todense()
    assert tmp[9, 0] == 1
    assert tmp[0, 1] == 1


def test_vstack():
    frames = []
    data = []
    for _ in range(10):
        values = np.identity(5)
        data.append(values)
        sf = SparseFrame(values,
                         columns=list('ABCDE'))
        frames.append(sf)
    sf = SparseFrame.vstack(frames)
    assert np.all(sf.data.todense() == np.vstack(data))

    with pytest.raises(AssertionError):
        frames[2] = SparseFrame(np.identity(5),
                                columns=list('XYZWQ'))
        SparseFrame.vstack(frames)


def test_vstack_multi_index(clickstream):
    df_0 = clickstream.iloc[:len(clickstream) // 2]
    df_1 = clickstream.iloc[len(clickstream) // 2:]
    sf_0 = sparse_one_hot(df_0,
                          categories=list('ABCDE'),
                          column='page_id',
                          index_col=['index', 'id'])
    sf_1 = sparse_one_hot(df_1,
                          categories=list('ABCDE'),
                          column='page_id',
                          index_col=['index', 'id'])
    res = SparseFrame.vstack([sf_0, sf_1])
    assert isinstance(res.index, pd.MultiIndex)


def test_boolean_indexing():
    sf = SparseFrame(np.identity(5))
    res = sf.loc[sf.index > 2]
    assert isinstance(res, SparseFrame)
    assert res.shape == (2, 5)
    assert res.index.tolist() == [3, 4]


def test_dask_loc(clickstream):
    sf = dd.from_pandas(clickstream, npartitions=10) \
        .map_partitions(
        sparse_one_hot,
        column='page_id',
        categories=list('ABCDE'),
        meta=list
    )

    res = sf.loc['2016-01-15':'2016-02-15']
    res = SparseFrame.concat(res.compute(get=get_sync).tolist())
    assert res.index.date.max() == dt.date(2016, 2, 15)
    assert res.index.date.min() == dt.date(2016, 1, 15)


def test_dask_multi_index_loc(clickstream):
    sf = dd.from_pandas(clickstream, npartitions=10) \
        .map_partitions(
            sparse_one_hot,
            column='page_id',
            index_col=['index', 'id'],
            categories=list('ABCDE'),
            meta=list
    )
    res = sf.loc['2016-01-15':'2016-02-15']
    res = SparseFrame.vstack(res.compute(get=get_sync).tolist())
    assert res.index.get_level_values(0).date.min() == dt.date(2016, 1, 15)
    assert res.index.get_level_values(0).date.max() == dt.date(2016, 2, 15)


def test_rename():
    old_names = list('ABCDE')
    func = lambda x: x + '_new'
    new_names = list(map(func, old_names))
    sf = SparseFrame(np.identity(5), columns=old_names)

    sf_renamed = sf.rename(columns=func)
    assert np.all(sf.columns == old_names), "Original frame was changed."
    assert np.all(sf_renamed.columns == new_names), "New frame has old names."

    sf.rename(columns=func, inplace=True)
    assert np.all(sf.columns == new_names), "In-place renaming didn't work."


def test_dropna():
    index = np.arange(5, dtype=float)
    index[[1, 3]] = np.nan
    sf = SparseFrame(np.identity(5), index=index)

    sf_cleared = sf.dropna()

    correct = np.zeros((3, 5))
    correct[[0, 1, 2], [0, 2, 4]] = 1

    assert np.all(sf_cleared.data.todense() == correct)


def test_drop_duplicate_idx():
    sf = SparseFrame(np.identity(5), index=np.arange(5))
    sf_dropped = sf.drop_duplicate_idx()
    assert np.all(sf_dropped.data.todense() == sf.data.todense())

    sf = SparseFrame(np.identity(8), index=[0, 0, 2, 3, 3, 5, 5, 5])
    sf_dropped = sf.drop_duplicate_idx()
    correct = np.identity(8)[[0, 2, 3, 5], :]
    assert np.all(sf_dropped.data.todense() == correct)


def test_repr():
    sf = SparseFrame(sparse.csr_matrix((10, 10000)))
    res = sf.__repr__()
    assert isinstance(res, str)
    assert '10x10000' in res
    assert '0 stored' in res

    sf = SparseFrame(np.array([]), index=[], columns=['A', 'B'])
    res = sf.__repr__()
    assert isinstance(res, str)


def test_init_with_pandas():
    df = pd.DataFrame(np.identity(5),
                      index=[
                          pd.date_range("2100-01-01", periods=5),
                          np.arange(5)
                      ],
                      columns=list('ABCDE'))
    sf = SparseFrame(df)
    assert sf.shape == (5, 5)
    assert isinstance(sf.index, pd.MultiIndex)
    assert sf.columns.tolist() == list('ABCDE')

    s = pd.Series(np.ones(10))
    sf = SparseFrame(s)

    assert sf.shape == (10, 1)
    assert np.all(sf.data.todense() == np.ones(10).reshape(-1, 1))

    df['A'] = 'bla'
    with pytest.raises(TypeError):
        sf = SparseFrame(df)
