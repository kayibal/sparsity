# coding=utf-8
import datetime as dt
import os

from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
from moto import mock_s3
from scipy import sparse
from sparsity import SparseFrame, sparse_one_hot
from sparsity.io import _csr_to_dict

from .conftest import tmpdir

try:
    import traildb
except (ImportError, OSError):
    traildb = False


@contextmanager
def mock_s3_fs(bucket, data=None):
    """Mocks an s3 bucket

    Parameters
    ----------
    bucket: str
        bucket name
    data: dict
        dictionary with paths relative to bucket and
        bytestrings as values. Will mock data in bucket
        if supplied.

    Returns
    -------
    """
    try:
        m = mock_s3()
        m.start()
        import boto3
        import s3fs
        client = boto3.client('s3', region_name='eu-west-1')
        client.create_bucket(Bucket=bucket)
        if data is not None:
            data = data.copy()
            for key, value in data.items():
                client.put_object(Bucket=bucket, Key=key, Body=value)
        yield
    finally:
        if data is not None:
            for key in data.keys():
                client.delete_object(Bucket=bucket, Key=key)
        m.stop()


def test_empty_init():
    sf = SparseFrame(np.array([]), index=[], columns=['A', 'B'])
    assert sf.data.shape == (0, 2)


def test_groupby(groupby_frame):
    t = groupby_frame
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

    # test all
    assert np.all(sf.loc[list("ABCDE")].data.todense() == np.identity(5))
    assert np.all(sf.loc[:, :].data.todense() == np.identity(5))
    assert np.all(sf.loc[:].data.todense() == np.identity(5))

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


def test_loc_multi_index(sf_midx, sf_midx_int):

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

    assert np.all(sf_midx_int.loc[1].todense().values == sf_midx.data[:4,:])
    assert np.all(sf_midx_int.loc[0].todense().values == sf_midx.data[4, :])


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


def test_save_load_multiindex(sf_midx):
    with tmpdir() as tmp:
        # test new
        path = os.path.join(tmp, 'sf.npz')
        sf_midx.to_npz(path)
        res = SparseFrame.read_npz(path)
        assert isinstance(res.index, pd.MultiIndex)

        # test backwards compatibility
        def _to_npz_legacy(sf, filename):
            data = _csr_to_dict(sf.data)
            data['frame_index'] = sf.index.values
            data['frame_columns'] = sf.columns.values
            np.savez(filename, **data)

        _to_npz_legacy(sf_midx, path)
        res = SparseFrame.read_npz(path)
        assert isinstance(res.index, pd.MultiIndex)


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


def test_csr_one_hot_series_disk_categories(sampledata):
    with tmpdir() as tmp:
        categories = ['Sunday', 'Monday', 'Tuesday', 'Wednesday',
                      'Thursday', 'Friday', 'Saturday']
        cat_path = os.path.join(tmp, 'bla.pickle')
        pd.Series(categories).to_pickle(cat_path)
        sparse_frame = sparse_one_hot(sampledata(49),
                                      categories={'weekday': cat_path})
        res = sparse_frame.groupby_sum(np.tile(np.arange(7), 7)).data.todense()
        assert np.all(res == np.identity(7) * 7)


def test_csr_one_hot_series_legacy(sampledata):
    categories = ['Sunday', 'Monday', 'Tuesday', 'Wednesday',
                  'Thursday', 'Friday', 'Saturday']
    sparse_frame = sparse_one_hot(sampledata(49), 'weekday', categories)
    res = sparse_frame.groupby_sum(np.tile(np.arange(7), 7)).data.todense()
    assert np.all(res == np.identity(7) * 7)


def test_csr_one_hot_series(sampledata, weekdays, weekdays_abbr):
    correct = np.hstack((np.identity(7) * 7,
                         np.identity(7) * 7))

    categories = {'weekday': weekdays,
                  'weekday_abbr': weekdays_abbr}

    sparse_frame = sparse_one_hot(sampledata(49), categories=categories,
                                  order=['weekday', 'weekday_abbr'])

    res = sparse_frame.groupby_sum(np.tile(np.arange(7), 7)).data.todense()
    assert np.all(res == correct)
    assert all(sparse_frame.columns == (weekdays + weekdays_abbr))


def test_csr_one_hot_series_categorical_same_order(sampledata, weekdays,
                                                   weekdays_abbr):
    correct = np.hstack((np.identity(7) * 7,
                         np.identity(7) * 7))

    data = sampledata(49, categorical=True)

    categories = {'weekday': data['weekday'].cat.categories.tolist(),
                  'weekday_abbr': data['weekday_abbr'].cat.categories.tolist()}

    sparse_frame = sparse_one_hot(data,
                                  categories=categories,
                                  order=['weekday', 'weekday_abbr'],
                                  ignore_cat_order_mismatch=False)

    res = sparse_frame.groupby_sum(np.tile(np.arange(7), 7)) \
        .todense()[weekdays + weekdays_abbr].values
    assert np.all(res == correct)
    assert set(sparse_frame.columns) == set(weekdays + weekdays_abbr)


def test_csr_one_hot_series_categorical_different_order(sampledata, weekdays,
                                                        weekdays_abbr):
    correct = np.hstack((np.identity(7) * 7,
                         np.identity(7) * 7))

    data = sampledata(49, categorical=True)

    categories = {
        'weekday': data['weekday'].cat.categories.tolist()[::-1],
        'weekday_abbr': data['weekday_abbr'].cat.categories.tolist()[::-1]
    }

    with pytest.raises(ValueError):
        sparse_frame = sparse_one_hot(data,
                                      categories=categories,
                                      order=['weekday', 'weekday_abbr'],
                                      ignore_cat_order_mismatch=False)


def test_csr_one_hot_series_categorical_different_order_ignore(
        sampledata, weekdays, weekdays_abbr):

    correct = np.hstack((np.identity(7) * 7,
                         np.identity(7) * 7))

    data = sampledata(49, categorical=True)

    categories = {
        'weekday': data['weekday'].cat.categories.tolist()[::-1],
        'weekday_abbr': data['weekday_abbr'].cat.categories.tolist()[::-1]
    }

    sparse_frame = sparse_one_hot(data,
                                  categories=categories,
                                  order=['weekday', 'weekday_abbr'],
                                  ignore_cat_order_mismatch=True)

    res = sparse_frame.groupby_sum(np.tile(np.arange(7), 7)) \
        .todense()[weekdays + weekdays_abbr].values
    assert np.all(res == correct)
    assert set(sparse_frame.columns) == set(weekdays + weekdays_abbr)


def test_csr_one_hot_series_categorical_no_categories(
        sampledata, weekdays, weekdays_abbr):

    correct = np.hstack((np.identity(7) * 7,
                         np.identity(7) * 7))

    data = sampledata(49, categorical=True)

    categories = {
        'weekday': None,
        'weekday_abbr': None
    }

    sparse_frame = sparse_one_hot(data,
                                  categories=categories,
                                  order=['weekday', 'weekday_abbr'],
                                  ignore_cat_order_mismatch=True)

    res = sparse_frame.groupby_sum(np.tile(np.arange(7), 7)) \
        .todense()[weekdays + weekdays_abbr].values
    assert np.all(res == correct)
    assert set(sparse_frame.columns) == set(weekdays + weekdays_abbr)


def test_csr_one_hot_series_other_order(sampledata, weekdays, weekdays_abbr):

    categories = {'weekday': weekdays,
                  'weekday_abbr': weekdays_abbr}

    sparse_frame = sparse_one_hot(sampledata(49), categories=categories,
                                  order=['weekday_abbr', 'weekday'])

    assert all(sparse_frame.columns == (weekdays_abbr + weekdays))


def test_csr_one_hot_series_no_order(sampledata, weekdays, weekdays_abbr):

    categories = {'weekday': weekdays,
                  'weekday_abbr': weekdays_abbr}

    sparse_frame = sparse_one_hot(sampledata(49), categories=categories)

    assert sorted(sparse_frame.columns) == sorted(weekdays_abbr + weekdays)


def test_csr_one_hot_series_prefixes(sampledata, weekdays, weekdays_abbr):
    correct = np.hstack((np.identity(7) * 7,
                         np.identity(7) * 7))

    categories = {'weekday': weekdays,
                  'weekday_abbr': weekdays_abbr}

    sparse_frame = sparse_one_hot(sampledata(49), categories=categories,
                                  order=['weekday', 'weekday_abbr'],
                                  prefixes=True)

    res = sparse_frame.groupby_sum(np.tile(np.arange(7), 7)).data.todense()
    assert np.all(res == correct)
    correct_columns = list(map(lambda x: 'weekday_' + x, weekdays)) \
        + list(map(lambda x: 'weekday_abbr_' + x, weekdays_abbr))
    assert all(sparse_frame.columns == correct_columns)


def test_csr_one_hot_series_same_categories(weekdays):
    sample_data = pd.DataFrame(
        dict(date=pd.date_range("2017-01-01", periods=7)))
    sample_data["weekday"] = sample_data.date.dt.weekday_name
    sample_data["weekday2"] = sample_data.date.dt.weekday_name

    categories = {'weekday': weekdays,
                  'weekday2': weekdays}

    with pytest.raises(ValueError):
        sparse_one_hot(sample_data, categories=categories,
                       order=['weekday', 'weekday2'])

    sparse_frame = sparse_one_hot(sample_data, categories=categories,
                                  order=['weekday', 'weekday2'],
                                  prefixes=True)

    correct_columns = list(map(lambda x: 'weekday_' + x, weekdays)) \
        + list(map(lambda x: 'weekday2_' + x, weekdays))
    assert all(sparse_frame.columns == correct_columns)


def test_csr_one_hot_series_too_much_categories(sampledata):
    categories = ['Sunday', 'Monday', 'Tuesday', 'Wednesday',
                  'Thursday', 'Friday', 'Yesterday', 'Saturday', 'Birthday']
    sparse_frame = sparse_one_hot(sampledata(49),
                                  categories={'weekday': categories})
    res = sparse_frame.groupby_sum(np.tile(np.arange(7), 7)).data.todense()

    correct = np.identity(7) * 7
    correct = np.hstack((correct[:,:6], np.zeros((7, 1)),
                         correct[:, 6:], np.zeros((7, 1))))

    assert np.all(res == correct)


def test_csr_one_hot_series_too_little_categories(sampledata):
    categories = ['Sunday', 'Monday', 'Tuesday', 'Wednesday',
                  'Thursday', 'Friday']
    with pytest.raises(ValueError):
        sparse_one_hot(sampledata(49), categories={'weekday': categories})


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


def test_npz_io_s3(complex_example):
    with mock_s3_fs('sparsity'):
        sf, second, third = complex_example
        sf.to_npz('s3://sparsity/sparse.npz')
        loaded = SparseFrame.read_npz('s3://sparsity/sparse.npz')
        assert np.all(loaded.data.todense() == sf.data.todense())
        assert np.all(loaded.index == sf.index)
        assert np.all(loaded.columns == sf.columns)


def test_getitem():
    id_ = np.identity(10)
    sf = SparseFrame(id_, columns=list('abcdefghij'))
    assert sf['a'].data.todense()[0] == 1
    assert sf['j'].data.todense()[9] == 1
    assert np.all(sf[['a', 'b']].data.todense() == np.asmatrix(id_[:, [0, 1]]))
    tmp = sf[['j', 'a']].data.todense()
    assert tmp[9, 0] == 1
    assert tmp[0, 1] == 1
    assert (sf[list('abcdefghij')].data.todense() == np.identity(10)).all()
    assert sf[[]].shape == (10, 0)
    assert len(sf[[]].columns) == 0
    assert isinstance(sf.columns, type(sf[[]].columns))
    with pytest.raises(ValueError):
        sf[None]


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
                          categories={'page_id': list('ABCDE')},
                          index_col=['index', 'id'])
    sf_1 = sparse_one_hot(df_1,
                          categories={'page_id': list('ABCDE')},
                          index_col=['index', 'id'])
    res = SparseFrame.vstack([sf_0, sf_1])
    assert isinstance(res.index, pd.MultiIndex)


def test_boolean_indexing():
    sf = SparseFrame(np.identity(5))
    res = sf.loc[sf.index > 2]
    assert isinstance(res, SparseFrame)
    assert res.shape == (2, 5)
    assert res.index.tolist() == [3, 4]


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


def test_groupby_agg(groupby_frame):
    res = groupby_frame.groupby_agg(
        level=0,
        agg_func=lambda x: x.sum(axis=0)
    ).data.todense()
    assert np.all(res == (np.identity(10) * 10))

    res = groupby_frame.groupby_agg(
        level=0,
        agg_func=lambda x: x.mean(axis=0)
    ).data.todense()
    assert np.all(res.round() == np.identity(10))


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


def test_multiply_rowwise():
    # Row wise multiplication with different types
    sf = SparseFrame(np.ones((5, 5)))
    other = np.arange(5)
    msg = "Row wise multiplication failed"

    # nd.array
    other = other.reshape(1, -1)
    res = sf.multiply(other, axis=0)
    assert np.all(res.sum(axis=0) == 5 * other), msg

    # SparseFrame
    _other = SparseFrame(other)
    res = sf.multiply(_other, axis=0)
    assert np.all(res.sum(axis=0) == 5 * other), msg

    # csr_matrix
    _other = _other.data
    res = sf.multiply(_other, axis=0)
    assert np.all(res.sum(axis=0) == 5 * other), msg


def test_multiply_colwise():
    # Column wise multiplication with different types
    sf = SparseFrame(np.ones((5, 5)))
    other = np.arange(5)
    msg = "Column wise multiplication failed"

    # nd.array
    other = other.reshape(-1, 1)
    res = sf.multiply(other, axis=1)
    assert np.all(res.sum(axis=1) == 5 * other), msg

    # SparseFrame
    _other = SparseFrame(other)
    res = sf.multiply(_other, axis=1)
    assert np.all(res.sum(axis=1) == 5 * other), msg

    # csr_matrix
    _other = _other.data
    _other.toarray()
    res = sf.multiply(_other, axis=1)
    assert np.all(res.sum(axis=1) == 5 * other), msg


def test_drop_single_label():
    old_names = list('ABCDE')
    sf = SparseFrame(np.identity(5), columns=old_names)
    sf = sf.drop('A', axis=1)

    correct = np.identity(5)[:, 1:]
    assert sf.columns.tolist() == list('BCDE')
    np.testing.assert_array_equal(sf.data.todense(), correct)


def test_drop_multiple_labels():
    old_names = list('ABCDE')
    sf = SparseFrame(np.identity(5), columns=old_names)
    sf = sf.drop(['A', 'C'], axis=1)

    correct = np.identity(5)[:, [1, 3, 4]]
    assert sf.columns.tolist() == list('BDE')
    np.testing.assert_array_equal(sf.data.todense(), correct)


def test_label_based_indexing_col(sample_frame_labels):
    key = ['A', 'B']
    results = [
        sample_frame_labels[key],
        sample_frame_labels.loc[:, key],
        sample_frame_labels.reindex(columns=key)
    ]
    for res in results:
        np.testing.assert_array_equal(
            res.data.todense(), np.identity(5)[:, :2])
        assert (res.index == pd.Index(list('VWXYZ'))).all()
        assert (res.columns == pd.Index(list('AB'))).all()


def test_label_based_indexing_idx(sample_frame_labels):
    key = ['X', 'Y', 'Z']
    results = [
        sample_frame_labels.loc[key],
        sample_frame_labels.loc[key, :],
        sample_frame_labels.reindex(labels=key, axis=0),
        sample_frame_labels.reindex(index=key)
    ]
    for res in results:
        np.testing.assert_array_equal(
            res.data.todense(), np.identity(5)[2:, :])
        assert (res.index == pd.Index(['X', 'Y', 'Z'])).all()
        assert (res.columns == pd.Index(list('ABCDE'))).all()


def test_label_based_col_and_idx(sample_frame_labels):
    key = ['V', 'W'], ['A', 'B']
    results = [
        sample_frame_labels.loc[key],
        sample_frame_labels.loc[['V', 'W'], ['A', 'B']],
        sample_frame_labels.reindex(index=key[0], columns=key[1])
    ]
    for res in results:
        np.testing.assert_array_equal(
            res.data.todense(), np.identity(2))
        assert (res.index == pd.Index(list('VW'))).all()
        assert (res.columns == pd.Index(list('AB'))).all()


def test_indexing_boolean_label_col_and_idx(sample_frame_labels):
    res = sample_frame_labels.loc[[True, True, False, False, False], ['A', 'B']]
    np.testing.assert_array_equal(
        res.data.todense(), np.identity(2))
    assert (res.index == pd.Index(list('VW'))).all()
    assert (res.columns == pd.Index(list('AB'))).all()

    res = sample_frame_labels.loc[['V', 'W'], [True, True, False, False, False]]
    np.testing.assert_array_equal(
        res.data.todense(), np.identity(2))
    assert (res.index == pd.Index(list('VW'))).all()
    assert (res.columns == pd.Index(list('AB'))).all()


def test_error_reindex_duplicate_axis():
    sf = SparseFrame(np.identity(5),
                     columns = list('ABCDE'),
                     index = list('UUXYZ'))
    with pytest.raises(ValueError):
        sf.reindex(['U', 'V'])


def test_empty_elemwise():
    sf_empty = SparseFrame(np.array([]), columns=['A', 'B'])
    sf = SparseFrame(np.identity(2), columns=['A', 'B'])

    res = sf_empty.add(sf).data.todense()
    assert np.all(res == sf.data.todense())

    res = sf.add(sf_empty).data.todense()
    assert np.all(res == sf.data.todense())

    with pytest.raises(ValueError):
        res = sf.add(sf_empty, fill_value=None)


def test_loc_duplicate_index():
    sf = SparseFrame(np.identity(5),
                     columns=list('UUXYZ'),
                     index=list('AAABB'))
    assert len(sf.loc['A'].index) == 3
    assert len(sf.loc['B'].index) == 2
    assert np.all(sf.loc['A'].todense().values == np.identity(5)[:3])
    assert np.all(sf.loc['B'].todense().values == np.identity(5)[3:])

    assert len(sf.loc[:, 'U'].columns) == 2
    assert np.all(sf.loc[:, 'U'].todense().values == np.identity(5)[:, :2])
