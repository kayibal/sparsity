from io import BytesIO
from pathlib import PurePath, Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy import sparse

_filesystems = {}

try:
    from dask.bytes.local import LocalFileSystem
except ImportError:

    class LocalFileSystem:
        open = open

_filesystems[''] = LocalFileSystem
_filesystems['file'] = LocalFileSystem

try:
    import s3fs
    _filesystems['s3'] = s3fs.S3FileSystem
except ImportError:
    pass

try:
    import gcsfs
    _filesystems['gs'] = gcsfs.GCSFileSystem
    _filesystems['gcs'] = gcsfs.GCSFileSystem
except ImportError:
    pass


def to_npz(sf, filename, block_size=None, storage_options=None):
    """Write to npz file format.

    Parameters
    ----------
    sf: sp.SparseFrame
        sparse frame to store.
    filename: str
        path to write to.
    block_size: int
         block size in bytes when sending data to external filesystem.
         Default is 100MB.
    storage_options: dict
        (optional) storage options for external filesystems.

    Returns
    -------
    sf: SparseFrame
    """
    filename = path2str(filename)
    data = _csr_to_dict(sf.data)
    data['metadata'] = \
        {'multiindex': True if isinstance(sf.index, pd.MultiIndex) else False}
    data['frame_index'] = sf.index.values
    data['frame_columns'] = sf.columns.values
    if not filename.endswith('.npz'):
        filename += '.npz'

    _write_dict_npz(data, filename, block_size, storage_options)


def _write_dict_npz(data, filename, block_size, storage_options):
    filename = path2str(filename)
    protocol = urlparse(filename).scheme or 'file'
    if protocol == 'file':
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'wb') as fp:
            np.savez(fp, **data)
    else:
        if block_size is None:
            block_size = 2 ** 20 * 100  # 100 MB
        buffer = BytesIO()
        np.savez(buffer, **data)
        buffer.seek(0)
        _save_remote(buffer, filename, block_size, storage_options)


def _save_remote(buffer, filename, block_size=None, storage_options=None):
    if storage_options is None:
        storage_options = {}
    filename = path2str(filename)
    protocol = urlparse(filename).scheme
    fs = _filesystems[protocol](**storage_options)
    with fs.open(filename, 'wb', block_size) as remote_f:
        while True:
            data = buffer.read(block_size)
            if len(data) == 0:
                break
            remote_f.write(data)


def read_npz(filename, storage_options=None):
    """Read from a npz file.

    Parameters
    ----------
    filename: str
        path to file.
    storage_options: dict
        (optional) storage options for external filesystems.

    Returns
    -------
    sf: sp.SparseFrame
    """
    loader = _open_npz_archive(filename, storage_options)
    try:
        csr_mat = _load_csr(loader)
        idx = _load_idx_from_npz(loader)
        cols = loader['frame_columns']
    finally:
        loader.close()
    return csr_mat, idx, cols


def _open_npz_archive(filename, storage_options=None):
    if storage_options is None:
        storage_options = {}
    filename = path2str(filename)
    protocol = urlparse(filename).scheme or 'file'
    open_f = _filesystems[protocol](**storage_options).open
    fp = open_f(filename, 'rb')
    loader = np.load(fp, allow_pickle=True)
    return loader


def _csr_to_dict(array):
    return dict(data = array.data ,indices=array.indices,
                indptr =array.indptr, shape=array.shape)


def _load_csr(loader):
    return sparse.csr_matrix((loader['data'],
                              loader['indices'],
                              loader['indptr']),
                             shape=loader['shape'])


def _load_idx_from_npz(loader):
    idx = loader['frame_index']
    try:
        if loader['metadata'][()]['multiindex']:
            idx = pd.MultiIndex.from_tuples(idx)
    except KeyError:
        if all(map(lambda x: isinstance(x, tuple), idx)):
            idx = pd.MultiIndex.from_tuples(idx)
    return idx


def _just_read_array(path):
    path = path2str(path)
    if path.endswith('hdf') or path.endswith('hdf5'):
        return pd.read_hdf(path, '/df').values
    elif path.endswith('csv'):
        return pd.read_csv(path).values
    elif path.endswith('pickle'):
        return pd.read_pickle(path).values


def path2str(arg):
    """Convert arg into its string representation.

    This is only done if arg is subclass of PurePath
    """
    if issubclass(type(arg), PurePath):
        return str(arg)
    return arg
