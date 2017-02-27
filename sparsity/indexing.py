from pandas.core.indexing import _LocIndexer, _iLocIndexer

class _CsrLocationIndexer(_LocIndexer):

    def __getitem__(self, item):
        if not isinstance(item, slice):
            raise NotImplementedError()
        return super().__getitem__(item)

    def _slice(self, slice, axis=0, kind=None):
        if axis != 0:
            raise NotImplementedError()
        self.obj._slice(slice)

class _CsrILocationIndexer(_iLocIndexer):

    def __getitem__(self, item):
        if not isinstance(item, slice) or not isinstance(item, int):
            raise NotImplementedError()
        return super().__getitem__(item)

    def _slice(self, slice, axis=0, kind=None):
        if axis != 0:
            raise NotImplementedError()
        self.obj._slice(slice)