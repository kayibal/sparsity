import os
import pytest

import sparsity


@pytest.fixture()
def testdb():
    return os.path.join(sparsity.__path__[0], 'test/tiny.tdb')