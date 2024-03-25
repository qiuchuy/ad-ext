import ailang as al
import numpy as np

class TestArray:
    def test_from_numpy(self):
        a = np.random.randn(10, 10)
        b = al.from_numpy(a)
        assert a.shape == b.shape
        assert a.strides == b.strides
        assert a.tolist() == b.tolist()

