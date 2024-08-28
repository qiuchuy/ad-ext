import ailang as al
import typing
import numpy as np

from ailang import array

class TestOp:
    @staticmethod
    def numeric_check(a: array, b: np.ndarray):
        return np.allclose(a.tolist(), b.tolist())

    def test_flatten(self):
        a = np.random.randn(2, 2)
        b = al.from_numpy(a)
        c = al.flatten(b)
        assert c.shape == (4,)
        assert c.strides == (8,)
        assert TestOp.numeric_check(c, a.flatten())

    def test_reshape(self):
        a = np.random.randn(3, 2)
        b = al.from_numpy(a)
        c = al.reshape(b, (2, 3))
        assert c.shape == (2, 3)
        assert c.strides == (24, 8)
        assert TestOp.numeric_check(c, a.reshape((2, 3)))

        d = al.reshape(c, (3, 2))
        assert d.shape == (3, 2)
        assert d.strides == (16, 8)
        assert TestOp.numeric_check(d, a)

    def test_slice(self):
        a = np.random.randn(3, 2)
        b = al.from_numpy(a)
        c = al.slice(b, [0, 0], [1, 2], [1, 1])

        assert c.shape == (1, 2)
        assert c.strides == (16, 8)
        assert TestOp.numeric_check(c, a[0:1])
