import ailang as al
import numpy as np


class TestOp:
    def test_flatten(self):
        a = np.random.randn(2, 2)
        b = al.from_numpy(a)
        c = al.flatten(b)
        assert c.shape == (4,)
        assert c.strides == (8,)
