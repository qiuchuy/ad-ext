import functools
import numpy as np
import ailang as al

from ailang import array


class TestJIT:
    @staticmethod
    def numeric_check(a: al.array, b: np.ndarray):
        return np.allclose(a.tolist(), b.tolist())

    def test_transpose(self):
        @al.jit
        def g(x):
            b = al.transpose(x)
            c = al.transpose(b)
            if c.shape[0]:
                d = al.transpose(c)
            else:
                d = al.transpose(b)
            return d

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = al.from_numpy(a)
        iree_result = g(b)
        np_result = np.transpose(np.transpose(a)).T
        assert TestJIT.numeric_check(iree_result, np_result)
