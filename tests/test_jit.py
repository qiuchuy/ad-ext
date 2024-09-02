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
            return b

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = al.from_numpy(a)
        iree_result = g(b)
        np_result = np.transpose(a)
        assert TestJIT.numeric_check(iree_result, np_result)
