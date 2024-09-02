import functools
import numpy as np
import ailang as al

from ailang import array


class TestJIT:
    @staticmethod
    def numeric_check(a: np.ndarray, b: np.ndarray):
        return np.allclose(a, b)

    def test_return(self):
        @al.grad
        def g(x):
            return x

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = al.from_numpy(a)
        iree_result = g(b)
        assert TestJIT.numeric_check(iree_result, np.ones_like(iree_result))
