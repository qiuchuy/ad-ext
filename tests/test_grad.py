import functools
import numpy as np
import ailang as al

from ailang import array

class TestJVP:
    @staticmethod
    def numeric_check(a: al.array, b: np.ndarray):
        return np.allclose(a.tolist(), b.tolist())

    def test_return(self):
        @al.jvp
        def g(x):
            return x

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = al.from_numpy(a)
        value, grad = g(b)
        TestJVP.numeric_check(value, a)
        TestJVP.numeric_check(grad, np.ones_like(a))

    def test_multiple_input(self):
        @al.jvp
        def g(x, y):
            return x

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = al.from_numpy(a)
        value0, value1, grad0, grad1 = g(b, b)
        TestJVP.numeric_check(value0, a)
        TestJVP.numeric_check(value1, a)
        TestJVP.numeric_check(grad0, np.ones_like(a))
        TestJVP.numeric_check(grad1, np.ones_like(a))

    
    