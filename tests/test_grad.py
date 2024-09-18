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
        value, gradx, grady = g(b, b)
        TestJVP.numeric_check(value, a)
        TestJVP.numeric_check(gradx, np.ones_like(a))
        TestJVP.numeric_check(grady, np.zeros_like(a))

    def test_exp(self):
        @al.jvp
        def g(x):
            return al.exp(x)

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = al.from_numpy(a)
        value, grad = g(b)
        TestJVP.numeric_check(value, np.exp(a))
        TestJVP.numeric_check(grad, np.exp(a) * np.ones_like(a))

    def test_add(self):
        @al.jvp
        def g(x, y):
            return al.add(x, y)

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        c = al.from_numpy(a)
        d = al.from_numpy(b)
        value, gradx, grady = g(c, d)
        TestJVP.numeric_check(value, a + b)
        TestJVP.numeric_check(gradx, np.ones_like(a))
        TestJVP.numeric_check(grady, np.ones_like(b))

    def test_neg(self):
        @al.jvp
        def g(x):
            return al.neg(x)
        
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = al.from_numpy(a)
        value, grad = g(b)
        TestJVP.numeric_check(value, -a)
        TestJVP.numeric_check(grad, -np.ones_like(a))

    def test_broadcast(self):
        @al.jvp
        def g(x):
            y = al.broadcast_to(x, (2, 2))
            return y
        a = np.array(1.).astype(np.float32)
        b = al.from_numpy(a)
        value, grad = g(b)
        TestJVP.numeric_check(value, np.ones((2, 2)))
        TestJVP.numeric_check(grad, np.array(4.))

    
    