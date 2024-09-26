import functools
import numpy as np
import ailang as al

from ailang import array


class TestGrad:
    @staticmethod
    def numeric_check(a: al.array, b: np.ndarray):
        return np.allclose(a.tolist(), b.tolist())

    def test_return(self):
        @al.grad
        def g(x):
            return al.sum(x)

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = al.from_numpy(a)
        value, grad = g(b)
        assert TestGrad.numeric_check(value, np.sum(a))
        assert TestGrad.numeric_check(grad, np.ones_like(a))

    def test_exp(self):
        @al.grad
        def g(x):
            return al.sum(al.exp(x))

        a = np.array([[0, 0], [0, 0]], dtype=np.float32)
        b = al.from_numpy(a)
        value, grad = g(b)
        assert TestGrad.numeric_check(value, np.sum(np.exp(a)))
        assert TestGrad.numeric_check(grad, np.exp(a) * np.ones_like(a))

    def test_add(self):
        @al.grad
        def g(x, y):
            return al.sum(al.add(x, y))

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        c = al.from_numpy(a)
        d = al.from_numpy(b)
        value, gradx, grady = g(c, d)
        assert TestGrad.numeric_check(value, np.sum(a + b))
        assert TestGrad.numeric_check(gradx, np.ones_like(a))
        assert TestGrad.numeric_check(grady, np.ones_like(b))

    def test_mul(self):
        @al.grad
        def g(x, y):
            return al.sum(al.mul(x, y))

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        c = al.from_numpy(a)
        d = al.from_numpy(b)
        value, gradx, grady = g(c, d)
        assert TestGrad.numeric_check(value, np.sum(a * b))
        assert TestGrad.numeric_check(gradx, b)
        assert TestGrad.numeric_check(grady, a)

    def test_div(self):
        @al.grad
        def g(x, y):
            return al.sum(al.div(x, y))

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        c = al.from_numpy(a)
        d = al.from_numpy(b)
        value, gradx, grady = g(c, d)
        assert TestGrad.numeric_check(value, np.sum(a / b))
        assert TestGrad.numeric_check(gradx, 1 / b)
        assert TestGrad.numeric_check(grady, -a / b**2)

    def test_neg(self):
        @al.grad
        def g(x):
            return al.sum(al.neg(x))

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = al.from_numpy(a)
        value, grad = g(b)
        assert TestGrad.numeric_check(value, np.sum(-a))
        assert TestGrad.numeric_check(grad, -np.ones_like(a))

    def test_broadcast(self):
        @al.grad
        def g(x):
            y = al.broadcast_to(x, (2, 2))
            return al.sum(y)

        a = np.array(1.0).astype(np.float32)
        b = al.from_numpy(a)
        value, grad = g(b)
        assert TestGrad.numeric_check(value, np.sum(np.ones((2, 2))))
        assert TestGrad.numeric_check(grad, np.array(4.0))

    def test_transpose(self):
        @al.grad
        def g(x):
            # for
            return al.sum(al.transpose(x, [1, 0]))

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = al.from_numpy(a)
        value, grad = g(b)
        assert TestGrad.numeric_check(value, np.sum(a.T))
        assert TestGrad.numeric_check(grad, np.ones_like(a.T))

    def test_tanh(self):
        @al.grad
        def g(x):
            return al.sum(al.tanh(x))

        a = np.array([[0, 0], [0, 0]], dtype=np.float32)
        b = al.from_numpy(a)
        value, grad = g(b)
        assert TestGrad.numeric_check(value, np.sum(np.tanh(a)))
        assert TestGrad.numeric_check(grad, 1 - np.tanh(a) ** 2)

    def test_matmul(self):
        @al.grad
        def g(x, y):
            return al.sum(al.matmul(x, y))

        a = np.array([[1], [2]], dtype=np.float32)
        b = np.array([[3, 4]], dtype=np.float32)
        c = al.from_numpy(a)
        d = al.from_numpy(b)
        value, gradx, grady = g(c, d)
        assert TestGrad.numeric_check(value, np.sum(np.matmul(a, b)))
        assert TestGrad.numeric_check(gradx, np.ones((2, 2)) @ b.T)
        assert TestGrad.numeric_check(grady, a.T @ np.ones((2, 2)))
