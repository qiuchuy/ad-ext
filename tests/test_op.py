import ailang as al
import typing
import numpy as np


class TestOp:
    @staticmethod
    def numeric_check(a: al.array, b: np.ndarray):
        return np.allclose(a.tolist(), b.tolist())

    def gen_random_nparray(
        self, shape: typing.Tuple[int], dtype: np.dtype
    ) -> np.ndarray:
        random_nparray = np.random.randn(*shape).astype(dtype)
        return random_nparray

    def test_flatten(self):
        a = self.gen_random_nparray((2, 2), np.float32)
        b = al.from_numpy(a)
        c = al.flatten(b)
        assert c.shape == (4,)
        assert c.strides == (4,)
        assert TestOp.numeric_check(c, a.flatten())

    def test_reshape(self):
        a = self.gen_random_nparray((3, 2), np.float32)
        b = al.from_numpy(a)
        c = al.reshape(b, (2, 3))
        assert c.shape == (2, 3)
        assert c.strides == (12, 4)
        assert TestOp.numeric_check(c, a.reshape((2, 3)))

        d = al.reshape(c, (3, 2))
        assert d.shape == (3, 2)
        assert d.strides == (8, 4)
        assert TestOp.numeric_check(d, a)

    def test_slice(self):
        a = self.gen_random_nparray((3, 2), np.float32)
        b = al.from_numpy(a)
        c = al.slice(b, [0, 0], [1, 2], [1, 1])

        assert c.shape == (1, 2)
        assert c.strides == (8, 4)
        assert TestOp.numeric_check(c, a[0:1])

    def test_mean(self):
        a = self.gen_random_nparray((3, 2), np.float32)
        b = al.from_numpy(a)
        c = al.standard.mean(b)
        print(c)
        assert TestOp.numeric_check(c, np.mean(a))

    def test_transpose(self):
        a = self.gen_random_nparray((2, 3), np.float32)
        b = al.from_numpy(a)
        c = al.standard.transpose(b)
        assert c.shape == (3, 2)
        assert TestOp.numeric_check(c, a.T)
