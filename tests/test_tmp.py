import numpy as np
import ailang as al
import pytest
import typing

def convert(x: np.array):
    return al.from_numpy(x)


class TestOP:
    @staticmethod
    def numeric_check(a: al.array, b: np.ndarray):
        return np.allclose(a.tolist(), b.tolist())

    def gen_random_nparray(
        self, shape: typing.Tuple[int], dtype: np.dtype
    ) -> np.ndarray:
        if len(shape):
            random_nparray = np.random.randn(*shape).astype(dtype)
            return random_nparray
        else:
            return dtype(np.random.randn())
        
    @pytest.mark.new
    def test_new(self):
        a = np.array([[1, 2], [9, 6]], dtype=np.float32)
        b = al.from_numpy(a)
        d = al.standard.var(b)
        print(d)
    @pytest.mark.broadcast
    def test_standard_broadcast_to(self):
        a = self.gen_random_nparray((2, 1), np.float32)
        b = al.from_numpy(a)
        c = al.standard.broadcast_to(b, (2, 3))
        print(c)
    @pytest.mark.batchnorm2d
    def test_standard_batchnorm2d(self):
        operand = al.from_numpy(
            np.array(
                [[[1.0, 2.0], [3.0, 4.0]], [[3.0, 4.0], [1.0, 2.0]]], dtype=np.float32
            )
        )
        offset = al.from_numpy(np.array([1.0, 1.0], dtype=np.float32))
        scale = al.from_numpy(np.array([1.0, 1.0], dtype=np.float32))
        mean = al.from_numpy(np.array([2.0, 3.0], dtype=np.float32))
        variance = al.from_numpy(np.array([1.0, 1.0], dtype=np.float32))
        res = al.standard.batchnorm2d(operand, scale, offset, mean, variance)

    @pytest.mark.maxpool2d
    def test_unary_op_maxpool(self):
        x = np.array([[[[ 0,  1,  2,  3,  4],
         [ 5,  6,  7,  8,  9],
         [10, 11, 12, 13, 14],
         [15, 16, 17, 18, 19],
         [20, 21, 22, 23, 24]],

        [[25, 26, 27, 28, 29],
         [30, 31, 32, 33, 34],
         [35, 36, 37, 38, 39],
         [40, 41, 42, 43, 44],
         [45, 46, 47, 48, 49]],

        [[50, 51, 52, 53, 54],
         [55, 56, 57, 58, 59],
         [60, 61, 62, 63, 64],
         [65, 66, 67, 68, 69],
         [70, 71, 72, 73, 74]]]], dtype=np.float32)
        y = al.from_numpy(x)
        
        iree_result = al.standard.maxpool2d(y,(1,1,3,3),(1,1,2,2),(1,1,1,1),(1,1,1,1),(0,0,0,0,0,0,0,0))
        print("Result: ", iree_result)

    @pytest.mark.mean
    def test_unary_op_mean(self):
        @al.jit
        def g(x):
            return al.mean(x)

        a = np.array([[1, 2], [-1, 0], [3, 4]], dtype=np.float32)
        b = al.from_numpy(a)
        iree_result = g(b)
        print("Result: ", iree_result)

    @pytest.mark.conv2d
    def test_unary_op_conv(self):
        a = np.random.randn(1, 224, 224, 1).astype(np.float32)
        b = np.random.randn(3, 3, 1, 1).astype(np.float32)
        c = al.from_numpy(a)
        d = al.from_numpy(b)
        iree_result = al.standard.conv2d(c, d,(4,4),(2,2),(1,1),(0,0,0,0),(0,0))
        print("Result: ", iree_result)

    @pytest.mark.relu
    def test_unary_op_relu(self):
        @al.jit
        def g(x):
            return al.relu(x)

        a = np.array([[1, 2], [-1, 0]], dtype=np.float32)
        b = al.from_numpy(a)
        iree_result = g(b)
        print("Result: ", iree_result)
