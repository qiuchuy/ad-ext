import numpy as np
import ailang as al
import pytest


def convert(x: np.array):
    return al.from_numpy(x)


class TestOP:
    @pytest.mark.new
    def test_new(self):
        a = np.array([[1, 2], [9, 6]], dtype=np.float32)
        b = al.from_numpy(a)
        d = al.standard.var(b)
        print(d)
    
    @pytest.mark.batchnorm2d
    def test_batchnorm2d(self):
        # // %operand: [
        # //            [[1.0, 2.0], [3.0, 4.0]],
        # //            [[3.0, 4.0], [1.0, 2.0]]
        # //           ]
        # // %scale: [1.0, 1.0]
        # // %offset: [1.0, 1.0]
        # // %mean: [2.0, 3.0]
        # // %variance: [1.0, 1.0]
        operand = convert(
            np.array(
                [[[1.0, 2.0], [3.0, 4.0]], [[3.0, 4.0], [1.0, 2.0]]], dtype=np.float32
            )
        )
        offset = convert(np.array([1.0, 1.0], dtype=np.float32))
        scale = convert(np.array([1.0, 1.0], dtype=np.float32))
        mean = convert(np.array([1.0, 2.0], dtype=np.float32))
        variance = convert(np.array([1.0, 1.0], dtype=np.float32))

        @al.jit
        def g(operand, scale, offset, mean, variance):
            return al.batchnorm2d(operand, scale, offset, mean, variance)

        res = g(operand, scale, offset, mean, variance)
        print("eval res", res)

    @pytest.mark.maxpool2d
    def test_unary_op_maxpool(self):
        @al.jit
        def g(x):
            return al.maxpool2d(x)

        x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        x = al.from_numpy(x)
        iree_result = g(x)
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

    @pytest.mark.add
    def test_unary_op_add(self):
        @al.jit
        def g(x, y):
            b = al.transpose(y)
            a = al.transpose(b)
            z = al.transpose(a)
            m = al.add(x, z)
            return m

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[1, 4], [3, 7]], dtype=np.float32)
        c = al.from_numpy(a)
        d = al.from_numpy(b)
        iree_result = g(c, d)
        print("Result: ", iree_result)

    @pytest.mark.conv
    def test_unary_op_conv(self):
        @al.jit
        def g(x, y):
            return al.conv2d(x, y, (2, 2), (0, 0), (1, 1))

        a = np.ones((1, 4, 4, 1), dtype=np.float32)
        b = np.ones((3, 3, 1, 1), dtype=np.float32)
        c = al.from_numpy(a)
        d = al.from_numpy(b)
        iree_result = g(c, d)
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
