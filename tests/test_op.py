import ailang as al
import typing
import numpy as np
import torch

torch.set_printoptions(precision=6)


class TestOp:
    @staticmethod
    def numeric_check(a: al.array, b: np.ndarray):
        return np.allclose(
            a.tolist(), b.tolist(), rtol=1e-03, atol=1e-06, equal_nan=True
        )

    def gen_random_nparray(
        self, shape: typing.Tuple[int], dtype: np.dtype
    ) -> np.ndarray:
        if len(shape):
            random_nparray = np.random.randn(*shape).astype(dtype)
            return random_nparray
        else:
            return dtype(np.random.randn())

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

    def test_standard_sqrt(self):
        a = self.gen_random_nparray((3, 2), np.float32)
        a = np.abs(a)  # you must have this with numeric_check's equal_nan = False
        b = al.from_numpy(a)
        assert TestOp.numeric_check(al.standard.sqrt(b), np.sqrt(a))

    def test_standard_mean(self):
        a = self.gen_random_nparray((3, 2, 3), np.float32)
        b = al.from_numpy(a)
        c = al.standard.mean(b, [0, 1, 2])
        assert TestOp.numeric_check(c, np.mean(a, axis=(0, 1, 2)))

    def test_standard_sum(self):
        a = self.gen_random_nparray((3, 2), np.float32)
        b = al.from_numpy(a)
        c = al.standard.sum(b)
        assert TestOp.numeric_check(c, np.sum(a))


    def test_standard_transpose(self):
        a = self.gen_random_nparray((2, 3), np.float32)
        b = al.from_numpy(a)
        c = al.standard.transpose(b)
        assert c.shape == (3, 2)
        assert TestOp.numeric_check(c, a.T)

    def test_standard_add(self):
        a = self.gen_random_nparray((2, 3), np.float32)
        b = self.gen_random_nparray((2, 3), np.float32)
        c = al.from_numpy(a)
        d = al.from_numpy(b)
        e = al.standard.add(c, d)
        assert TestOp.numeric_check(e, a + b)

        x = self.gen_random_nparray((), np.float32)
        y = self.gen_random_nparray((2, 3), np.float32)
        m = al.from_numpy(x)
        n = al.from_numpy(y)
        k = al.standard.add(m, n)
        assert TestOp.numeric_check(k, x + y)

        i = 1.0
        j = self.gen_random_nparray((2, 3), np.float32)
        al_j = al.from_numpy(j)
        r = al.standard.add(i, al_j)
        assert TestOp.numeric_check(r, i + j)

    def test_standard_sub(self):
        a = self.gen_random_nparray((2, 3), np.float32)
        b = self.gen_random_nparray((2, 3), np.float32)
        c = al.from_numpy(a)
        d = al.from_numpy(b)
        e = al.standard.sub(c, d)
        assert TestOp.numeric_check(e, a - b)

        x = self.gen_random_nparray((), np.float32)
        y = self.gen_random_nparray((2, 3), np.float32)
        m = al.from_numpy(x)
        n = al.from_numpy(y)
        k = al.standard.sub(m, n)
        assert TestOp.numeric_check(k, x - y)

    def test_standard_relu(self):
        a = self.gen_random_nparray((2, 3), np.float32)
        b = al.from_numpy(a)
        c = al.standard.relu(b)
        assert TestOp.numeric_check(c, np.maximum(a, 0))

    def test_standard_conv2d(self):
        a = self.gen_random_nparray((1, 3, 224, 224), np.float32)  # N C H W
        b = self.gen_random_nparray((2, 3, 4, 4), np.float32)  #  O I H W
        c = al.from_numpy(a)
        d = al.from_numpy(b)
        e = al.standard.conv2d(c, d, (2, 2), (1, 1), (1, 1), (1, 1, 1, 1), (0, 0))
        torch_conv = torch.nn.Conv2d(
            3, 2, kernel_size=4, stride=2, padding=1, bias=False
        )
        torch_conv.weight.data = torch.from_numpy(b)
        torch_res = torch_conv(torch.from_numpy(a)).detach().numpy()
        assert TestOp.numeric_check(e, torch_res)

    def test_standard_var(self):
        a = self.gen_random_nparray((3, 3, 2), np.float32)
        b = al.from_numpy(a)
        c = al.standard.var(b, [0, 1], 1)
        assert TestOp.numeric_check(
            c, torch.var(torch.from_numpy(a), dim=(0, 1), unbiased=True).numpy()
        )

    def test_standard_batchnorm2d_training(self):
        a = self.gen_random_nparray((1, 3, 20, 20), np.float32)  # N C H W
        # a = np.ones((1, 3, 5, 5), dtype=np.float32)
        b = al.from_numpy(a)
        al_bn = al.nn.Batchnorm2d(3)
        tc_bn = torch.nn.BatchNorm2d(3, eps=1e-5, momentum=None, affine=False)
        tc_bn.train()
        al_bn.train()
        c = al_bn(b)
        t = tc_bn(torch.from_numpy(a))
        assert TestOp.numeric_check(c, t)

    def test_standard_batchnorm2d_eval(self):
        a = self.gen_random_nparray((1, 64, 224, 224), np.float32)  # N C H W
        # a = np.ones((1, 3, 5, 5), dtype=np.float32)
        b = al.from_numpy(a)
        al_bn = al.nn.Batchnorm2d(64)
        tc_bn = torch.nn.BatchNorm2d(64, momentum=None, affine=False)
        tc_bn.eval()
        al_bn.eval()
        c = al_bn(b)
        t = tc_bn(torch.from_numpy(a))
        assert TestOp.numeric_check(c, t)

    def test_standard_maxpool2d(self):
        a = self.gen_random_nparray((1, 4, 224, 224), np.float32)  # N C H W
        b = al.from_numpy(a)
        # also u can use nn
        c = al.standard.maxpool2d(
            b,
            (1, 1, 4, 4),  # kernel_size
            (1, 1, 3, 3),  # stride
            (1, 1, 1, 1),  # dilation
            (1, 1, 1, 1),  # dilation
            (0, 0, 0, 0, 0, 0, 0, 0),  # padding
        )
        torch_maxpool2d = torch.nn.MaxPool2d(4, 3, 0)
        torch_res = torch_maxpool2d(torch.from_numpy(a)).numpy()
        assert TestOp.numeric_check(c, torch_res)

    def test_standard_avgpool2d(self):
        a = self.gen_random_nparray((1, 3, 224, 224), np.float32)  # N C H W
        b = al.from_numpy(a)
        # also u can use nn
        c = al.standard.avgpool2d(
            b,
            (1, 1, 4, 4),  # kernel_size
            (1, 1, 3, 3),  # stride
            (1, 1, 1, 1),  # dilation
            (1, 1, 1, 1),  # dilation
            (0, 0, 0, 0, 0, 0, 0, 0),  # padding
        )
        torch_avgpool2d = torch.nn.AvgPool2d(4, 3, 0)
        torch_res = torch_avgpool2d(torch.from_numpy(a)).numpy()
        assert TestOp.numeric_check(c, torch_res)

    def test_standard_div(self):
        a = self.gen_random_nparray((2, 3), np.float32)
        b = self.gen_random_nparray((2, 3), np.float32)
        c = al.from_numpy(a)
        d = al.from_numpy(b)
        e = al.standard.div(c, d)
        assert TestOp.numeric_check(e, a / b)

        x = self.gen_random_nparray((), np.float32)
        y = self.gen_random_nparray((2, 3), np.float32)
        m = al.from_numpy(x)
        n = al.from_numpy(y)
        k = al.standard.div(m, n)
        assert TestOp.numeric_check(k, x / y)

    def test_standard_mul(self):
        a = self.gen_random_nparray((2, 3), np.float32)
        b = self.gen_random_nparray((2, 3), np.float32)
        c = al.from_numpy(a)
        d = al.from_numpy(b)
        e = al.standard.mul(c, d)
        assert TestOp.numeric_check(e, a * b)

        x = self.gen_random_nparray((), np.float32)
        y = self.gen_random_nparray((2, 3), np.float32)
        m = al.from_numpy(x)
        n = al.from_numpy(y)
        k = al.standard.mul(m, n)
        assert TestOp.numeric_check(k, x * y)

    def test_standard_neg(self):
        a = self.gen_random_nparray((2, 3), np.float32)
        b = al.from_numpy(a)
        c = al.standard.neg(b)
        assert TestOp.numeric_check(c, -a)

    def test_standard_cat(self):
        a = self.gen_random_nparray((2, 3), np.float32)
        b = self.gen_random_nparray((2, 3), np.float32)
        c = al.from_numpy(a)
        d = al.from_numpy(b)
        e = al.standard.cat([c, d], 1)
        assert TestOp.numeric_check(e, np.concatenate([a, b], axis=1))

    def test_standard_exp(self):
        a = self.gen_random_nparray((2, 3), np.float32)
        b = al.from_numpy(a)
        c = al.standard.exp(b)
        assert TestOp.numeric_check(c, np.exp(a))

    def test_standard_tanh(self):
        a = self.gen_random_nparray((2, 3), np.float32)
        b = al.from_numpy(a)
        c = al.standard.tanh(b)
        assert TestOp.numeric_check(c, np.tanh(a))

    def test_standard_broadcast_to(self):
        a = self.gen_random_nparray((2, 1), np.float32)
        b = al.from_numpy(a)
        c = al.standard.broadcast_to(b, (2, 3))
        assert TestOp.numeric_check(c, np.broadcast_to(a, (2, 3)))

        d = self.gen_random_nparray((), np.float32)
        e = al.from_numpy(d)
        f = al.standard.broadcast_to(e, (2, 3))
        assert TestOp.numeric_check(f, np.broadcast_to(d, (2, 3)))

    def test_standard_matmul(self):
        a = self.gen_random_nparray((2, 3), np.float32)
        b = self.gen_random_nparray((3, 4), np.float32)
        c = al.from_numpy(a)
        d = al.from_numpy(b)
        e = al.standard.matmul(c, d)
        assert TestOp.numeric_check(e, np.matmul(a, b))
