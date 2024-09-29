import functools
import ailang as al
import numpy as np
import torch
import typing

from ailang import array

torch.set_printoptions(precision=6)

class TestGrad:
    @staticmethod
    def numeric_check(a: al.array, b: np.ndarray):
        return np.allclose(a.tolist(), b.tolist(), rtol=1e-03, atol=1e-06)

    @staticmethod
    def gen_random_nparray(
        shape: typing.Tuple[int], dtype: np.dtype
    ) -> np.ndarray:
        if len(shape):
            random_nparray = np.random.randn(*shape).astype(dtype)
            return random_nparray
        else:
            return dtype(np.random.randn())

    def test_return(self):
        @al.grad
        def g(x):
            return al.sum(x)

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = al.from_numpy(a)
        value, grad = g(b)
        assert TestGrad.numeric_check(grad, np.ones_like(a))

    def test_exp(self):
        @al.grad
        def g(x):
            return al.sum(al.exp(x))

        a = np.array([[0, 0], [0, 0]], dtype=np.float32)
        b = al.from_numpy(a)
        value, grad = g(b)
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
        assert TestGrad.numeric_check(gradx, 1 / b)
        assert TestGrad.numeric_check(grady, -a / b**2)

    def test_neg(self):
        @al.grad
        def g(x):
            return al.sum(al.neg(x))

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = al.from_numpy(a)
        value, grad = g(b)
        assert TestGrad.numeric_check(grad, -np.ones_like(a))

    def test_sqrt(self):
        @al.grad
        def g(x):
            return al.sum(al.sqrt(x))

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = al.from_numpy(a)
        value, grad = g(b)
        assert TestGrad.numeric_check(grad, 0.5 / np.sqrt(a))

    def test_broadcast(self):
        @al.grad
        def g(x):
            y = al.broadcast_to(x, (2, 2))
            return al.sum(y)

        a = np.array(1.0).astype(np.float32)
        b = al.from_numpy(a)
        value, grad = g(b)
        assert TestGrad.numeric_check(grad, np.array(4.))

    def test_transpose(self):
        @al.grad
        def g(x):
            # for
            return al.sum(al.transpose(x, [1, 0]))

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = al.from_numpy(a)
        value, grad = g(b)
        assert TestGrad.numeric_check(grad, np.ones_like(a.T))

    def test_tanh(self):
        @al.grad
        def g(x):
            return al.sum(al.tanh(x))

        a = np.array([[0, 0], [0, 0]], dtype=np.float32)
        b = al.from_numpy(a)
        value, grad = g(b)
        assert TestGrad.numeric_check(grad, 1 - np.tanh(a)**2)

    def test_matmul(self):
        @al.grad
        def g(x, y):
            return al.sum(al.matmul(x, y))

        a = np.array([[1], [2]], dtype=np.float32)
        b = np.array([[3, 4]], dtype=np.float32)
        c = al.from_numpy(a)
        d = al.from_numpy(b)
        value, gradx, grady = g(c, d)
        assert TestGrad.numeric_check(gradx, np.ones((2, 2)) @ b.T)
        assert TestGrad.numeric_check(grady, a.T @ np.ones((2, 2)))

    def test_relu(self):
        @al.grad
        def g(x):
            return al.sum(al.relu(x))
        a = np.array([[1, -2], [-3, 4]], dtype=np.float32)
        b = al.from_numpy(a)
        value, grad = g(b)
        assert TestGrad.numeric_check(grad, np.where(a > 0, 1, 0))

    def test_mean(self):
        @al.grad
        def g(x):
            return al.sum(al.mean(x))
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = al.from_numpy(a)
        value, grad = g(b)
        assert TestGrad.numeric_check(grad, np.ones_like(a) / 4)

    def test_var(self):
        @al.grad
        def g(x):
            return al.sum(al.var(x,[0, 1], 0))
        a = np.array([[1, 1], [3, 3]], dtype=np.float32)
        b = al.from_numpy(a)
        value, grad = g(b)
        assert TestGrad.numeric_check(grad, np.array([[-0.5, -0.5], [0.5, 0.5]]))

    def test_batchnorm2d(self):
        a = self.gen_random_nparray((1, 3, 20, 20), np.float32)  # N C H W
        # a = np.ones((1, 3, 5, 5), dtype=np.float32)
        b = al.from_numpy(a)
        al_bn = al.nn.Batchnorm2d(3)
        tc_bn = torch.nn.BatchNorm2d(3, eps=1e-5, momentum=None, affine=False)
        tc_bn.eval()
        al_bn.eval()

        @al.grad
        def g(x):
            return al.sum(al_bn(x))

        value, grad = g(b)
        torch_input = torch.from_numpy(a)
        torch_input.requires_grad = True
        t = tc_bn(torch_input).sum()
        t.backward()
        assert TestGrad.numeric_check(grad, torch_input.grad.detach().numpy())

    def test_max(self):
        @al.grad
        def g(x):
            return al.sum(al.max(x, [0, 1], True))

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = al.from_numpy(a)
        value, grad = g(b)
        assert TestGrad.numeric_check(grad, np.array([[0, 0], [0, 1]]))

    def test_sum(self):
        @al.grad
        def g(x):
            return al.sum(al.sum(x, [0, 1], True))

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = al.from_numpy(a)
        value, grad = g(b)
        assert TestGrad.numeric_check(grad, np.ones_like(a))

    def test_softmax(self):
        @al.grad
        def g(x):
            return al.sum(al.prim.softmax(x, dim=[1]))

        def torch_softmax(x):
            exp_x = torch.exp(x)
            sum_exp_x = torch.sum(exp_x, dim=1).reshape(-1, 1)
            return torch.div(exp_x, sum_exp_x)

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = al.from_numpy(a)
        value, grad = g(b)
        torch_input = torch.from_numpy(a)
        torch_input.requires_grad = True
        torch_output = torch.sum(torch_softmax(torch_input))
        torch_output.backward()
        assert TestGrad.numeric_check(grad, torch_input.grad.detach().numpy())

    # def test_conv2d(self):
    #     @al.grad
    #     def g(x, y):
    #         return al.sum(al.standard.conv2d(x, y, (2, 2), (1, 1), (1, 1), (1, 1, 1, 1), (0, 0)))


    #     a = self.gen_random_nparray((1, 3, 224, 224), np.float32)  # N C H W
    #     b = self.gen_random_nparray((2, 3, 4, 4), np.float32)  #  O I H W
    #     c = al.from_numpy(a)
    #     d = al.from_numpy(b)
    #     al_value, al_gradx, al_grady = g(c, d, (2, 2), (1, 1), (1, 1), (1, 1, 1, 1), (0, 0))
    #     torch_conv = torch.nn.Conv2d(
    #         3, 2, kernel_size=4, stride=2, padding=1, bias=False
    #     )
    #     torch_conv.weight.data = torch.from_numpy(b)
    #     torch_input = torch.from_numpy(a)
    #     torch_res = torch.sum(torch_conv(torch_input))
    #     torch_res.backward()
    #     torch_gradx = torch_input.grad.detach().numpy()
    #     torch_grady = torch_conv.weight.grad.detach().numpy()
    #     torch_value = torch_conv(torch.from_numpy(a)).detach().numpy()
    #     assert TestGrad.numeric_check(al_value, torch_value)
    #     assert TestGrad.numeric_check(al_gradx, torch_gradx)
    #     assert TestGrad.numeric_check(al_grady, torch_grady)


    
