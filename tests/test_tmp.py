import numpy as np
import ailang as al


class TestOP:
    def test_unary_op_add(self):
        @al.jit(debug=False)
        def g(x, y):
            b = al.transpose(y)
            a = al.transpose(b)
            z = al.transpose(a)
            return al.add(x, z)

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[1, 2], [3, 4]], dtype=np.float32)
        c = al.from_numpy(a)
        d = al.from_numpy(b)
        iree_result = g(c, d)
        print("Result: ", iree_result)

    def test_unary_opp_conv(self):
        @al.jit(debug=False)
        def g(x, y):
            return al.conv2d(x, y, (2, 2), (0, 0), (1, 1))

        a = np.ones((1, 224, 224, 3), dtype=np.float32)
        b = np.ones((3, 3, 3, 5), dtype=np.float32)
        c = al.from_numpy(a)
        d = al.from_numpy(b)
        iree_result = g(c, d)
        print("Result: ", iree_result)
