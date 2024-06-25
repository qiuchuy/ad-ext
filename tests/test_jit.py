import numpy as np
import ailang as al

class TestJIT:
    def test_unary_op(self):
        @al.jit(debug=False)
        def g(x, y):
            b = al.transpose(y)
            a = al.transpose(b)
            z = al.transpose(a)
            return al.matmul(x, z)

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[1, 2], [3, 4]], dtype=np.float32)
        c = al.from_numpy(a)
        d = al.from_numpy(b)
        iree_result = g(c, d)
        print("Result: ", iree_result)

    def test_compare(self):
        @al.jit(debug=False)
        def g(x, y):
            return x == y

        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[1, 2], [3, 4]], dtype=np.float32)
        c = al.from_numpy(a)
        d = al.from_numpy(b)
        iree_result = g(c, d)
        print("Result: ", iree_result)

    def test_if(self):
        @al.jit(debug=True)
        def g(x):
            y = al.transpose(x)
            cond = al.transpose(x)

            def false_branch():
                z = al.transpose(x)
                return al.matmul(z, y)
            
            result = al.ifop(lambda: al.matmul(x, y), false_branch, cond)
            return result

        a = np.random.randn(2, 2)
        b = a.astype(np.int32)
        c = al.from_numpy(b)
        iree_result = g(c)
        print("Result: ", iree_result)
    """

    def test_while_loop(self):
        @al.jit(debug=True)
        def g(x):
            y = al.transpose(x)
            def cond(x, y):
                return x == y

            def body(x, y):
                return x, y

            a, b = al.while_loop(cond, body, (x, y))
            return b

        a = np.random.randn(2, 2)
        b = a.astype(np.int32)
        c = al.from_numpy(b)
        iree_result = g(c)
        print("Result: ", iree_result)
    """