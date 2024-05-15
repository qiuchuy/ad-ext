import ailang as al
import numpy as np

class TestJVP:
    def test_reshape(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[2, 3], [1, 4]])
        primal = al.from_numpy(a)
        tangent = al.from_numpy(b)
        value, grad = al.jvp(lambda x: al.reshape(x, (1, 4)), (primal, ), (tangent, ))
        assert np.allclose(value.tolist(), [[1, 2, 3, 4]])
        assert np.allclose(grad.tolist(), [[2, 3, 1, 4]])

    def test_transpose(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[2, 3], [1, 4]])
        primal = al.from_numpy(a)
        tangent = al.from_numpy(b)
        value, grad = al.jvp(lambda x: al.transpose(x), (primal, ), (tangent, ))
        assert np.allclose(value.tolist(), [[1, 3], [2, 4]])
        assert np.allclose(grad.tolist(), [[2, 1], [3, 4]])





  
