import ailang as al
import numpy as np

i = 0
a = np.random.randn(2, 2)
b = al.from_numpy(a)

def cond(i, b):
    return i < 0

def body(i, b):
    #i += 1
    b = al.transpose(b)
    return i, b

iter, result = al.while_loop(cond, body, (i, b))
assert np.allclose(result.tolist(), a.tolist())

class TestControlFlow:
    def test_range_for(self):
        a = np.random.randn(2, 2)
        d = np.random.randn(2, 2)
        b = al.from_numpy(a)
        c = al.from_numpy(d)
        for _ in range(10):
            b = al.transpose(b)
        assert np.allclose(b.tolist(), a.tolist())

        for _ in range(11):
            c = al.transpose(c)
        assert np.allclose(c.tolist(), d.T.tolist())

    def test_non_dependent_while(self):
        a = np.random.randn(2, 2)
        b = al.from_numpy(a)
        i = 0
        while i < 10:
            b = al.transpose(b)
            i += 1
        assert np.allclose(b.tolist(), a.tolist())

    def test_non_dependent_if(self):
        a = np.random.randn(2, 2)
        b = al.from_numpy(a)
        if True:
            b = al.transpose(b)
        assert np.allclose(b.tolist(), a.T.tolist())

    def test_dependent_if(self):
        a = np.random.randint(1, 10) 
        cond = al.from_numpy(np.array(a, dtype=np.int32)) > 5
        b = np.random.randn(2, 2)
        c = al.from_numpy(b) 
        if (cond):
            c = al.reshape(c, (1, 4))
        else:
            c = al.transpose(c)

        if (cond):
            assert c.shape == (1, 4)
        else:
            assert c.shape == (2, 2)

    """
    def test_builtin_while(self): 
        # loop variables: i, b
            # - python scalar: i
            # - ai tracer
        i = 0
        a = np.random.randn(2, 2)
        b = al.from_numpy(a)

        # cond
        def cond(i, b):
            return i < 10

        # body
        def body(i, b):
            # i += 1
            b = al.transpose(b)
            return i, b

        iter, result = al.while_loop(cond, body, (i, b))
        assert np.allclose(result.tolist(), a.tolist())
    """

