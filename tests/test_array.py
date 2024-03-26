import ailang as al
import numpy as np

class TestArray:
    def test_from_numpy(self):
        a = np.random.randn(10, 10)
        b = al.from_numpy(a)
        assert a.shape == b.shape
        assert a.strides == b.strides
        assert a.tolist() == b.tolist()

        c = np.random.randint(0, 10, (3, 4, 5), dtype=np.int16)
        d = al.from_numpy(c) 
        assert c.shape == d.shape
        assert c.strides == d.strides
        assert c.tolist() == d.tolist()
    
    def test_iterator(self):
        a = np.random.randn(10, 10)
        b = al.from_numpy(a)
        for c in b:
            print(c)