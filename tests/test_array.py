import ailang as al
import numpy as np

class TestArray:
    def test_from_numpy(self):
        a = np.random.randn(10, 10)
        b = al.from_numpy(a)
        assert a.shape == b.shape
        assert a.strides == b.strides
        assert a.tolist() == b.tolist()

        c = np.random.randint(1, 10, (3, 4, 5), dtype=np.int16)
        d = al.from_numpy(c) 
        assert c.shape == d.shape
        assert c.strides == d.strides
        assert c.tolist() == d.tolist()

    def test_indexing(self): 
        a = np.random.randn(3, 2)
        b = al.from_numpy(a)
        c = b[0:1]
        assert c.shape == (1, 2)
        assert c.strides == (16, 8)
        assert al.flatten(c).tolist() == a[0].flatten().tolist()

        d = b[0:1, 0:1]
        assert d.shape == (1, 1)
        assert d.strides == (8, 8)
        assert al.flatten(d).tolist() == a[0][0].flatten().tolist()


    def test_iterator(self):
        a = np.random.randn(2, 2, 2)
        b = al.from_numpy(a)
        i = 0
        for c in b:
            assert c.tolist() == a[i].tolist()
            i = i + 1
        
        j = 0
        for d in b[0]:
            assert al.flatten(d).tolist() == a[0][j].tolist()
            j = j + 1