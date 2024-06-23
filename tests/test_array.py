import ailang as al
import numpy as np

class TestArray:
    def test_from_numpy_array(self):
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

        e = np.array(True)
        f = al.from_numpy(e)
        assert e.shape == f.shape
        assert e.strides == f.strides
        assert e.tolist() == f.tolist()

    def test_from_numpy_scalar(self):
        a = np.array(3)
        b = al.from_numpy(a)
        assert b.item() == a.item()

    def test_compare_scalar(self):
        num = np.random.randint(1, 10)
        a = np.array(num, dtype=np.int32)
        b = al.from_numpy(a)
        assert b == num
        assert not (b < num)
        assert b <= num
        assert not b > num
        assert b >= num
        assert not b != num

    def test_compare_tracer(self):
        a = np.array(1)
        b = np.array(2)
        c = al.from_numpy(a)
        d = al.from_numpy(b)
        assert c < d
        assert c <= d
        assert c != d
        assert d > c
        assert d >= c

        e = np.array(3)
        f = np.array(3)
        g = al.from_numpy(e)
        h = al.from_numpy(f)
        assert g == h

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
    
