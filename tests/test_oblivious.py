import unittest
import numpy as np
from sparse_pca.oblivious import matvec,bitonic


class NetworkTests(unittest.TestCase):
    def test_padding_and_rectangular_shapes(self):
        select=lambda bit,y,n:y if bit else n
        for seed in range(12):
            rng=np.random.default_rng(seed)
            for rows,cols in [(1,7),(7,1),(3,5),(5,3)]:
                x=rng.normal(size=(rows,cols)); x[rng.random(x.shape)<.7]=0
                entries=[(int(i),int(j),x[i,j]) for i,j in zip(*np.nonzero(x))]
                entries += [(0,0,0.)]*3
                v=rng.normal(size=cols)
                np.testing.assert_allclose(matvec(entries,v,rows,select),x@v,atol=1e-10)
                u=rng.normal(size=rows)
                np.testing.assert_allclose(matvec([(j,i,z) for i,j,z in entries],u,cols,select),x.T@u,atol=1e-10)

    def test_duplicate_coordinates(self):
        result=matvec([(0,0,2.),(0,0,3.),(1,0,-1.)],[4.],3,lambda b,y,n:y if b else n)
        np.testing.assert_allclose(result,[20,-4,0])

    def test_bitonic_non_power_of_two(self):
        data=[[5],[2],[7],[0],[3]]
        result=bitonic(data,[100],lambda b,y,n:y if b else n)
        self.assertEqual([r[0] for r in result[:5]],[0,2,3,5,7])
