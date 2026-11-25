import unittest
import numpy as np
from sparse_pca.synthetic import generate
from sparse_pca.data import partition
from sparse_pca.reference import dense_operator,sparse_operator,distributed_operator,power_iteration,oracle
from sparse_pca.validation import private_operator


class ReferenceTests(unittest.TestCase):
    def test_centering_and_partitions(self):
        cases=[np.zeros((6,4)),np.ones((6,4)),np.diag([1.,2.,3.,4.,5.,6.]),
               np.array([[0,0,0],[1,0,0],[0,0,0],[0,2,0],[0,0,0],[0,0,0]],float)]
        for support in ['uniform','uneven','clustered']:
            cases.append(generate(9,5,.35,seed=13,support=support))
        for x in cases:
            with self.subTest(shape=x.shape,sum=x.sum()):
                v=np.random.default_rng(5).normal(size=x.shape[1])
                z=x-x.mean(0); expected=z.T@(z@v)
                for actual in [dense_operator(x,v),sparse_operator(x,v),
                    distributed_operator(partition(x,[.5,.25,.25],7),v),
                    private_operator(x,v,x.size)]:
                    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)

    def test_power_iteration(self):
        x=np.array([[0,0],[3,0],[0,1],[0,0],[3,0],[0,1]],float)
        v,lam=power_iteration(x,80,seed=4); rv,rl=oracle(x)
        self.assertGreater(abs(v@rv),.999999)
        self.assertAlmostEqual(lam,rl,places=9)

    def test_zero_and_constant(self):
        for x in [np.zeros((6,3)),np.ones((6,3))]:
            v,lam=power_iteration(x)
            self.assertTrue(np.isfinite(v).all()); self.assertAlmostEqual(lam,0,places=12)

    def test_support_density_and_determinism(self):
        for support in ['uniform','uneven','clustered']:
            for density in [0,.1,1]:
                a=generate(6,5,density,7,support)
                np.testing.assert_array_equal(a,generate(6,5,density,7,support))
                self.assertEqual(np.count_nonzero(a),round(30*density))

    def test_invalid_inputs(self):
        for args in [(1,3,.2),(4,0,.2),(4,3,-.1),(4,3,1.1)]:
            with self.assertRaises(ValueError): generate(*args)
        with self.assertRaises(ValueError): partition(np.zeros((6,2)),[1,0,0],1)
