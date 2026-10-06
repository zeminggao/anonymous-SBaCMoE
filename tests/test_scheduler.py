import unittest
import numpy as np
from sbac import regroup,token_features,kernel_matrix,batch_mmd2,cost

class SchedulerTests(unittest.TestCase):
    def test_original_mmd_formula(self):
        rng=np.random.default_rng(12);tokens=rng.integers(0,900,(32,64))
        x=token_features(tokens);k=kernel_matrix(x);row=np.arange(16)
        def rbf(a,b):return np.exp(-((a[:,None]-b[None,:])**2).sum(-1)/2).mean()
        direct=rbf(x[row],x[row])+rbf(x,x)-2*rbf(x[row],x)
        self.assertAlmostEqual(batch_mmd2(np.arange(32),k,16)[0],direct,places=12)

    def test_constraints_and_objective(self):
        rng=np.random.default_rng(42)
        for nodes in [(0,0,0,1),(0,0,1,1)]:
            for mb in [1,2,4]:
                n=64;tokens=rng.integers(0,3000,(n,64));a=rng.integers(0,100,(n,3));b=rng.integers(0,100,(n,3))
                k=kernel_matrix(token_features(tokens));eps=float(batch_mmd2(np.arange(n),k,mb*4).max())+1e-10
                s,audit=regroup(a,b,tokens,mb,nodes,eps,iterations=2000,seeds=(2,))
                np.testing.assert_array_equal(np.sort(s),np.arange(n))
                self.assertLessEqual(batch_mmd2(s,k,mb*4).max(),eps+1e-10)
                self.assertLessEqual(cost(s,a,b,mb,nodes),cost(np.arange(n),a,b,mb,nodes)+1e-8)
                self.assertGreater(audit['accepted_swaps'],0)
                self.assertGreater(audit['mmd_rejections'],0)
                again,_=regroup(a,b,tokens,mb,nodes,eps,iterations=2000,seeds=(2,))
                np.testing.assert_array_equal(s,again)

    def test_infeasible_fails_closed(self):
        tokens=np.repeat(np.array([0,1]),16)[:,None];a=np.ones((32,2))
        with self.assertRaisesRegex(ValueError,'infeasible'):regroup(a,a,tokens,epsilon=0)

    def test_no_traffic(self):
        a=np.zeros((32,2));tokens=np.zeros((32,4),dtype=int)
        s,audit=regroup(a,a,tokens,epsilon=0,iterations=100)
        self.assertEqual(audit['gain_pct'],0)
        np.testing.assert_array_equal(s,np.arange(32))

if __name__=='__main__':unittest.main()
