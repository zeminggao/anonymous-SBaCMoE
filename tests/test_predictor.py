import tempfile,unittest
from pathlib import Path
import numpy as np
from sbac.predictor import TokenMarginalPredictor
from sbac.predictive_search import search

class PredictorTests(unittest.TestCase):
    def test_causal_marginals(self):
        rng=np.random.default_rng(3);tokens=rng.integers(0,9,(4,12))
        routes=np.argsort(rng.random((4,2,12,8)),axis=-1)[...,:3]
        p=TokenMarginalPredictor(vocab=9,layers=2,experts=8,topk=3,prefix=4)
        p.update(tokens,routes,np.arange(4),4);value,_=p.predict(tokens,4)
        expected=np.zeros_like(value)
        for s in range(4):
            for t in range(12):expected[s]+=p.tables[int(t>=4),:,tokens[s,t]]
        np.testing.assert_allclose(value,expected)
        np.testing.assert_allclose(value.sum(-1),36,atol=1e-5)
        with self.assertRaises(AssertionError):p.predict(tokens,3)
        with self.assertRaises(AssertionError):p.update(tokens,routes,np.arange(4),4)

    def test_search_conservation_and_stopping(self):
        rng=np.random.default_rng(8);tokens=rng.integers(0,5,(32,32));counts=rng.random((32,2,64))
        owner=np.tile(np.repeat(np.arange(4),16),(2,1))
        with tempfile.TemporaryDirectory() as td:
            plan,result=search(tokens,counts,owner,5,Path(td),'test')
        np.testing.assert_array_equal(np.sort(plan),np.arange(32))
        self.assertLessEqual(result['mmd_max'],.02+1e-10)
        self.assertGreaterEqual(result['rounds'],10)

if __name__=='__main__':unittest.main()
