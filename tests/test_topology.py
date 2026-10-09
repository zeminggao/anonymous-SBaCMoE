import json,tempfile,unittest,warnings
from pathlib import Path
import numpy as np
from sbac.topology import load_topology
from sbac.predictive_search import search,evaluate

class TopologyTests(unittest.TestCase):
    def test_ep16_default(self):
        with warnings.catch_warnings(record=True) as found:
            config=load_topology(Path(__file__).resolve().parents[1]/'configs/ep16.json')
        self.assertEqual(config['world_size'],16)
        self.assertEqual(len(config['rank_to_bottleneck_side']),16)
        self.assertTrue(found)

    def test_weighted_ep16_objective(self):
        rng=np.random.default_rng(11);tokens=rng.integers(0,5,(128,32));counts=rng.random((128,2,64))
        owner=np.tile(np.repeat(np.arange(16),4),(2,1));sides=np.array([0]*8+[1]*8)
        expected=0.
        for start in (0,64):
            directions=np.zeros((2,2))
            for pos in range(64):
                source_side=sides[pos//4]
                for l in range(2):
                    for e in range(64):
                        if sides[owner[l,e]]!=source_side:directions[source_side,l]+=counts[start+pos,l,e]
            expected+=np.maximum(directions[0]*3,directions[1]).sum()
        self.assertAlmostEqual(evaluate(np.arange(128),counts,owner,inverse_bandwidth=(3,1)),expected)
        with tempfile.TemporaryDirectory() as td:
            plan,result=search(tokens,counts,owner,5,Path(td),'ep16',inverse_bandwidth=(3,1))
        np.testing.assert_array_equal(np.sort(plan),np.arange(128))
        self.assertAlmostEqual(result['fifo'],expected)
        self.assertAlmostEqual(result['final'],evaluate(plan,counts,owner,inverse_bandwidth=(3,1)))
        self.assertLessEqual(result['mmd_max'],.02+1e-10)

    def test_invalid_topology(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bad.json';p.write_text(json.dumps(dict(world_size=16,rank_to_bottleneck_side=[0,1],inverse_bandwidth_weights=[1,1])))
            with self.assertRaises(ValueError):load_topology(p)

if __name__=='__main__':unittest.main()
