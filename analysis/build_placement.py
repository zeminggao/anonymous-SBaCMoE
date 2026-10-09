"""Compute-only balanced-cardinality placement from calibration route counts."""
import argparse
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('counts',help='NPY [sequences,16,64], calibration samples only')
    p.add_argument('--ranks',type=int,default=16);p.add_argument('--output',default='work/data/native_placements.npz');a=p.parse_args()
    assert a.ranks>0 and 64%a.ranks==0
    counts=np.load(a.counts,allow_pickle=False)
    assert counts.ndim==3 and counts.shape[1:]==(16,64) and np.isfinite(counts).all() and (counts>=0).all()
    weights=counts.sum(0);owner=np.empty((16,64),dtype=np.int64)
    for l in range(16):
        loads=np.zeros(a.ranks);sizes=np.zeros(a.ranks,dtype=int)
        for e in np.argsort(-weights[l],kind='stable'):
            rank=min((r for r in range(a.ranks) if sizes[r]<64//a.ranks),key=lambda r:(loads[r],r))
            owner[l,e]=rank;loads[rank]+=weights[l,e];sizes[rank]+=1
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    np.savez(target,compute_balanced=owner)

if __name__=='__main__':main()
