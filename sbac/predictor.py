"""Causal OLMoE token-to-expert marginals; no model forward or route collection.

Requires previously saved token-level route IDs, not sequence count aggregates.
Selection probabilities replace the small model's Top-2 pair table: enumerating
64-choose-8 expert combinations would be impractical. Scores are expected counts,
not joint Top-8 probabilities. Prefix/rest tables retain the existing split at 128.
"""
import time
import numpy as np
from numba import njit

@njit(cache=True)
def accumulate(hist,seen,tokens,routes,prefix):
    for s in range(len(tokens)):
        for t in range(tokens.shape[1]):
            seg=0 if t<prefix else 1;tok=tokens[s,t]
            seen[seg,tok]+=1
            for l in range(routes.shape[1]):
                for k in range(routes.shape[-1]):hist[seg,l,tok,routes[s,l,t,k]]+=1

@njit(cache=True)
def expected_counts(tokens,tables,prefix):
    n=len(tokens);layers=tables.shape[1];experts=tables.shape[3]
    out=np.zeros((n,layers,experts),np.float64)
    for s in range(n):
        for t in range(tokens.shape[1]):
            seg=0 if t<prefix else 1;tok=tokens[s,t]
            for l in range(layers):
                for e in range(experts):out[s,l,e]+=tables[seg,l,tok,e]
    return out

class TokenMarginalPredictor:
    def __init__(self,vocab=50304,layers=16,experts=64,topk=8,prefix=128,pseudocount=32.):
        assert 0<topk<=experts and pseudocount>0
        self.vocab=vocab;self.layers=layers;self.experts=experts;self.topk=topk
        self.prefix=prefix;self.pseudocount=pseudocount;self.last_feedback_id=-1
        self.hist=np.zeros((2,layers,vocab,experts),np.uint64)
        self.seen=np.zeros((2,vocab),np.uint64);self.tables=None

    def update(self,tokens,routes,sample_ids,next_window_first_id):
        start=time.perf_counter();tokens=np.asarray(tokens);routes=np.asarray(routes)
        ids=np.asarray(sample_ids)
        assert tokens.ndim==2 and len(tokens)>0
        assert routes.shape==(len(tokens),self.layers,tokens.shape[1],self.topk)
        assert ids.shape==(len(tokens),) and len(np.unique(ids))==len(ids)
        assert ids.min()>self.last_feedback_id and ids.max()<next_window_first_id
        assert tokens.min()>=0 and tokens.max()<self.vocab
        assert routes.min()>=0 and routes.max()<self.experts
        assert np.all(np.diff(np.sort(routes,axis=-1),axis=-1)>0)
        accumulate(self.hist,self.seen,tokens,routes,self.prefix)
        table=np.empty(self.hist.shape,np.float32)
        for seg in range(2):
            count=float(self.seen[seg].sum())
            for l in range(self.layers):
                # Uniform inclusion prior has sum Top-K and each value <= 1.
                prior=(self.hist[seg,l].sum(0).astype(float)+self.topk/self.experts)/(count+1.)
                table[seg,l]=(self.hist[seg,l]+self.pseudocount*prior)/(self.seen[seg,:,None]+self.pseudocount)
        np.testing.assert_allclose(table.sum(-1),self.topk,atol=2e-5)
        assert table.min()>=0 and table.max()<=1+1e-6
        self.tables=table;self.last_feedback_id=int(ids.max())
        return dict(update_seconds=time.perf_counter()-start,feedback_sequences=len(tokens),last_feedback_id=self.last_feedback_id)

    def predict(self,tokens,first_sample_id):
        start=time.perf_counter();tokens=np.asarray(tokens)
        assert self.tables is not None and first_sample_id>self.last_feedback_id
        assert tokens.ndim==2 and tokens.min()>=0 and tokens.max()<self.vocab
        result=expected_counts(tokens,self.tables,self.prefix)
        np.testing.assert_allclose(result.sum(-1),tokens.shape[1]*self.topk,rtol=2e-6)
        assert result.min()>=0 and result.max()<=tokens.shape[1]+1e-4
        return result,dict(prediction_seconds=time.perf_counter()-start)

def self_test():
    # Synthetic correctness checks only; deliberately not a timing benchmark.
    rng=np.random.default_rng(1);tokens=rng.integers(0,11,(4,12))
    routes=np.argsort(rng.random((4,2,12,8)),axis=-1)[...,:3]
    p=TokenMarginalPredictor(vocab=11,layers=2,experts=8,topk=3,prefix=4)
    p.update(tokens,routes,np.arange(4),4);out,_=p.predict(tokens,4)
    reference=np.zeros_like(out)
    for s in range(4):
        for t in range(12):reference[s]+=p.tables[int(t>=4),:,tokens[s,t]]
    np.testing.assert_allclose(out,reference)
    try:p.predict(tokens,3)
    except AssertionError:pass
    else:raise AssertionError('future leakage guard failed')
    print('PASS: marginal normalization, independent prediction, causal guard; synthetic only')

if __name__=='__main__':self_test()
