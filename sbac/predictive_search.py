"""MMD-constrained search with feasibility repair and relative-gain stopping."""
import os,time,json
import numpy as np
from sbac.search_core import state,rnd
from sbac.scheduler import token_features,kernel_matrix,cost

def search(tokens,c,own,seed,out,tag):
    start=time.perf_counter();timers={};nodes=np.array([0,0,0,1]);mb=4;n=len(tokens);s=np.arange(n)
    t=time.perf_counter();a=np.ascontiguousarray((c*(own==3)).sum(-1),dtype=float);b=np.ascontiguousarray((c*(own!=3)).sum(-1),dtype=float);timers['traffic_s']=time.perf_counter()-t
    t=time.perf_counter();k=kernel_matrix(token_features(tokens));timers['mmd_kernel_s']=time.perf_counter()-t
    t=time.perf_counter();aa,bb,ref,xx,xy=state(s,a,b,mb,nodes,k);fifo=float(np.maximum(aa,bb).sum());timers['init_s']=time.perf_counter()-t
    t=time.perf_counter();repair=0
    while (xx/256+ref.mean()-xy/8).max()>.02+1e-12 and repair<1000:
        rnd(s,a,b,mb,nodes,k,aa,bb,ref,xx,xy,.02,n,seed+repair,True);repair+=1
    timers['repair_s']=time.perf_counter()-t;assert (xx/256+ref.mean()-xy/8).max()<=.02+1e-10,'MMD repair failed; threshold unchanged'
    t=time.perf_counter();trace=[];streak=0
    for i in range(2000):
        before=float(np.maximum(aa,bb).sum());ac,rj=rnd(s,a,b,mb,nodes,k,aa,bb,ref,xx,xy,.02,n,seed+10000+i,False)
        after=float(np.maximum(aa,bb).sum());gain=(before-after)/before;streak=streak+1 if gain<.001 else 0
        trace.append(dict(round=i+1,cost=after,relative_gain=gain,low_gain_streak=streak,accepted=ac,mmd_rejected=rj))
        if streak>=10:break
    assert streak>=10;timers['search_s']=time.perf_counter()-t
    t=time.perf_counter();assert np.array_equal(np.sort(s),np.arange(n));yy=ref.mean()
    exact=np.array([k[np.ix_(z,z)].mean()+yy-2*ref[z].mean() for z in s.reshape(-1,16)])
    np.testing.assert_allclose(exact,xx/256+yy-xy/8,atol=1e-10);assert exact.max()<=.02+1e-10
    np.testing.assert_allclose(cost(s,a,b,mb,nodes),after,rtol=1e-12);timers['audit_s']=time.perf_counter()-t
    t=time.perf_counter()
    with open(out/(tag+'.npy'),'wb') as f:np.save(f,s);f.flush();os.fsync(f.fileno())
    np.testing.assert_array_equal(np.load(out/(tag+'.npy')),s);timers['delivery_s']=time.perf_counter()-t
    wall=time.perf_counter()-start;(out/(tag+'_trace.json')).write_text(json.dumps(trace))
    return s,dict(timers=timers,wall_s=wall,rounds=len(trace),repair_rounds=repair,mmd_max=float(exact.max()),fifo=fifo,final=after,gain_pct=100*(1-after/fifo))

def evaluate(s,c,own):
    a=np.ascontiguousarray((c*(own==3)).sum(-1),dtype=float);b=np.ascontiguousarray((c*(own!=3)).sum(-1),dtype=float)
    return float(cost(s,a,b,4,np.array([0,0,0,1])))
