"""Fixed-placement NUMA objective; sample-conserving, MMD-constrained swaps.

MMD is the biased RBF estimator (diagonal entries included), as in the
original SBaC implementation. This is a local search, not a proven optimum.
"""
import numpy as np
from numba import njit


def token_features(tokens, dimension=256):
    tokens = np.asarray(tokens)
    if tokens.ndim != 2 or not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError('tokens must be a two-dimensional integer array')
    if dimension < 1 or np.any(tokens < 0):
        raise ValueError('dimension must be positive and tokens nonnegative')
    features = np.stack([np.bincount(row % dimension, minlength=dimension)
                         for row in tokens]).astype(np.float64)
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def kernel_matrix(features, sigma=1.0):
    x = np.asarray(features, dtype=np.float64)
    if sigma <= 0 or x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('finite 2D features and positive sigma required')
    norm = (x*x).sum(1)
    d = np.maximum(norm[:, None] + norm[None, :] - 2*x@x.T, 0)
    return np.exp(-d/(2*sigma*sigma))


def batch_mmd2(schedule, kernel, batch_size):
    means = kernel.mean(1)
    return np.array([kernel[np.ix_(row, row)].mean() + kernel.mean()
                     - 2*means[row].mean()
                     for row in np.asarray(schedule).reshape(-1, batch_size)])


@njit
def _loads(schedule, a, b, microbatch, rank_nodes):
    batch_size = microbatch*len(rank_nodes)
    aa = np.zeros((len(schedule)//batch_size, a.shape[1]))
    bb = np.zeros_like(aa)
    for pos, sid in enumerate(schedule):
        if rank_nodes[(pos % batch_size)//microbatch] == 0:
            aa[pos//batch_size] += a[sid]
        else:
            bb[pos//batch_size] += b[sid]
    return aa, bb


def cost(schedule, a, b, microbatch=4, rank_nodes=(0, 0, 0, 1)):
    aa, bb = _loads(np.asarray(schedule), np.asarray(a), np.asarray(b),
                    microbatch, np.asarray(rank_nodes))
    return float(np.maximum(aa, bb).sum())


@njit
def _search(start, a, b, mb, nodes, kernel, epsilon, iterations, seed):
    np.random.seed(seed)
    s = start.copy(); batch_size = mb*len(nodes)
    aa, bb = _loads(s, a, b, mb, nodes)
    n = len(s); nb = n//batch_size
    ref = np.empty(n)
    for i in range(n): ref[i] = kernel[i].mean()
    yy = ref.mean()
    xx = np.zeros(nb); xy = np.zeros(nb)
    for j in range(nb):
        row = s[j*batch_size:(j+1)*batch_size]
        for x in row:
            xy[j] += ref[x]
            for y in row: xx[j] += kernel[x,y]
    accepted = 0; rejected_mmd = 0
    for _ in range(iterations):
        p = np.random.randint(n); q = np.random.randint(n)
        if p == q: continue
        u = p//batch_size; v = q//batch_size; x = s[p]; y = s[q]
        sidep = nodes[(p % batch_size)//mb]; sideq = nodes[(q % batch_size)//mb]
        if u == v and sidep == sideq: continue
        du = 0.; dv = 0.
        if u != v:
            for k in range(u*batch_size,(u+1)*batch_size):
                if k != p: du += 2*(kernel[y,s[k]]-kernel[x,s[k]])
            du += kernel[y,y]-kernel[x,x]
            for k in range(v*batch_size,(v+1)*batch_size):
                if k != q: dv += 2*(kernel[x,s[k]]-kernel[y,s[k]])
            dv += kernel[x,x]-kernel[y,y]
            mu = (xx[u]+du)/(batch_size**2)+yy-2*(xy[u]+ref[y]-ref[x])/batch_size
            mv = (xx[v]+dv)/(batch_size**2)+yy-2*(xy[v]+ref[x]-ref[y])/batch_size
            if max(mu,mv) > epsilon+1e-12:
                rejected_mmd += 1; continue
        old = np.maximum(aa[u],bb[u]).sum()
        if u != v: old += np.maximum(aa[v],bb[v]).sum()
        if sidep == 0: aa[u] += a[y]-a[x]
        else: bb[u] += b[y]-b[x]
        if sideq == 0: aa[v] += a[x]-a[y]
        else: bb[v] += b[x]-b[y]
        new = np.maximum(aa[u],bb[u]).sum()
        if u != v: new += np.maximum(aa[v],bb[v]).sum()
        if new < old-1e-6:
            s[p] = y; s[q] = x; accepted += 1
            if u != v:
                xx[u] += du; xx[v] += dv
                xy[u] += ref[y]-ref[x]; xy[v] += ref[x]-ref[y]
        else:
            if sidep == 0: aa[u] -= a[y]-a[x]
            else: bb[u] -= b[y]-b[x]
            if sideq == 0: aa[v] -= a[x]-a[y]
            else: bb[v] -= b[x]-b[y]
    return s, accepted, rejected_mmd


def regroup(a, b, tokens, microbatch=4, rank_nodes=(0,0,0,1), epsilon=0.02,
            sigma=1.0, feature_dim=256, iterations=None, seeds=(1729,1730,1731)):
    """Return local sequence indices and an audit; never silently relax epsilon.

    a[s,l]: remote token-expert assignments if sequence s originates on node 0.
    b[s,l]: corresponding count if it originates on node 1.
    Every dispatch has microbatch samples per rank. All samples are consumed once.
    The reference distribution is the *entire current window*.
    """
    a=np.ascontiguousarray(a,dtype=np.float64);b=np.ascontiguousarray(b,dtype=np.float64)
    nodes=np.asarray(rank_nodes,dtype=np.int64);n=len(a);batch=microbatch*len(nodes)
    if microbatch<1 or set(nodes.tolist())!={0,1} or not n or n%batch:
        raise ValueError('require two NUMA nodes and complete dispatches')
    if a.ndim!=2 or b.shape!=a.shape or len(tokens)!=n or not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a<0) or np.any(b<0):
        raise ValueError('invalid route counts or token/sample alignment')
    if not np.isfinite(epsilon) or epsilon<0:raise ValueError('epsilon must be finite and nonnegative')
    kernel=kernel_matrix(token_features(tokens,feature_dim),sigma)
    start=np.arange(n,dtype=np.int64);before_mmd=batch_mmd2(start,kernel,batch)
    if before_mmd.max()>epsilon+1e-12:
        raise ValueError(f'FIFO is infeasible: max MMD2={before_mmd.max():.8g} > epsilon={epsilon}; calibrate explicitly before running')
    before=cost(start,a,b,microbatch,nodes);best=start;accepted=0;rejected=0
    iterations=n*1600 if iterations is None else int(iterations)
    if iterations<0:raise ValueError('iterations must be nonnegative')
    for seed in seeds:
        best,acc,rej=_search(best,a,b,microbatch,nodes,kernel,epsilon,iterations,int(seed))
        accepted+=acc;rejected+=rej
    final=cost(best,a,b,microbatch,nodes);after_mmd=batch_mmd2(best,kernel,batch)
    assert np.array_equal(np.sort(best),start)
    assert final<=before+1e-6 and after_mmd.max()<=epsilon+1e-10
    return best,dict(before=before,after=final,gain_pct=100*(1-final/before) if before else 0,
        mmd_epsilon=float(epsilon),sigma=sigma,feature_dim=feature_dim,
        initial_max_mmd2=float(before_mmd.max()),final_max_mmd2=float(after_mmd.max()),
        accepted_swaps=accepted,mmd_rejections=rejected,sample_multiset_verified=True,
        objective='sum_dispatch,sum_layer max(cross_0_to_1,cross_1_to_0)',
        scope='fixed-placement route-count proxy; heuristic local search, not runtime speedup')
