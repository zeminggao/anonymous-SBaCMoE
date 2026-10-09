import numpy as np
from numba import njit
from sbac.scheduler import _loads
@njit
def state(s,a,b,mb,nodes,k):
 aa,bb=_loads(s,a,b,mb,nodes);batch=mb*len(nodes);nb=len(s)//batch
 ref=np.empty(len(s))
 for i in range(len(s)):ref[i]=k[i].mean()
 xx=np.zeros(nb);xy=np.zeros(nb)
 for j in range(nb):
  for x in s[j*batch:(j+1)*batch]:
   xy[j]+=ref[x]
   for y in s[j*batch:(j+1)*batch]:xx[j]+=k[x,y]
 return aa,bb,ref,xx,xy
@njit
def rnd(s,a,b,mb,nodes,k,aa,bb,ref,xx,xy,eps,trials,seed,repair):
 np.random.seed(seed);n=len(s);batch=mb*len(nodes);yy=ref.mean();acc=0;rej=0
 for _ in range(trials):
  p=np.random.randint(n);q=np.random.randint(n)
  if p==q:continue
  u=p//batch;v=q//batch;x=s[p];y=s[q]
  sp=nodes[(p%batch)//mb];sq=nodes[(q%batch)//mb]
  if u==v and (sp==sq or repair):continue
  du=0.;dv=0.
  oldmu=xx[u]/batch**2+yy-2*xy[u]/batch
  oldmv=xx[v]/batch**2+yy-2*xy[v]/batch
  mu=oldmu;mv=oldmv
  if u!=v:
   for pos in range(u*batch,(u+1)*batch):
    if pos!=p:du+=2*(k[y,s[pos]]-k[x,s[pos]])
   du+=k[y,y]-k[x,x]
   for pos in range(v*batch,(v+1)*batch):
    if pos!=q:dv+=2*(k[x,s[pos]]-k[y,s[pos]])
   dv+=k[x,x]-k[y,y]
   mu=(xx[u]+du)/batch**2+yy-2*(xy[u]+ref[y]-ref[x])/batch
   mv=(xx[v]+dv)/batch**2+yy-2*(xy[v]+ref[x]-ref[y])/batch
  if repair:
   oldpen=max(oldmu-eps,0)**2+max(oldmv-eps,0)**2
   newpen=max(mu-eps,0)**2+max(mv-eps,0)**2
   if newpen>=oldpen-1e-16:continue
  elif max(mu,mv)>eps+1e-12:
   rej+=1;continue
  old=np.maximum(aa[u],bb[u]).sum()
  if u!=v:old+=np.maximum(aa[v],bb[v]).sum()
  if sp==0:aa[u]+=a[y]-a[x]
  else:bb[u]+=b[y]-b[x]
  if sq==0:aa[v]+=a[x]-a[y]
  else:bb[v]+=b[x]-b[y]
  new=np.maximum(aa[u],bb[u]).sum()
  if u!=v:new+=np.maximum(aa[v],bb[v]).sum()
  if repair or new<old-1e-6:
   s[p]=y;s[q]=x;acc+=1
   if u!=v:xx[u]+=du;xx[v]+=dv;xy[u]+=ref[y]-ref[x];xy[v]+=ref[x]-ref[y]
  else:
   if sp==0:aa[u]-=a[y]-a[x]
   else:bb[u]-=b[y]-b[x]
   if sq==0:aa[v]-=a[x]-a[y]
   else:bb[v]-=b[x]-b[y]
 return acc,rej
