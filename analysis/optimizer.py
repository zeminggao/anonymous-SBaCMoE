import os, math
import torch
import torch.distributed as dist

class MasterAdam:
    def __init__(self,model,lr):
        rank=dist.get_rank();world=dist.get_world_size();self.named=list(model.named_parameters());self.lr=lr;self.stepno=0;self.state={};self.dpowners={};loads=[0]*world
        shared=sorted([(n,p) for n,p in self.named if '.experts.' not in n],key=lambda t:-t[1].numel())
        for n,p in shared:
            owner=min(range(world),key=lambda r:(loads[r],r));self.dpowners[n]=owner;loads[owner]+=p.numel()
        for n,p in self.named:
            if '.experts.' in n or self.dpowners[n]==rank:self.state[n]=[p.detach().float().clone(),torch.zeros_like(p,dtype=torch.float32),torch.zeros_like(p,dtype=torch.float32)]
    @torch.no_grad()
    def step(self):
        world=dist.get_world_size();rank=dist.get_rank();norm=torch.zeros((),device='cuda',dtype=torch.float32)
        for n,p in self.named:
            assert p.grad is not None,n
            if '.experts.' not in n:dist.all_reduce(p.grad)
            p.grad.div_(world)
            if n in self.state:norm+=p.grad.float().square().sum()
        dist.all_reduce(norm);norm.sqrt_();assert bool(torch.isfinite(norm));scale=min(1.,1./(float(norm)+1e-6));self.stepno+=1;t=self.stepno
        for n,p in self.named:
            if n in self.state:
                master,m,v=self.state[n];g=p.grad.float().mul_(scale);m.mul_(.9).add_(g,alpha=.1);v.mul_(.95).addcmul_(g,g,value=.05);den=v.sqrt().div_(math.sqrt(1-.95**t)).add_(1e-8)
                master.mul_(1-self.lr*.1).addcdiv_(m,den,value=-self.lr/(1-.9**t));p.copy_(master);del g,den
            if '.experts.' not in n:dist.broadcast(p,src=self.dpowners[n])
        model_zero(self.named);return float(norm)
    def save(self,path,extra):
        # CPU serialization avoids an additional GPU copy; includes optimizer and RNG state.
        blob=dict(model={n:p.detach().cpu() for n,p in self.named},optimizer={n:[t.cpu() for t in s] for n,s in self.state.items()},optimizer_step=self.stepno,lr=self.lr,dpowners=self.dpowners,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),**extra)
        tmp=str(path)+'.tmp';torch.save(blob,tmp);os.replace(tmp,path)

def model_zero(named):
    for _,p in named:p.grad=None
