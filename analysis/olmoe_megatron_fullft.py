"""OLMoE -> Megatron Core 0.15.3. Native MoE router/dispatcher/experts.

OLMoE-specific full-projection QK RMSNorm and SDPA are ModuleSpec extensions.
BF16 training uses the previously verified FP32-master sharded Adam harness.
"""
import os,json,time,argparse,gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from safetensors import safe_open
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from optimizer import MasterAdam,model_zero
from megatron.core.transformer.moe import moe_utils
# 0.15.3 optional-TE import leaves this sentinel undefined without TE.
if not hasattr(moe_utils,"te_general_gemm"):moe_utils.te_general_gemm=None

class SharedStateOffloadAdam(MasterAdam):
    """Same FP32 math; park only shared-parameter optimizer states during F/B."""
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.park()
    def park(self):
        for n in list(self.state):
            if '.experts.' not in n:self.state[n]=[x.cpu() for x in self.state[n]]
    @torch.no_grad()
    def step(self):
        # Bound temporary GPU memory, including the large embedding/output states.
        world=dist.get_world_size();norm=torch.zeros((),device='cuda',dtype=torch.float32)
        chunk=1048576
        for n,p in self.named:
            assert p.grad is not None,n
            if '.experts.' not in n:dist.all_reduce(p.grad)
            p.grad.div_(world)
            if n in self.state:
                for g in p.grad.reshape(-1).split(chunk):norm+=g.float().square().sum()
        dist.all_reduce(norm);norm.sqrt_();assert bool(torch.isfinite(norm))
        scale=min(1.,1./(float(norm)+1e-6));self.stepno+=1;t=self.stepno
        for n,p in self.named:
            if n in self.state:
                state=[x.reshape(-1) for x in self.state[n]]
                flat=p.reshape(-1);grad=p.grad.reshape(-1)
                for i in range(0,p.numel(),chunk):
                    views=[x[i:i+chunk] for x in state]
                    master,m,v=[x.to(p.device) for x in views]
                    g=grad[i:i+chunk].float().mul_(scale)
                    m.mul_(.9).add_(g,alpha=.1);v.mul_(.95).addcmul_(g,g,value=.05)
                    den=v.sqrt().div_((1-.95**t)**.5).add_(1e-8)
                    master.mul_(1-self.lr*.1).addcdiv_(m,den,value=-self.lr/(1-.9**t))
                    flat[i:i+chunk].copy_(master)
                    if views[0].device!=p.device:
                        for dst,src in zip(views,[master,m,v]):dst.copy_(src)
                    del master,m,v,g,den,views
            if '.experts.' not in n:dist.broadcast(p,src=self.dpowners[n])
            p.grad=None
        return float(norm)

class RMS(torch.nn.Module):
    def __init__(self,config,hidden_size,eps=1e-5,**kw):
        super().__init__();self.weight=torch.nn.Parameter(torch.ones(hidden_size,dtype=config.params_dtype));self.eps=eps
    def forward(self,x):
        y=x.float();return self.weight*(y*torch.rsqrt(y.square().mean(-1,keepdim=True)+self.eps)).to(x.dtype)

class FullQKRMS(RMS):
    def __init__(self,config,hidden_size,eps=1e-5,**kw):super().__init__(config,config.hidden_size,eps)
    def forward(self,x):return super().forward(x.flatten(-2)).reshape(x.shape)

class SDPA(torch.nn.Module):
    def __init__(self,config,layer_number,**kw):super().__init__();self.dropout=config.attention_dropout
    def forward(self,q,k,v,attention_mask=None,**kw):
        y=F.scaled_dot_product_attention(q.permute(1,2,0,3),k.permute(1,2,0,3),v.permute(1,2,0,3),dropout_p=self.dropout if self.training else 0.,is_causal=True)
        return y.permute(2,0,1,3).contiguous().flatten(-2)

def make(root,dtype,load_weights=True):
    cfg=TransformerConfig(num_layers=16,hidden_size=2048,num_attention_heads=16,num_query_groups=16,ffn_hidden_size=1024,
        num_moe_experts=64,moe_ffn_hidden_size=1024,moe_router_topk=8,moe_router_pre_softmax=True,
        moe_router_load_balancing_type='none',moe_aux_loss_coeff=0.,moe_token_dispatcher_type='alltoall',
        expert_model_parallel_size=4,tensor_model_parallel_size=1,pipeline_model_parallel_size=1,
        normalization='RMSNorm',layernorm_epsilon=1e-5,qk_layernorm=True,add_bias_linear=False,
        gated_linear_unit=True,activation_func=F.silu,hidden_dropout=0.,attention_dropout=0.,
        params_dtype=dtype,bf16=dtype==torch.bfloat16,use_cpu_initialization=True,perform_initialization=False,
        gradient_accumulation_fusion=False,masked_softmax_fusion=False,bias_activation_fusion=False,
        bias_dropout_fusion=False,apply_rope_fusion=False,moe_grouped_gemm=False)
    spec=get_gpt_layer_local_spec(num_experts=64,normalization='RMSNorm',qk_layernorm=True)
    spec.submodules.input_layernorm=RMS;spec.submodules.pre_mlp_layernorm=RMS
    spec.submodules.self_attention.submodules.q_layernorm=FullQKRMS
    spec.submodules.self_attention.submodules.k_layernorm=FullQKRMS
    spec.submodules.self_attention.submodules.core_attention=SDPA
    model=GPTModel(cfg,spec,50304,4096,share_embeddings_and_output_weights=False,position_embedding_type='rope',rotary_base=10000)
    model.decoder.final_layernorm=RMS(cfg,2048,1e-5)
    owner=np.load(root/'data/native_placements.npz')['compute_balanced']
    assert owner.shape==(16,64) and np.issubdtype(owner.dtype,np.integer)
    assert all(np.array_equal(np.bincount(row,minlength=4),np.full(4,16)) for row in owner)
    order=np.array([np.concatenate([np.where(row==r)[0] for r in range(4)]) for row in owner])
    if not load_weights:
        model=model.cuda();model.post_process=False;return model,order
    index=json.loads((root/'model/model.safetensors.index.json').read_text())['weight_map']
    handles={name:safe_open(str(root/'model'/name),framework='pt',device='cpu') for name in set(index.values())}
    def get(k):return handles[index[k]].get_tensor(k)
    mapped=set()
    named=dict(model.named_parameters())
    def put(k,t):
        assert k in named,k;assert named[k].shape==t.shape,(k,named[k].shape,t.shape)
        with torch.no_grad():named[k].copy_(t)
        mapped.add(k)
    put('embedding.word_embeddings.weight',get('model.embed_tokens.weight'))
    put('output_layer.weight',get('lm_head.weight'))
    put('decoder.final_layernorm.weight',get('model.norm.weight'))
    for l in range(16):
        src=f'model.layers.{l}.';dst=f'decoder.layers.{l}.'
        for a,b in [('input_layernorm.weight','input_layernorm.weight'),('pre_mlp_layernorm.weight','post_attention_layernorm.weight'),('self_attention.linear_proj.weight','self_attn.o_proj.weight'),('self_attention.q_layernorm.weight','self_attn.q_norm.weight'),('self_attention.k_layernorm.weight','self_attn.k_norm.weight')]:put(dst+a,get(src+b))
        qkv=torch.stack([get(src+'self_attn.'+k+'_proj.weight').reshape(16,128,2048) for k in ['q','k','v']],1).reshape(6144,2048)
        put(dst+'self_attention.linear_qkv.weight',qkv)
        put(dst+'mlp.router.weight',get(src+'mlp.gate.weight')[order[l]])
        for local,e in enumerate(order[l,dist.get_rank()*16:(dist.get_rank()+1)*16]):
            pre=src+f'mlp.experts.{e}.';target=dst+f'mlp.experts.local_experts.{local}.'
            put(target+'linear_fc1.weight',torch.cat([get(pre+'gate_proj.weight'),get(pre+'up_proj.weight')]))
            put(target+'linear_fc2.weight',get(pre+'down_proj.weight'))
    assert mapped==set(named),set(named)-mapped
    report=dict(rank=dist.get_rank(),parameters=sum(p.numel() for p in model.parameters()),mapped_parameters=len(mapped),expert_order=order.tolist(),qk_norm='full projection 2048',routing='softmax over all 64 then Top8, no renormalization',framework='Megatron Core 0.15.3 native router/alltoall dispatcher/sequential experts')
    (root/f'megatron_mapping_rank{dist.get_rank()}.json').write_text(json.dumps(report,indent=2))
    model=model.cuda();model.post_process=False
    return model,order

def hidden(model,ids):
    pos=torch.arange(ids.shape[1],device=ids.device)[None].expand_as(ids)
    return model(ids,pos,None).transpose(0,1)

def validate(model,root,order):
    from transformers import OlmoeForCausalLM
    rank=dist.get_rank();model.eval()
    hf=OlmoeForCausalLM.from_pretrained(root/'model',torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).eval()
    ids=torch.tensor(np.load(root/'data/input_ids.npy')[rank,:32].astype(np.int64))[None]
    # CPU FP32 reference avoids a second 27GB GPU model and disables BF16 ambiguity.
    with torch.no_grad():
        a=hf.model(ids,use_cache=False).last_hidden_state
        b=hidden(model,ids.cuda()).cpu()
        la=F.cross_entropy(hf.lm_head(a[:,:-1]).reshape(-1,50304),ids[:,1:].reshape(-1))
        lb=F.cross_entropy(F.linear(b[:,:-1].cuda(),model.output_layer.weight).cpu().reshape(-1,50304),ids[:,1:].reshape(-1))
    rel=float((a-b).norm()/a.norm());delta=float(abs(la-lb))
    result=dict(hidden_relative_error=rel,loss_abs_delta=delta,hf_loss=float(la),megatron_loss=float(lb),reference='Unmodified native HF FP32 on CPU',tokens=32)
    (root/f'megatron_correctness_rank{rank}.json').write_text(json.dumps(result,indent=2))
    assert rel<1e-3 and delta<1e-3,result
    # Check shared router/QK norm and one EP-owned expert against native autograd.
    hf.requires_grad_(False)
    logical=int(order[0,0])
    refs=[hf.model.layers[0].mlp.gate.weight,hf.model.layers[0].self_attn.q_norm.weight,hf.model.layers[0].mlp.experts[logical].gate_proj.weight]
    for p in refs:p.requires_grad_(True)
    ha=hf.model(ids,use_cache=False).last_hidden_state
    memory_setup(model);model.train()
    hb=hidden(model,ids.cuda())
    torch.manual_seed(987+rank);upstream=torch.randn_like(ha)/ha.numel()
    (ha*upstream).sum().backward();(hb*upstream.cuda()).sum().backward()
    pairs=[('router',refs[0].grad[order[0]].cuda(),model.decoder.layers[0].mlp.router.weight.grad),('q_norm',refs[1].grad.cuda(),model.decoder.layers[0].self_attention.q_layernorm.weight.grad)]
    eg=refs[2].grad.cuda();dist.all_reduce(eg)
    if rank==0:pairs.append(('expert_gate',eg,model.decoder.layers[0].mlp.experts.local_experts[0].linear_fc1.weight.grad[:1024]))
    result['gradient_relative_errors']={name:float((x-y).norm()/x.norm().clamp_min(1e-12)) for name,x,y in pairs}
    (root/f'megatron_correctness_rank{rank}.json').write_text(json.dumps(result,indent=2))
    assert max(result['gradient_relative_errors'].values())<.002,result
    del hf,a,b;gc.collect()

def memory_setup(model):
    # Full-layer recomputation, plus inner expert recomputation to bound backward peaks.
    model.config.recompute_granularity='full';model.config.recompute_method='uniform';model.config.recompute_num_layers=1
    for layer in model.decoder.layers:
        for expert in layer.mlp.experts.local_experts:
            original=expert.forward
            def recompute(x,*a,_fn=original,**kw):
                return checkpoint(lambda z:_fn(z,*a,**kw),x,use_reentrant=False) if torch.is_grad_enabled() else _fn(x,*a,**kw)
            expert.forward=recompute
        moe_forward=layer.mlp.forward
        def recompute_moe(x,*a,_fn=moe_forward,**kw):
            return checkpoint(lambda z:_fn(z,*a,**kw),x,use_reentrant=False) if torch.is_grad_enabled() else _fn(x,*a,**kw)
        layer.mlp.forward=recompute_moe
        # Avoid retaining all expert outputs plus a second full torch.cat buffer.
        experts=layer.mlp.experts
        def sequential_into_buffer(x,tokens_per_expert,probs,_experts=experts):
            out=torch.empty_like(x);offset=0
            for expert,n in zip(_experts.local_experts,tokens_per_expert.tolist()):
                y,bias=expert(x[offset:offset+n],probs[offset:offset+n]);assert bias is None
                out[offset:offset+n]=y;offset+=n
            assert offset==len(x)
            return out,None
        experts.forward=sequential_into_buffer

def train(model,root,args,order):
    rank=dist.get_rank();model.train()
    memory_setup(model)
    opt=(SharedStateOffloadAdam if args.shared_state_offload else MasterAdam)(model,1e-6)
    cal=np.array(json.loads((root/'data/data.json').read_text())['calibration'])
    bank=np.load(root/'data/input_ids.npy')[~cal]
    if args.arm=='fifo':schedule=np.arange(512)
    else:
        assert args.mb==4
        schedule=np.load(root/'data/oracle_5090_schedules.npz')['source_contiguous_mb4_w512_any_source']
    plan=schedule[:args.steps*4*args.mb].reshape(args.steps,4,args.mb)
    suffix='_sharedoffload' if args.shared_state_offload else ''
    path=root/f'megatron_{args.arm}_mb{args.mb}{suffix}_rank{rank}.jsonl'
    for step,batch in enumerate(plan):
        ids=torch.tensor(bank[batch[rank]].astype(np.int64),device='cuda');model_zero(opt.named)
        torch.cuda.synchronize();dist.barrier();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        h=hidden(model,ids)[:,:-1].reshape(-1,2048);target=ids[:,1:].reshape(-1);loss=0
        for i in range(0,len(target),128):
            loss=loss+checkpoint(lambda x,y:F.cross_entropy(F.linear(x,model.output_layer.weight).float(),y,reduction='sum'),h[i:i+128],target[i:i+128],use_reentrant=True)/len(target)
        loss.backward();norm=opt.step();torch.cuda.synchronize()
        values=torch.tensor([float(loss),time.perf_counter()-start,torch.cuda.max_memory_allocated()/2**30],device='cuda');dist.all_reduce(values,op=dist.ReduceOp.MAX)
        row=dict(shared_state_offload=args.shared_state_offload,step=step+1,mb=args.mb,arm=args.arm,loss_local=float(loss),full_step_max_s=float(values[1]),peak_gib=float(values[2]),gradnorm=norm)
        with path.open('a') as f:f.write(json.dumps(row)+'\n')
        if rank==0:print(json.dumps(row),flush=True)
        del h,target,ids,loss
