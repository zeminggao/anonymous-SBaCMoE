"""Paired all-parameter SFT; global shuffle, optional window-local route regroup.
Only loss/evaluation logging. Four microbatches per optimizer update.
"""
import os,json,math,random,shutil,argparse,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed,get_cuda_rng_tracker
from olmoe_megatron_fullft import make,hidden,memory_setup,SharedStateOffloadAdam
from optimizer import model_zero
R=Path(os.environ.get('OLMOE_QUALITY_ROOT','work'))

def ce_sum(model,tokens,mask,backward,scale=1.,per_sequence=False):
    x=torch.as_tensor(tokens[:,:2048].astype(np.int64),device='cuda')
    y=torch.as_tensor(tokens[:,1:].astype(np.int64),device='cuda').reshape(-1)
    active=torch.as_tensor(mask[:,1:],device='cuda',dtype=torch.bool).reshape(-1);y=y.masked_fill(~active,-100)
    h=hidden(model,x).reshape(-1,2048);loss=0.;seq_loss=torch.zeros(len(tokens),device='cuda')
    for i in range(0,len(y),128):
        def f(a,b):return F.cross_entropy(F.linear(a,model.output_layer.weight).float(),b,reduction='sum',ignore_index=-100)
        chunk_loss=checkpoint(f,h[i:i+128],y[i:i+128],use_reentrant=True) if backward else f(h[i:i+128],y[i:i+128])
        loss=loss+chunk_loss
        if per_sequence:seq_loss[i//2048]+=chunk_loss.detach()
    if backward:(loss*scale).backward()
    return seq_loss if per_sequence else loss.detach()

def evaluate(model,data,split,out,step,smoke=False):
    model.eval();tokens=np.load(data/f'{split}_tokens.npy',mmap_mode='r');mask=np.load(data/f'{split}_mask.npy',mmap_mode='r');rank=dist.get_rank();world=dist.get_world_size();records=[]
    total=torch.zeros(2,device='cuda',dtype=torch.float64)
    n=16 if smoke else len(tokens)
    assert n%(4*world)==0, 'Evaluation sequences must fill a global microbatch'
    with torch.no_grad():
        for start in range(0,n,4*world):
            ids=np.arange(start+rank*4,start+rank*4+4)
            v=ce_sum(model,tokens[ids],mask[ids],False,per_sequence=True)
            den=mask[ids,1:].sum(1)
            records.extend([dict(sequence_id=int(i),loss_sum=float(a),assistant_targets=int(b)) for i,a,b in zip(ids,v.cpu(),den)])
            total[0]+=v.double().sum();total[1]+=int(den.sum())
    gathered=[None]*world if rank==0 else None;dist.gather_object(records,gathered,dst=0)
    if rank==0:(out/f'{split}_per_sequence_{step:04d}.json').write_text(json.dumps([r for part in gathered for r in part]))
    dist.all_reduce(total);loss=float(total[0]/total[1]);assert math.isfinite(loss)
    row=dict(step=step,split=split,loss=loss,ppl=math.exp(loss),assistant_targets=int(total[1]),sequences=n)
    if rank==0:
        with (out/f'{split}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    model.train();return row

def regroup(model,bank,ids,out,window,epsilon):
    from sbac import regroup as plan_regroup
    model.eval();rank=dist.get_rank();captured=[];hooks=[]
    for layer in model.decoder.layers:
        def capture(mod,inp,result):
            route=result[1];captured.append(route.reshape(2048,4,64).sum(0).to(torch.int32))
        hooks.append(layer.mlp.router.register_forward_hook(capture))
    chunks=[]
    with torch.no_grad():
        for start in range(0,len(ids),16):
            local=ids[start+rank*4:start+(rank+1)*4];captured.clear()
            hidden(model,torch.tensor(bank[local,:2048].astype(np.int64),device='cuda'))
            assert len(captured)==16
            local_counts=torch.stack(captured,1).contiguous()
            gathered=[torch.empty_like(local_counts) for _ in range(4)];dist.all_gather(gathered,local_counts)
            if rank==0:chunks.append(torch.cat(gathered).cpu().numpy())
    for hook in hooks:hook.remove()
    permutation=np.arange(len(ids),dtype=np.int64)
    if rank==0:
        counts=np.concatenate(chunks);assert np.all(counts.sum(-1)==2048*8)
        # Native physical slots: experts 0..47 are NUMA0, 48..63 NUMA1.
        a=counts[:,:,48:].sum(-1).astype(np.float64);b=counts[:,:,:48].sum(-1).astype(np.float64)
        permutation,audit=plan_regroup(a,b,bank[ids,:2048],epsilon=epsilon)
        np.save(out/f'route_counts_w{window:03d}.npy',counts)
        (out/f'plan_audit_w{window:03d}.json').write_text(json.dumps(audit,indent=2))
    plan=torch.as_tensor(permutation,device='cuda');dist.broadcast(plan,0)
    model.train();return ids[plan.cpu().numpy()]

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['fifo','regroup'],required=True);p.add_argument('--updates',type=int,default=384);p.add_argument('--smoke',action='store_true');p.add_argument('--resume');p.add_argument('--mmd-epsilon',type=float,default=0.02);args=p.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(4);dist.init_process_group('nccl')
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,pipeline_model_parallel_size=1,expert_model_parallel_size=4)
    random.seed(1234);np.random.seed(1234);torch.manual_seed(1234);model_parallel_cuda_manual_seed(1234)
    data=R/'quality_data';meta=json.loads((data/'manifest.json').read_text());bank=np.load(data/'train_tokens.npy',mmap_mode='r');mask=np.load(data/'train_mask.npy',mmap_mode='r')
    out=R/'quality_runs'/('smoke_'+args.arm if args.smoke else args.arm);out.mkdir(parents=True,exist_ok=True)
    model,expert_order=make(R,torch.bfloat16,load_weights=not bool(args.resume));memory_setup(model);model.train();opt=SharedStateOffloadAdam(model,1e-6)
    start=0
    if args.resume:
        state=torch.load(Path(args.resume)/f'rank{rank}.pt',map_location='cpu',weights_only=False)
        assert state['data_manifest']==meta and state['arm']==args.arm
        assert state['mmd_epsilon']==args.mmd_epsilon
        with torch.no_grad():
            for n,param in opt.named:param.copy_(state['model'][n])
        for n,values in state['optimizer'].items():opt.state[n]=[v.cuda() if '.experts.' in n else v for v in values]
        opt.stepno=state['optimizer_step'];opt.lr=state['lr'];start=state['completed_steps']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng']);get_cuda_rng_tracker().set_states(state['parallel_rng']);del state
    evaluate(model,data,'validation',out,start,args.smoke)
    window_size=64 if args.smoke else 4096;per_window=window_size//64;plans={}
    for step in range(start,args.updates):
        epoch=step//384;win_in_epoch=(step%384)//per_window;window=step//per_window
        if window not in plans:
            sequence_order=np.arange(len(bank)) if epoch==0 else np.random.default_rng(1729+epoch).permutation(len(bank))
            ids=sequence_order[win_in_epoch*window_size:(win_in_epoch+1)*window_size]
            if args.arm=='regroup':ids=regroup(model,bank,ids,out,window,args.mmd_epsilon)
            plans[window]=ids
            if rank==0:np.save(out/f'schedule_w{window:03d}.npy',ids)
        ids=plans[window][(step%per_window)*64:(step%per_window+1)*64].reshape(4,4,4)
        den=int(mask[ids.ravel(),1:].sum());assert den>0
        model_zero(opt.named);loss_sum=torch.zeros((),device='cuda')
        for micro in range(4):
            local=ids[micro,rank];loss_sum+=ce_sum(model,bank[local],mask[local],True,4./den)
        norm=opt.step();dist.all_reduce(loss_sum);loss=float(loss_sum/den);assert math.isfinite(loss)
        if rank==0:
            row=dict(step=step+1,input_tokens=(step+1)*131072,loss=loss,assistant_targets=den)
            with (out/'train_loss.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if (step+1)%32==0 or step+1==args.updates:evaluate(model,data,'validation',out,step+1,args.smoke)
    if not args.smoke:
        evaluate(model,data,'test',out,args.updates)
        # Keep ranks participating in collectives while archival frees storage.
        ready=torch.zeros((),device='cuda',dtype=torch.int32)
        while True:
            if rank==0:ready.fill_(int(shutil.disk_usage(R).free>110*2**30))
            dist.broadcast(ready,0)
            if ready.item():break
            if rank==0:
                (out/'WAITING_SAVE_SPACE.json').write_text(json.dumps(dict(required_GiB=110,free_GiB=shutil.disk_usage(R).free/2**30)))
            time.sleep(30)
        if rank==0:(out/'WAITING_SAVE_SPACE.json').unlink(missing_ok=True)
        target=R/'final_states'/args.arm;target.mkdir(parents=True,exist_ok=True)
        extra=dict(mmd_epsilon=args.mmd_epsilon,completed_steps=args.updates,arm=args.arm,accumulation_phase=0,accumulation_steps=4,model_config=json.loads((R/'model/config.json').read_text()),optimizer_hparams=dict(betas=(.9,.95),eps=1e-8,weight_decay=.1,clip_grad=1.),data_manifest=meta,expert_order=expert_order,python_rng=random.getstate(),numpy_rng=np.random.get_state(),parallel_rng={k:v.cpu() for k,v in get_cuda_rng_tracker().get_states().items()},scheduler=dict(type='constant',lr=opt.lr,step=args.updates),schedule_state=dict(window_size=4096,completed_windows=args.updates//64,plans=plans),predictor_state='No learned predictor; reprofile current model at each new window',gradient_dtype='bf16')
        opt.save(target/f'rank{rank}.pt',extra)
        dist.barrier()
        if rank==0:
            shutil.copytree(out,target/'logs',dirs_exist_ok=True)
            shutil.copytree(data,target/'quality_data',dirs_exist_ok=True)
            shutil.copytree(R/'model',target/'model',ignore=shutil.ignore_patterns('*.safetensors','.cache'),dirs_exist_ok=True)
            (target/'data').mkdir(exist_ok=True);shutil.copy2(R/'data/native_placements.npz',target/'data/native_placements.npz')
            (target/'analysis').mkdir(exist_ok=True)
            for name in ['train_olmoe_quality.py','olmoe_megatron_fullft.py','optimizer.py']:
                shutil.copy2(Path(__file__).resolve().parent/name,target/'analysis'/name)
            shutil.copytree(Path(__file__).resolve().parents[1]/'sbac',target/'sbac',dirs_exist_ok=True)
            (target/'RESUME.txt').write_text('Set OLMOE_QUALITY_ROOT and PYTHONPATH to this directory. Run torchrun --standalone --nproc_per_node=4 analysis/train_olmoe_quality.py --arm '+args.arm+' --resume . --updates 768. Requires the recorded Megatron Core 0.15.3/PyTorch environment. Final state is at an optimizer-update boundary; no accumulated gradients are pending.')
            (target/'READY.json').write_text(json.dumps(dict(arm=args.arm,completed_steps=args.updates,full_state=True)))
    if rank==0:(out/'COMPLETED.json').write_text(json.dumps(dict(arm=args.arm,updates=args.updates,smoke=args.smoke)))
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
