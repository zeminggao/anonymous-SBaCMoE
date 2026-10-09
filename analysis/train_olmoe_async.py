"""PPL-only OLMoE training with causal asynchronous planning; no forced routes."""
import argparse,json,os,random,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from olmoe_megatron_fullft import make,memory_setup,SharedStateOffloadAdam
from optimizer import model_zero
from train_olmoe_quality import ce_sum,evaluate
from sbac.async_planner import PlannerClient,RouteFeedback

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=['fifo','regroup'],required=True)
    p.add_argument('--window',type=int,choices=[512,1024,2048,4096],default=4096)
    p.add_argument('--updates',type=int,default=384)
    p.add_argument('--eval-every',type=int,default=200)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--planner-cpu',type=int)
    p.add_argument('--run-name',required=True)
    args=p.parse_args();assert args.updates>0 and args.eval_every>0
    root=Path(os.environ.get('OLMOE_QUALITY_ROOT','work'));data=root/'quality_data'
    out=root/'quality_runs'/args.run_name
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(4)
    dist.init_process_group('nccl');assert dist.get_world_size()==4
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,pipeline_model_parallel_size=1,expert_model_parallel_size=4)
    random.seed(1234);np.random.seed(1234);torch.manual_seed(1234);model_parallel_cuda_manual_seed(1234)
    bank=np.load(data/'train_tokens.npy',mmap_mode='r');mask=np.load(data/'train_mask.npy',mmap_mode='r')
    total=args.updates*64;assert len(bank)>=total and bank.shape[1]==2049
    assert not (out/'train_loss.jsonl').exists(),'Choose a new run name'
    out.mkdir(parents=True,exist_ok=True)
    model,order=make(root,torch.bfloat16);memory_setup(model)
    opt=SharedStateOffloadAdam(model,args.lr);client=feedback=None
    if args.arm=='regroup':
        feedback=RouteFeedback(model,out,rank)
        if rank==0:client=PlannerClient(data/'train_tokens.npy',out,args.window,total,args.planner_cpu)
    if rank==0:
        config=vars(args)|dict(seed=1234,microbatch_per_rank=4,accumulation_steps=4,sequence_length=2048,expert_order=order.tolist(),checkpoint_saving=False,data_manifest=json.loads((data/'manifest.json').read_text()),model_config=json.loads((root/'model/config.json').read_text()))
        (out/'config.json').write_text(json.dumps(config,indent=2))
    seen=[];current=-1;per_window=args.window//64
    try:
        evaluate(model,data,'validation',out,0)
        for step in range(args.updates):
            started=time.perf_counter();w=step//per_window
            if w!=current:
                ids=np.arange(w*args.window,min((w+1)*args.window,total))
                if args.arm=='regroup':
                    if rank==0:ids=client.take(w)
                    tensor=torch.as_tensor(ids,device='cuda');dist.broadcast(tensor,0);ids=tensor.cpu().numpy()
                    if rank==0:client.request(w+1)
                plan=ids;current=w
                if rank==0:np.save(out/f'schedule_w{w:06d}.npy',plan)
            ids=plan[(step%per_window)*64:(step%per_window+1)*64].reshape(4,4,4)
            seen.extend(ids.ravel().tolist());den=int(mask[ids.ravel(),1:].sum());assert den>0
            model.train();model_zero(opt.named);loss_sum=torch.zeros((),device='cuda')
            for micro in range(4):
                local=ids[micro,rank]
                if feedback is not None and micro==0:feedback.begin(local,w)
                loss_sum+=ce_sum(model,bank[local],mask[local],True,4./den)
                if feedback is not None and micro==0:feedback.end()
            opt.step();dist.all_reduce(loss_sum);loss=float(loss_sum/den);assert np.isfinite(loss)
            if feedback is not None and ((step+1)%per_window==0 or step+1==args.updates):feedback.finish_window(w)
            if rank==0:
                row=dict(step=step+1,loss=loss,step_seconds=time.perf_counter()-started,planner_choice=client.metrics[-1]['choice'] if client else 'fifo')
                with open(out/'train_loss.jsonl','a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
            if (step+1)%args.eval_every==0 or step+1==args.updates:evaluate(model,data,'validation',out,step+1)
        assert sorted(seen)==list(range(total))
        evaluate(model,data,'test',out,args.updates)
        if rank==0:(out/'COMPLETED.json').write_text(json.dumps(dict(updates=args.updates,sample_conservation=True)))
    finally:
        if feedback is not None:feedback.close()
        if client is not None:client.close()
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
