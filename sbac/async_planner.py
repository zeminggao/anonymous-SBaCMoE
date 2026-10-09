"""Nonblocking one-window-ahead planner and bounded CUDA route feedback.

Training observes physical expert IDs (Megatron's reordered router), so owners
are contiguous blocks of 64 / EP. No current-window model forward is used to plan.
"""
import atexit,json,os,queue,subprocess,sys,threading,time
from pathlib import Path

def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value));os.replace(tmp,path)

class PlannerClient:
    def __init__(self,bank,out,window,total,cpu=None,topology=None):
        self.root=Path(out)/'async_planner';self.root.mkdir(parents=True,exist_ok=True)
        self.window=window;self.total=total
        topology=topology or dict(world_size=16,rank_to_bottleneck_side=[0]*8+[1]*8,inverse_bandwidth_weights=[1.,1.])
        atomic_json(self.root/'topology.json',topology)
        self.log=open(self.root/'worker.log','w')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',SBAC_TRAIN_PID=str(os.getpid()))
        if cpu is not None:env['SBAC_WORKER_CPU']=str(cpu)
        self.process=subprocess.Popen([sys.executable,'-m','sbac.async_planner','worker',str(bank),str(self.root)],env=env,stdout=self.log,stderr=subprocess.STDOUT)
        self.metrics=[];self.closed=False;atexit.register(self.close)
    def request(self,w):
        first=w*self.window;end=min(first+self.window,self.total)
        if first>=end:return
        atomic_json(self.root/f'request_{w:06d}.json',dict(window=w,first=first,end=end,feedback_through=w-2))
    def take(self,w):
        import numpy as np
        start=time.perf_counter();ids=np.arange(w*self.window,min((w+1)*self.window,self.total))
        response=self.root/f'response_{w:06d}.json';reason='not_ready';meta={}
        if response.exists():
            meta=json.loads(response.read_text());reason=meta['status']
            if reason=='ready':
                candidate=np.load(self.root/f'plan_{w:06d}.npy')
                if not np.array_equal(np.sort(candidate),ids):
                    reason='invalid_plan_fifo';meta['error']='Invalid asynchronous sample permutation'
                else:ids=candidate
        elif self.process.poll() is not None:reason='worker_exited'
        # A late response may never overwrite an already consumed FIFO choice.
        atomic_json(self.root/f'consumed_{w:06d}.json',dict(window=w,choice=reason))
        self.metrics.append(dict(window=w,choice=reason,decision_seconds=time.perf_counter()-start,worker=meta))
        atomic_json(self.root/'client_metrics.json',self.metrics)
        return ids
    def close(self):
        if self.closed:return
        self.closed=True
        atomic_json(self.root/'STOP.json',{})
        try:self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:self.process.terminate();self.process.wait(timeout=5)
        self.log.close()

class RouteFeedback:
    """Pinned-buffer pool bounds memory; a full pool skips observations, never data."""
    def __init__(self,model,out,rank):
        import torch
        self.torch=torch;self.root=Path(out)/'async_planner';self.root.mkdir(exist_ok=True)
        self.rank=rank;self.stream=torch.cuda.Stream();self.free=queue.Queue();self.jobs=queue.Queue()
        for _ in range(2):self.free.put(torch.empty((16,2048,4,8),dtype=torch.uint8,pin_memory=True))
        self.active=False;self.buffer=None;self.seen=set();self.serial=0;self.dropped=0;self.error=None;self.paths={};self.metrics={};self.host_seconds={}
        self.handles=[layer.mlp.router.register_forward_hook(self.hook(l)) for l,layer in enumerate(model.decoder.layers)]
        self.thread=threading.Thread(target=self.writer,daemon=True);self.thread.start()
    def begin(self,ids,w):
        if self.error:raise RuntimeError(self.error)
        self.active=False
        try:self.buffer=self.free.get_nowait()
        except queue.Empty:self.dropped+=1;return
        self.ids=ids.copy();self.w=w;self.seen=set();self.active=True
    def hook(self,l):
        def capture(module,inputs,result):
            if not self.active or l in self.seen:return
            started=time.perf_counter()
            self.seen.add(l);torch=self.torch
            # Capture the dispatch map itself, not router logits or guessed IDs.
            route=result[1];assert route.shape==(8192,64)
            ids=route.to(torch.uint8).topk(8,dim=-1).indices.to(torch.uint8).reshape(2048,4,8)
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                self.buffer[l].copy_(ids,non_blocking=True);ids.record_stream(self.stream)
            self.host_seconds[self.w]=self.host_seconds.get(self.w,0.)+time.perf_counter()-started
        return capture
    def end(self):
        if not self.active:return
        self.active=False
        if len(self.seen)!=16:raise RuntimeError('Incomplete router feedback')
        event=self.torch.cuda.Event();event.record(self.stream)
        self.jobs.put(('data',self.w,self.ids,self.buffer,event,self.serial));self.serial+=1;self.buffer=None
    def finish_window(self,w):self.jobs.put(('done',w,self.dropped,self.host_seconds.pop(w,0.)))
    def writer(self):
        import numpy as np
        try:
            while True:
                job=self.jobs.get()
                if job[0]=='stop':break
                if job[0]=='done':
                    _,w,dropped,host_seconds=job
                    atomic_json(self.root/f'feedback_w{w:06d}_rank{self.rank}.json',dict(files=self.paths.pop(w,[]),dropped_observations_total=dropped,host_enqueue_seconds=host_seconds,writer_seconds=self.metrics.pop(w,0.)))
                    continue
                _,w,ids,buf,event,serial=job;start=time.perf_counter();event.synchronize()
                file=self.root/f'feedback_w{w:06d}_rank{self.rank}_{serial:06d}.npz'
                # Written in chronological job order; done marker follows all files.
                np.savez(file,ids=ids,routes=buf.numpy().transpose(2,0,1,3))
                self.paths.setdefault(w,[]).append(file.name);self.metrics[w]=self.metrics.get(w,0.)+time.perf_counter()-start;self.free.put(buf)
        except Exception as e:self.error=repr(e);atomic_json(self.root/f'feedback_error_rank{self.rank}.json',dict(error=self.error))
    def close(self):
        self.jobs.put(('stop',));self.thread.join()
        for h in self.handles:h.remove()
        if self.error:raise RuntimeError(self.error)

def worker(bank_path,root_path):
    import numpy as np
    from sbac.predictor import TokenMarginalPredictor
    from sbac.predictive_search import search
    root=Path(root_path);bank=np.load(bank_path,mmap_mode='r')
    if 'SBAC_WORKER_CPU' in os.environ:os.sched_setaffinity(0,{int(os.environ['SBAC_WORKER_CPU'])})
    topology=json.loads((root/'topology.json').read_text());world=topology['world_size'];assert 64%world==0
    p=TokenMarginalPredictor();own=np.tile(np.repeat(np.arange(world),64//world),(16,1));last=-1;handled=set()
    while not (root/'STOP.json').exists():
        if os.name=='posix' and 'SBAC_TRAIN_PID' in os.environ:
            try:os.kill(int(os.environ['SBAC_TRAIN_PID']),0)
            except ProcessLookupError:break
        requests=sorted(root.glob('request_*.json'))
        pending=[f for f in requests if f.name not in handled]
        if not pending:time.sleep(.01);continue
        path=pending[0];handled.add(path.name);q=json.loads(path.read_text());w=q['window'];start=time.perf_counter()
        if (root/f'consumed_{w:06d}.json').exists():continue
        try:
            update_s=0.;feedback_rows=0
            for fw in range(last+1,q['feedback_through']+1):
                markers=[root/f'feedback_w{fw:06d}_rank{r}.json' for r in range(world)]
                if not all(f.exists() for f in markers):break
                ts=time.perf_counter();files=[root/name for f in markers for name in json.loads(f.read_text())['files']]
                if files:
                    records=[dict(np.load(f)) for f in files];ids=np.concatenate([x['ids'] for x in records]);routes=np.concatenate([x['routes'] for x in records]);order=np.argsort(ids)
                    p.update(np.array(bank[ids[order],:2048]),routes[order],ids[order],q['first']);feedback_rows+=len(ids)
                last=fw;update_s+=time.perf_counter()-ts
            if p.tables is None:
                meta=dict(status='cold_fifo',window=w)
            else:
                ts=time.perf_counter();tokens=np.array(bank[q['first']:q['end'],:2048]);read_s=time.perf_counter()-ts
                counts,pred=p.predict(tokens,q['first']);plan,stats=search(tokens,counts,own,1729,root,f'work_{w:06d}',rank_nodes=topology['rank_to_bottleneck_side'],inverse_bandwidth=topology['inverse_bandwidth_weights'])
                dest=root/f'plan_{w:06d}.npy';tmp=root/f'plan_{w:06d}.tmp.npy';np.save(tmp,plan+q['first']);os.replace(tmp,dest)
                meta=dict(status='ready',window=w,prediction=pred,search=stats,read_seconds=read_s)
            meta.update(feedback_through=last,feedback_sequences=feedback_rows,update_seconds=update_s,worker_seconds=time.perf_counter()-start)
            atomic_json(root/f'response_{w:06d}.json',meta)
        except Exception as e:atomic_json(root/f'response_{w:06d}.json',dict(status='failed_fifo',error=repr(e),window=w))

if __name__=='__main__':
    assert sys.argv[1]=='worker';worker(sys.argv[2],sys.argv[3])
