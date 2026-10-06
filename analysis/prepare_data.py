"""Deduplicated dialogue splits, native chat template, assistant-only targets."""
import json,hashlib,os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
R=Path(os.environ.get('OLMOE_QUALITY_ROOT','work'));OUT=R/'quality_data';OUT.mkdir(exist_ok=True)
tok=AutoTokenizer.from_pretrained(R/'model',local_files_only=True)
rows=[];seen=set()
for path in sorted((R/'tulu_raw').glob('*.parquet')):
    table=pq.read_table(path,columns=['messages'])
    for i,msgs in enumerate(table['messages'].to_pylist()):
        prompt=next((m['content'] for m in msgs if m['role']=='user'),'')
        key=hashlib.sha256(' '.join(prompt.split()).encode()).hexdigest()
        if not prompt or key in seen or not any(m['role']=='assistant' for m in msgs):continue
        seen.add(key);rows.append((msgs,key,path.name,i))
print('unique_dialogues',len(rows),flush=True)
rng=np.random.default_rng(1729);rng.shuffle(rows)
groups={'validation':[],'test':[],'train':[]}
for row in rows:
    bucket=int(row[1][:8],16)%100
    groups['validation' if bucket==0 else 'test' if bucket==1 else 'train'].append(row)
manifest=dict(dataset='allenai/tulu-v3.1-mix-preview-4096-OLMoE',revision='a16e10fdbec0b3430b4cc1e1716bec5ec39d08ce',seed=1729,loss='assistant content plus its closing tokens; user/system/template assistant headers masked',sequence_length=2048,split_rule='normalized first-user prompt SHA256 modulo 100, 0 validation, 1 test, remainder train',splits={})
for split,n in [('validation',512),('test',512),('train',24576)]:
    need=n*2048+1;tokens=[];masks=[];used=[]
    for messages,key,shard,row in groups[split]:
        text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
        spans=[]
        for j,m in enumerate(messages):
            if m['role']!='assistant':continue
            prefix=tok.apply_chat_template(messages[:j],tokenize=False,add_generation_prompt=True)
            after=tok.apply_chat_template(messages[:j+1],tokenize=False,add_generation_prompt=False)
            assert text.startswith(prefix) and text.startswith(after)
            spans.append((len(prefix),len(after)))
        enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
        mask=[int(any(end>a and start<b for a,b in spans)) for start,end in enc['offset_mapping']]
        if not any(mask):continue
        tokens.extend(enc['input_ids']);masks.extend(mask);used.append(dict(prompt_sha256=key,shard=shard,row=row))
        if len(tokens)>=need:break
    assert len(tokens)>=need,(split,len(tokens),need)
    t=np.array(tokens[:need],dtype=np.int32);m=np.array(masks[:need],dtype=np.uint8)
    indices=np.arange(n)[:,None]*2048+np.arange(2049)[None,:]
    packed=t[indices];packed_mask=m[indices]
    if split=='train':
        permutation=np.random.default_rng(1729).permutation(n)
        packed=packed[permutation];packed_mask=packed_mask[permutation];np.save(OUT/'global_permutation.npy',permutation)
    assert packed_mask[:,1:].sum()>0
    np.save(OUT/f'{split}_tokens.npy',packed);np.save(OUT/f'{split}_mask.npy',packed_mask)
    (OUT/f'{split}_provenance.json').write_text(json.dumps(used))
    manifest['splits'][split]=dict(sequences=n,input_tokens=n*2048,assistant_targets=int(packed_mask[:,1:].sum()),conversations=len(used),discarded_tail=len(tokens)-need)
    print(split,manifest['splits'][split],flush=True)
manifest['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.npy')}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('DATA_READY',flush=True)
