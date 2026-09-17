"""Measure ACT throughput, verify full-state recovery, and run a 64-window diagnostic."""
from __future__ import annotations
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.utils.data import DataLoader, default_collate
from mini_wam.models.act import ActionPolicy, act_loss
from mini_wam.training.act import train_act, train_act_step
from mini_wam.training.act_config import load_act_config
from mini_wam.training.act_data import build_act_training_data
from mini_wam.training.artifacts import sha256_file, sync_run_directory
from mini_wam.training.reproducibility import DeterministicBatchSampler, seed_everything

FIELDS = ('observation_image', 'agent_position', 'action_chunk', 'action_valid_mask')

def emit(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value), flush=True)

def exact(a, b, path='root'):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a: exact(a[key], b[key], f'{path}.{key}')
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)): exact(x, y, f'{path}[{i}]')
    else:
        assert a == b, path

def make_model(config):
    return ActionPolicy(**{k:v for k,v in config['model'].items() if k != 'name'}).cuda()

def profile(config, dataset, out):
    records=[]
    for batch_size, workers in ((32,0),(32,2),(32,4),(64,0),(64,2),(64,4)):
        seed_everything(0)
        sampler=DeterministicBatchSampler(len(dataset),batch_size,0)
        loader=DataLoader(dataset,batch_sampler=sampler,num_workers=workers,
            generator=torch.Generator().manual_seed(100000),persistent_workers=workers>0)
        batches=iter(loader)
        expected=default_collate([dataset[i] for i in next(iter(DeterministicBatchSampler(len(dataset),batch_size,0)))])
        first=next(batches)
        for field in FIELDS: exact(expected[field],first[field],field)
        sampler.mark_consumed()
        model=make_model(config)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        timing=[]
        torch.cuda.reset_peak_memory_stats()
        for i in range(15):
            torch.cuda.synchronize(); t0=time.perf_counter()
            batch=next(batches); t1=time.perf_counter()
            batch={k:v.cuda() for k,v in batch.items() if k in FIELDS}
            torch.cuda.synchronize(); t2=time.perf_counter()
            train_act_step(model=model,batch=batch,optimizer=optimizer,scheduler=None,
                sampler=sampler,device=torch.device('cuda'),config=config)
            torch.cuda.synchronize(); t3=time.perf_counter()
            if i>=3: timing.append((t1-t0,t2-t1,t3-t2,t3-t0))
        record={'batch_size':batch_size,'num_workers':workers,'used_fields_exact':True,
            **{name:statistics.mean(row[j] for row in timing) for j,name in enumerate(
                ('data_seconds','transfer_seconds','compute_seconds','end_to_end_seconds'))},
            'peak_memory_bytes':torch.cuda.max_memory_allocated()}
        record['samples_per_second']=batch_size/record['end_to_end_seconds']
        records.append(record); print('PROFILE',json.dumps(record),flush=True)
        del batches,loader,sampler,model,optimizer,batch,first,expected
        gc.collect();torch.cuda.empty_cache()
    selected=max(records,key=lambda r:r['samples_per_second'])
    result={'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),
            'python':sys.version,'cpu_count':os.cpu_count(),'records':records,'selected':selected}
    emit(out/'profile.json',result)
    return selected

def smoke(config, workers, out, mirror):
    # Fixed deterministic backend isolates restore correctness from convolution noise.
    import yaml
    config=copy.deepcopy(config);config['training']['batch_size']=8
    path=out/'smoke.yaml';path.write_text(yaml.safe_dump(config))
    continuous=out/'smoke-continuous'; split=out/'smoke-resumed'
    train_act(path,run_dir=continuous,device_name='cuda',num_workers=workers)
    train_act(path,run_dir=split,device_name='cuda',num_workers=workers,
        stop_after_step=2,mirror_dir=mirror/'smoke-resumed')
    # Recover from the durable Drive copy into a fresh local run directory.
    import shutil
    recovered=out/'smoke-recovered'
    shutil.copytree(mirror/'smoke-resumed',recovered)
    train_act(path,resume_path=recovered/'checkpoints/last.pt',device_name='cuda',
        num_workers=workers,mirror_dir=mirror/'smoke-recovered')
    a=torch.load(continuous/'checkpoints/last.pt',weights_only=True,map_location='cpu')
    b=torch.load(recovered/'checkpoints/last.pt',weights_only=True,map_location='cpu')
    for key in ('model','optimizer','scheduler','sampler','random_states','step','config','metadata'):
        exact(a[key],b[key],key)
    mirrored=mirror/'smoke-recovered/checkpoints/last.pt'
    assert sha256_file(mirrored)==sha256_file(recovered/'checkpoints/last.pt')
    assert json.loads((mirror/'smoke-recovered/completed.json').read_text())['step']==5
    emit(out/'smoke_result.json',{'passed':True,'step':5,'resume_exact':True,
        'drive_restore_exact':True,'drive_checkpoint_sha256':sha256_file(mirrored),
        'num_workers':workers})
    print('ACT_GPU_SMOKE_RESUME_DRIVE_VERIFIED',flush=True)

def overfit(config,dataset,out):
    seed_everything(0)
    indices=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(42))[:64].tolist()
    samples=default_collate([dataset[i] for i in indices])
    samples={k:v.cuda() for k,v in samples.items() if k in FIELDS}
    model=make_model(config);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    values=[];started=time.perf_counter()
    for step in range(300):
        model.train();optimizer.zero_grad(set_to_none=True)
        batch={k:v[32*(step%2):32*(step%2+1)] for k,v in samples.items()}
        result=model(batch['observation_image'],batch['agent_position'],batch['action_chunk'],batch['action_valid_mask'])
        losses=act_loss(result['actions'],batch['action_chunk'],batch['action_valid_mask'],result['mu'],result['logvar'],beta=0.)
        assert torch.isfinite(losses['loss'])
        losses['loss'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        values.append(float(losses['action_l1'].detach()))
        if (step+1)%50==0: print('OVERFIT',step+1,statistics.mean(values[-20:]),flush=True)
    start=statistics.mean(values[:20]);end=statistics.mean(values[-20:])
    passed=end<0.5*start
    emit(out/'overfit_result.json',{'passed':passed,'windows':64,'indices':indices,'steps':300,
        'beta':0.,'diagnostic_only':True,'initial_action_l1_20step_mean':start,
        'final_action_l1_20step_mean':end,'seconds':time.perf_counter()-started,'losses':values})
    assert passed, '64-window reconstruction did not decrease sufficiently'
    print('ACT_OVERFIT_VERIFIED',flush=True)
    del model,optimizer,samples;gc.collect();torch.cuda.empty_cache()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mirror',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();mirror=args.mirror.resolve()
    if out.exists() and any(out.iterdir()): raise ValueError('Use a new validation directory')
    out.mkdir(parents=True,exist_ok=True);mirror.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available() and any(x in torch.cuda.get_device_name() for x in ('L4','T4'))
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    config=load_act_config(ROOT/'configs/act_smoke.yaml')
    data=build_act_training_data(config)
    assert data.manifest['episode_count']==206 and len(data.dataset)==25444
    selected=profile(config,data.dataset,out)
    sync_run_directory(out,mirror)
    smoke(config,selected['num_workers'],out,mirror)
    sync_run_directory(out,mirror)
    overfit(config,data.dataset,out)
    emit(out/'validation_complete.json',{'passed':True,'gpu':torch.cuda.get_device_name(),'selected':selected})
    sync_run_directory(out,mirror)
    print('ACT_COLAB_VALIDATION_COMPLETE',flush=True)

if __name__=='__main__':main()
