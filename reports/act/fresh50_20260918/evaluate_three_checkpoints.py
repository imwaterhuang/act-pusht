import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys,json,hashlib,shutil,datetime,subprocess,platform,time
from pathlib import Path
import torch
from mini_wam.evaluation.act import generate_act_scene_split,load_act_checkpoint,evaluate_act_policy
repo=Path('/content/act-pusht-review')
run=Path('/content/drive/MyDrive/act-pusht-runs/act-seed0-20260917-l4')
out=Path('/content/act-review-20260918-50')
mirror=run.parent/'review-20260918-50'
assert not out.exists() and not mirror.exists(), 'Refuse duplicate review output'
assert json.loads((run/'completed.json').read_text())['step']==40000
assert torch.cuda.is_available()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
selection=json.loads((run/'selection.json').read_text())
selection_sha=sha(run/'selection.json')
records={r['step']:r for r in selection['candidates']}
steps=[30000,35000,40000]
out.mkdir()
(out/'weights').mkdir()
checkpoints={}
for step in steps:
    src=run/records[step]['checkpoint']
    target=out/'weights'/src.name
    shutil.copy2(src,target)
    assert sha(target)==records[step]['checkpoint_sha256']
    checkpoints[step]=target
excluded=[repo/'splits/act_development_scenes.json']
if (repo/'splits/dev_scenes.json').exists(): excluded.append(repo/'splits/dev_scenes.json')
for p in (run.parent/'validation-20260917-154249/interface').rglob('*scenes*.json'):
    if isinstance(json.loads(p.read_text()),dict) and 'scenes' in json.loads(p.read_text()): excluded.append(p)
scene_path=out/'scenes.json'
scenes=generate_act_scene_split(scene_path,split='review',generation_seed=2026091801,count=50,excluded_paths=tuple(excluded))
new_seeds={s['environment_seed'] for s in scenes['scenes']}
new_states={tuple(sorted(s['state'].items())) for s in scenes['scenes']}
for p in excluded:
    old=json.loads(p.read_text())['scenes']
    assert new_seeds.isdisjoint(s['environment_seed'] for s in old)
    assert new_states.isdisjoint(tuple(sorted(s['state'].items())) for s in old)
mirror.mkdir()
shutil.copy2(scene_path,mirror/'scenes.json')
shutil.copy2(__file__,mirror/'evaluate_three_checkpoints.py')
manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'python':platform.python_version(),'training_seed':0,'steps':steps,'scene_count':50,'generation_seed':2026091801,'scenes_hash':scenes['scenes_hash'],'excluded_scene_files':[str(p) for p in excluded],'disjoint_seeds_and_states':True,'original_best_step':selection['best_step'],'original_selection_sha256':selection_sha,'protocol':{'max_steps':300,'execute_steps':4,'success':'coverage > 0.95','inference_latent':'z=0'},'checkpoint_sha256':{str(s):sha(checkpoints[s]) for s in steps},'purpose':'User-requested paired comparison on 50 fresh scenes; original development selection is unchanged.'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
shutil.copy2(out/'manifest.json',mirror/'manifest.json')
print('REVIEW_SCENES_FROZEN',json.dumps(manifest),flush=True)
rows=[]
for step in steps:
    started=time.monotonic()
    model,norm,payload=load_act_checkpoint(checkpoints[step],repo/'artifacts/act/data_audit.json',torch.device('cuda'))
    assert payload['step']==step
    print('REVIEW_STEP_STARTED',step,flush=True)
    result=evaluate_act_policy(model,norm,scene_path,out/f'step_{step}',device=torch.device('cuda'),expected_split='review',max_steps=300,execute_steps=4)
    assert len(result['episodes'])==50 and result['scenes_hash']==scenes['scenes_hash']
    row={'step':step,'checkpoint_sha256':sha(checkpoints[step]),'seconds':time.monotonic()-started,**result['summary']}
    rows.append(row)
    shutil.copytree(out/f'step_{step}',mirror/f'step_{step}')
    (out/'summary.json').write_text(json.dumps({'manifest':manifest,'results':rows,'complete':len(rows)==3},indent=2))
    shutil.copy2(out/'summary.json',mirror/'summary.json')
    assert sha(out/f'step_{step}/results.json')==sha(mirror/f'step_{step}/results.json')
    print('REVIEW_STEP_COMPLETE',json.dumps(row),flush=True)
    del model,norm,payload,result
    torch.cuda.empty_cache()
assert sha(run/'selection.json')==selection_sha
print('ACT_FRESH50_COMPARISON_COMPLETE',json.dumps(rows),flush=True)
