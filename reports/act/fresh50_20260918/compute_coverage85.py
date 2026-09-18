import json,csv,shutil,zipfile,hashlib,math
from pathlib import Path
out=Path('/content/act-review-20260918-50')
mirror=Path('/content/drive/MyDrive/act-pusht-runs/review-20260918-50')
assert review_process.poll()==0
report=json.loads((mirror/'summary.json').read_text())
assert report['complete'] and [r['step'] for r in report['results']]==[30000,35000,40000]
episodes={s:json.loads((mirror/f'step_{s}/results.json').read_text())['episodes'] for s in [30000,35000,40000]}
ids=[[e['scene_id'] for e in episodes[s]] for s in episodes]
assert ids[0]==ids[1]==ids[2] and len(set(ids[0]))==50
paired=[]
for i,scene_id in enumerate(ids[0]):
    row={'scene_id':scene_id,'environment_seed':episodes[30000][i]['environment_seed']}
    for step in episodes:
        e=episodes[step][i]
        assert e['environment_seed']==row['environment_seed']
        row[f'success_{step}']=e['is_success']
        row[f'coverage_{step}']=e['final_coverage']
        row[f'steps_{step}']=e['steps']
    paired.append(row)
with (mirror/'paired_scenes.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
comparisons=[]
for a,b in [(30000,35000),(30000,40000),(35000,40000)]:
    only_a=sum(r[f'success_{a}'] and not r[f'success_{b}'] for r in paired)
    only_b=sum(r[f'success_{b}'] and not r[f'success_{a}'] for r in paired)
    n=only_a+only_b
    p=min(1.0,2*sum(math.comb(n,k) for k in range(min(only_a,only_b)+1))/2**n) if n else 1.0
    comparisons.append({'a':a,'b':b,'only_a_success':only_a,'only_b_success':only_b,'mcnemar_exact_two_sided_p':p})
report['paired_comparisons']=comparisons
(mirror/'summary.json').write_text(json.dumps(report,indent=2))
lines=['# ACT fresh 50-scene checkpoint comparison','',
'One training seed; the same 50 fresh random scenes for each checkpoint. Maximum 300 environment steps, execute 4 actions before reobserving, strict success: coverage > 0.95.',
'','| Training step | Success | Success rate | Mean final coverage | Mean max coverage |','|---:|---:|---:|---:|---:|']
for r in report['results']:
    lines.append(f"| {r['step']} | {r['success_count']}/50 | {r['success_rate']:.1%} | {r['mean_final_coverage']:.4f} | {r['mean_max_coverage']:.4f} |")
lines+=['','Scene generation seed: 2026091801. The manifest records disjoint seeds/states, source revision, checkpoint hashes and the scene hash.','',
'Original development-selected best.pt and selection.json were preserved. This requested comparison does not establish a statistically reliable training-step advantage or robustness across training seeds.']
(mirror/'REPORT.md').write_text('\n'.join(lines)+'\n')
shutil.copy2(review_log,mirror/'execution.log')
history=get_ipython().history_manager.input_hist_raw
cells=[{'cell_type':'markdown','metadata':{},'source':['# ACT fresh-scene comparison execution backup\n','Completed evaluation. Outputs are in this Drive folder. Do not run all again: the runner rejects existing result directories.']}]
for source in history[2:6]:
    cells.append({'cell_type':'code','metadata':{},'source':source.splitlines(keepends=True),'execution_count':None,'outputs':[]})
(mirror/'ACT_Fresh50_Comparison.ipynb').write_text(json.dumps({'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'}},'cells':cells},indent=2))
files={str(p.relative_to(mirror)):hashlib.sha256(p.read_bytes()).hexdigest() for p in mirror.rglob('*') if p.is_file()}
(mirror/'checksums.json').write_text(json.dumps(files,indent=2))
archive=Path('/content/act_fresh50_20260918.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in mirror.rglob('*'):
        if p.is_file(): z.write(p,arcname=str(p.relative_to(mirror)))
print('REPORT_JSON',json.dumps(report))
print('EVIDENCE_ARCHIVE',archive.stat().st_size,hashlib.sha256(archive.read_bytes()).hexdigest())
print('DRIVE_EVIDENCE_VERIFIED',str(mirror))
from google.colab import files


import json,csv,hashlib,zipfile
from pathlib import Path
mirror=Path('/content/drive/MyDrive/act-pusht-runs/review-20260918-50')
report=json.loads((mirror/'summary.json').read_text())
assert report['complete']
report['coverage85_definition']={'threshold':0.85,'comparison':'>','final':'coverage at episode end','ever':'maximum coverage over executed episode','note':'Includes strict >95% successes; additional descriptive metric only.'}
for row in report['results']:
    episodes=json.loads((mirror/f"step_{row['step']}/results.json").read_text())['episodes']
    row['final_coverage_gt_85_count']=sum(e['final_coverage']>0.85 for e in episodes)
    row['max_coverage_gt_85_count']=sum(e['max_coverage']>0.85 for e in episodes)
    row['final_coverage_gt_85_rate']=row['final_coverage_gt_85_count']/50
    row['max_coverage_gt_85_rate']=row['max_coverage_gt_85_count']/50
    row['ever_gt85_but_final_le85_count']=sum(e['max_coverage']>0.85 and e['final_coverage']<=0.85 for e in episodes)
(mirror/'summary.json').write_text(json.dumps(report,indent=2))
with (mirror/'coverage85.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=['step','episode_count','success_count','final_coverage_gt_85_count','max_coverage_gt_85_count','ever_gt85_but_final_le85_count','mean_final_coverage'],extrasaction='ignore');w.writeheader();w.writerows(report['results'])
lines=['','## Additional coverage threshold','',
'Counts use strictly > 0.85 and include >0.95 successes. Final and maximum coverage come from the original saved trajectories; no episodes were rerun.',
'','| Step | Final >85% | Ever >85% | Ever >85%, final <=85% |','|---:|---:|---:|---:|']
for r in report['results']:
    lines.append(f"| {r['step']} | {r['final_coverage_gt_85_count']}/50 | {r['max_coverage_gt_85_count']}/50 | {r['ever_gt85_but_final_le85_count']}/50 |")
with (mirror/'REPORT.md').open('a') as f: f.write('\n'.join(lines)+'\n')
(mirror/'compute_coverage85.py').write_text(get_ipython().history_manager.input_hist_raw[-1])
checks={str(p.relative_to(mirror)):hashlib.sha256(p.read_bytes()).hexdigest() for p in mirror.rglob('*') if p.is_file() and p.name!='checksums.json'}
(mirror/'checksums.json').write_text(json.dumps(checks,indent=2))
archive=Path('/content/act_fresh50_20260918_coverage85.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in mirror.rglob('*'):
        if p.is_file():z.write(p,arcname=str(p.relative_to(mirror)))
print('FINAL_REVIEW_REPORT',json.dumps(report))
print('FINAL_ARCHIVE_SHA256',hashlib.sha256(archive.read_bytes()).hexdigest())
from google.colab import files
files.download(str(archive))