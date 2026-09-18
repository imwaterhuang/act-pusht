"""Recompute the release metrics and replay selected recorded actions; no inference."""
from __future__ import annotations
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import imageio.v3 as iio
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mini_wam.evaluation.act import _wilson_95
from mini_wam.studio.pusht import SceneState, make_pusht_env

REVIEW = ROOT / 'reports/act/fresh50_20260918'
OUT = ROOT / 'reports/act/release87'
MEDIA = ROOT / 'docs/media'
THRESHOLD = 0.87
EXAMPLES = {
    'act-review-000': 'success',
    'act-review-014': 'success',
    'act-review-038': 'success',
    'act-review-003': 'failure',
    'act-review-004': 'failure',
}


def rescore(episodes, threshold=THRESHOLD):
    rows = []
    for e in episodes:
        trace = e['trace']
        assert trace and len(trace) == e['steps']
        maximum = max(t['coverage'] for t in trace)
        assert abs(maximum-e['max_coverage']) < 1e-10
        assert abs(trace[-1]['coverage']-e['final_coverage']) < 1e-10
        crossing = next((t for t in trace if t['coverage'] > threshold), None)
        rows.append(dict(scene_id=e['scene_id'], environment_seed=e['environment_seed'],
                         success=crossing is not None,
                         first_success_step=crossing['step'] if crossing else None,
                         original_final_coverage=e['final_coverage'],
                         original_max_coverage=maximum,
                         release_stop_coverage=(crossing or trace[-1])['coverage'],
                         original_steps=e['steps']))
    count = sum(r['success'] for r in rows)
    summary = dict(episode_count=len(rows), success_count=count,
                   success_rate=count/len(rows), success_rate_wilson_95=_wilson_95(count,len(rows)),
                   original_final_gt87_count=sum(r['original_final_coverage']>threshold for r in rows),
                   ever_success_but_original_final_le87=sum(r['success'] and r['original_final_coverage']<=threshold for r in rows),
                   mean_release_stop_coverage=float(np.mean([r['release_stop_coverage'] for r in rows])))
    return rows, summary


def main():
    OUT.mkdir(parents=True, exist_ok=True); MEDIA.mkdir(parents=True, exist_ok=True)
    results=[]
    for step in (30000,35000,40000):
        source=REVIEW/f'step_{step}/results.json'
        payload=json.loads(source.read_text())
        rows, summary=rescore(payload['episodes'])
        results.append(dict(step=step, **summary, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        (OUT/f'step_{step}.json').write_text(json.dumps(rows,indent=2)+'\n')
    report=dict(threshold=THRESHOLD, comparison='>',
                success_definition='Any executed step exceeds 0.87; first crossing ends release episode.',
                method='Post-hoc rescoring of all saved traces from the original >0.95 evaluation; no new inference.',
                original_selection_threshold=0.95, original_selected_step=30000,
                original_traces_preserved=True, results=results)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    scenes={r['scene_id']:r for r in json.loads((REVIEW/'scenes.json').read_text())['scenes']}
    episodes={e['scene_id']:e for e in json.loads((REVIEW/'step_30000/results.json').read_text())['episodes']}
    manifest=[]
    for scene_id, category in EXAMPLES.items():
        e=episodes[scene_id]; s=scenes[scene_id]
        crossing=next((t['step'] for t in e['trace'] if t['coverage']>THRESHOLD),None)
        assert (crossing is not None)==(category=='success')
        stop=crossing or e['steps']
        env=make_pusht_env(max_steps=300)
        env.reset(seed=s['environment_seed'],options={'reset_to_state':SceneState(**s['state']).as_array()})
        frames=[];errors=[]
        def frame(step, coverage):
            im=Image.fromarray(np.asarray(env.render(),dtype=np.uint8)).convert('RGB')
            canvas=Image.new('RGB',(384,432),'#101827');canvas.paste(im,(0,48))
            draw=ImageDraw.Draw(canvas)
            draw.text((12,8),f'{category.upper()} | {scene_id} | checkpoint 30000',fill='white')
            draw.text((12,27),f'Step {step:03d}/{stop}  Coverage {coverage:.1%}  Target >87%',fill='#b8e7db')
            return np.array(canvas)
        frames.append(frame(0,float(env.unwrapped._get_coverage())))
        try:
            # Verify the entire source trajectory, even after the release stopping point.
            for t in e['trace']:
                _,_,_,_,info=env.step(np.asarray(t['action'],dtype=np.float32))
                errors.append(abs(float(info['coverage'])-t['coverage']))
                if t['step']<=stop: frames.append(frame(t['step'],float(info['coverage'])))
        finally: env.close()
        assert max(errors)<1e-8, (scene_id,max(errors))
        name=f'{category}-{scene_id}'
        iio.imwrite(MEDIA/f'{name}.mp4',np.stack(frames),plugin='pyav',fps=10,codec='libx264',out_pixel_format='yuv420p')
        previews=[Image.fromarray(f).resize((288,324)) for f in frames[::2]]
        previews.append(Image.fromarray(frames[-1]).resize((288,324)))
        previews[0].save(MEDIA/f'{name}.gif',save_all=True,append_images=previews[1:],duration=200,loop=0)
        Image.fromarray(frames[-1]).save(MEDIA/f'{name}.png')
        manifest.append(dict(scene_id=scene_id,category=category,checkpoint_step=30000,
                             environment_seed=s['environment_seed'],video=f'docs/media/{name}.mp4',
                             preview=f'docs/media/{name}.gif',frames=len(frames),fps=10,
                             first_success_step=crossing,original_steps=e['steps'],
                             original_max_coverage=e['max_coverage'],original_final_coverage=e['final_coverage'],
                             verified_full_trace_max_coverage_error=max(errors),
                             video_sha256=hashlib.sha256((MEDIA/f'{name}.mp4').read_bytes()).hexdigest()))
        print(scene_id,category,'verified',max(errors),flush=True)
    (OUT/'videos.json').write_text(json.dumps(dict(method='Recorded-action replay; same scene and seed, all original coverage values verified to 1e-8; success videos stop at first >87% crossing.',examples=manifest),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (ROOT/'reports/act/training/train_metrics.csv').open() as f: rows=list(csv.DictReader(f))
    steps=[int(r['step']) for r in rows]
    assert steps==list(range(1,40001))
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for ax,field,title in zip(axes[:2],['action_l1','kl'],['Action L1 loss','KL (Kullback-Leibler) divergence']):
        values=np.array([float(r[field]) for r in rows]);means=values.reshape(-1,200).mean(1)
        ax.plot(np.arange(200,40001,200),means,color='#168878',lw=1.8)
        ax.set(title=title,xlabel='Training step',yscale='log');ax.grid(alpha=.2)
    selection=json.loads((ROOT/'reports/act/training/selection.json').read_text())
    candidates=selection['candidates']
    axes[2].plot([r['step'] for r in candidates],[r['summary']['success_rate'] for r in candidates],marker='o',color='#3360af')
    axes[2].set(title='Original development success (>95%)',xlabel='Training step',ylabel='Success rate',ylim=(0,1));axes[2].grid(alpha=.2)
    fig.suptitle('ACT Push-T | seed 0 | 40,000 updates | loss: means over 200 updates')
    fig.savefig(MEDIA/'training-curves.png',dpi=160);plt.close(fig)
    (OUT/'training_audit.json').write_text(json.dumps(dict(rows=len(rows),unique_steps=len(set(steps)),first_step=steps[0],last_step=steps[-1],
        last_1000_mean_action_l1=float(np.mean([float(r['action_l1']) for r in rows[-1000:]])),
        last_1000_mean_kl=float(np.mean([float(r['kl']) for r in rows[-1000:]])),
        elapsed_seconds=float(rows[-1]['elapsed_seconds']),
        train_metrics_sha256=hashlib.sha256((ROOT/'reports/act/training/train_metrics.csv').read_bytes()).hexdigest()),indent=2)+'\n')
    print(json.dumps(report['results'],indent=2))

if __name__=='__main__': main()
