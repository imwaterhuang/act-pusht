"""Check pretrained initialization and actual ACT closed-loop checkpoint selection."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
import yaml
from mini_wam.training.act import train_act
from mini_wam.evaluation.act import generate_act_scene_split
from mini_wam.training.artifacts import sync_run_directory, sha256_file

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mirror',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    config=yaml.safe_load((ROOT/'configs/act_seed0.yaml').read_text())
    scenes=a.output/'interface_scenes.json'
    generate_act_scene_split(scenes,split='development',generation_seed=20260917,count=5,
        excluded_paths=(ROOT/'splits/act_development_scenes.json',))
    config['training'].update(batch_size=64,train_steps=2,warmup_steps=0)
    config['checkpoint']={'frequency':2}
    config['evaluation'].update(frequency=2,max_steps=20,development_scenes=str(scenes.resolve()))
    cfg=a.output/'interface.yaml';cfg.write_text(yaml.safe_dump(config))
    run=train_act(cfg,run_dir=a.output/'run',mirror_dir=a.mirror/'run',device_name='cuda',num_workers=2)
    selection=json.loads((run/'selection.json').read_text())
    assert selection['best_step']==2
    assert selection['candidates'][0]['summary']['episode_count']==5
    assert (a.mirror/'run/checkpoints/best.pt').is_file()
    report={'passed':True,'pretrained':True,'step':2,'scenes':5,'max_steps':20,
        'best_selected':True,'summary':selection['candidates'][0]['summary'],
        'checkpoint_sha256':sha256_file(run/'checkpoints/last.pt')}
    (a.output/'interface_result.json').write_text(json.dumps(report,indent=2)+'\n')
    sync_run_directory(a.output,a.mirror)
    print('ACT_PRETRAINED_ROLLOUT_SELECTION_VERIFIED',json.dumps(report),flush=True)
if __name__=='__main__':main()
