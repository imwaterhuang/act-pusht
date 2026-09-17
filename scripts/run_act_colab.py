"""Start a frozen ACT run only after target-device and rollout validation passed."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from mini_wam.training.act import train_act
from mini_wam.training.act_config import load_act_config
from mini_wam.training.artifacts import sha256_file, sync_run_directory

def main():
    p=argparse.ArgumentParser();p.add_argument('--validation-dir',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--mirror-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/act_seed0.yaml')
    p.add_argument('--resume',type=Path);a=p.parse_args()
    complete=json.loads((a.validation_dir/'validation_complete.json').read_text())
    smoke=json.loads((a.validation_dir/'smoke_result.json').read_text())
    overfit=json.loads((a.validation_dir/'overfit_result.json').read_text())
    interface=json.loads((a.validation_dir/'interface/interface_result.json').read_text())
    assert all(x['passed'] for x in (complete,smoke,overfit,interface))
    assert smoke['resume_exact'] and smoke['drive_restore_exact'] and interface['best_selected']
    assert torch.cuda.is_available() and torch.cuda.get_device_name()==complete['gpu']
    cfg=load_act_config(a.config)
    assert cfg['training']['batch_size']==complete['selected']['batch_size']
    assert a.mirror_dir.parent.is_dir(), 'Persistent Drive root must already exist'
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    # The notebook saves a separate launch record before this call. No credentials in artifacts.
    train_act(a.config,run_dir=a.run_dir,resume_path=a.resume,mirror_dir=a.mirror_dir,
        device_name='cuda',num_workers=complete['selected']['num_workers'])
    run=a.run_dir
    complete_run=json.loads((run/'completed.json').read_text())
    assert complete_run['step']==cfg['training']['train_steps']
    print('ACT_FORMAL_TRAINING_COMPLETE',json.dumps(complete_run),flush=True)
if __name__=='__main__':main()
