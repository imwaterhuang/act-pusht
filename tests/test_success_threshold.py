"""Threshold boundaries must affect actual rollout stopping, not only labels."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import mini_wam.evaluation.act as evaluation

class Policy(torch.nn.Module):
    def forward(self, *_): return torch.zeros(1,4,2)

@pytest.mark.parametrize('threshold,expected_steps',[(0.87,2),(0.95,3)])
def test_threshold_changes_first_crossing_stop(monkeypatch,tmp_path,threshold,expected_steps):
    class Env:
        def reset(self,**kwargs): self.n=0; return {},{}
        def step(self,action):
            coverage=[0.87,0.871,0.96][self.n]; self.n+=1
            return {'agent_pos':np.zeros(2)},0.0,False,False,{'coverage':coverage}
        def close(self): pass
    monkeypatch.setattr(evaluation,'make_pusht_env',lambda **kw:Env())
    monkeypatch.setattr(evaluation,'load_scene_split',lambda *a,**kw:{'scenes_hash':'test','scenes':[{
        'scene_id':'boundary','environment_seed':0,
        'state':dict(agent_x=100,agent_y=100,block_x=200,block_y=200,block_angle=0)}]})
    monkeypatch.setattr(evaluation,'_observation_tensors',lambda *a:(None,None))
    model=Policy().train()
    result=evaluation.evaluate_act_policy(model,SimpleNamespace(denormalize_action=lambda a:a),
        tmp_path/'scenes.json',tmp_path/'result',device=torch.device('cpu'),success_coverage=threshold)
    assert result['episodes'][0]['steps']==expected_steps
    assert result['episodes'][0]['is_success'] is True
    assert result['protocol']['strict_success_coverage']==threshold
    assert model.training

@pytest.mark.parametrize('value',[-.1,1.1,float('nan')])
def test_rejects_invalid_threshold(tmp_path,value):
    with pytest.raises(ValueError,match='success_coverage'):
        evaluation.evaluate_act_policy(Policy(),None,tmp_path/'missing',tmp_path,
            device=torch.device('cpu'),success_coverage=value)
