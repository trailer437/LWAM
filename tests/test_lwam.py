import numpy as np
import torch
from lwam.models.lwam import LWAM
from lwam.models.vision_encoder import FrozenVisionEncoder
from lwam.data.latent_cache import save_episode
from lwam.data.dataset import LatentDataset,split_episodes,fit_stats
from lwam.training.losses import world_loss,action_loss

def test_forward_backward():
    m=LWAM(latent_dim=8,hidden_dim=16,horizon=3); z=torch.randn(2,8); s=torch.randn(2,2); g=torch.randn(2,8)
    out=m(z,s,g); assert out['refined'].shape==(2,3,2); assert out['future_z'].shape==(2,3,8)
    assert torch.equal(out['initial'],out['refined'])
    b=dict(actions=torch.randn(2,3,2),future_z=torch.randn(2,3,8),future_state=torch.randn(2,3,2))
    w=world_loss(*m.world.rollout(z,s,b['actions']),b); (w+action_loss(out['refined'],b)).backward()
    assert all(any(p.grad is not None for p in mod.parameters()) for mod in [m.world,m.initial,m.refiner])

def test_cache_split(tmp_path):
    for i in range(6): save_episode(tmp_path/f'episode_{i}.npz',np.ones((7,8))*i,np.arange(14).reshape(7,2),np.ones((6,2)))
    train,val,test=split_episodes(tmp_path); assert len(set(train+val+test))==6
    ds=LatentDataset(train,3,fit_stats(train)); b=ds[0]; assert b['future_z'].shape==(3,8); assert b['actions'].shape==(3,2)
    assert torch.equal(b['z'],b['future_z'][0])

def test_encoder_frozen():
    class Fake(torch.nn.Module):
        def __init__(self): super().__init__(); self.weight=torch.nn.Parameter(torch.ones(1))
        def forward_features(self,x): return {'x_norm_clstoken':x.mean((2,3))*self.weight}
    e=FrozenVisionEncoder(backbone=Fake()); e.train(); out=e(torch.zeros(2,3,32,32)); assert not out.requires_grad and not e.backbone.training
