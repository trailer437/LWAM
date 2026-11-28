import torch
from torch import nn

class TinyDynamics(nn.Module):
    """Predict next visual latent and robot state from executed action."""
    def __init__(self, latent_dim=384, state_dim=2, action_dim=2, hidden_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(nn.Linear(latent_dim+state_dim+action_dim,hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim,hidden_dim), nn.GELU(), nn.Linear(hidden_dim,latent_dim+state_dim))
    def forward(self, z, state, action):
        delta = self.net(torch.cat((z,state,action),dim=-1))
        return z+delta[...,:self.latent_dim], state+delta[...,self.latent_dim:]

    def rollout(self,z,state,actions):
        zs, ss = [], []
        for a in actions.unbind(dim=1):
            z,state = self(z,state,a)
            zs.append(z); ss.append(state)
        return torch.stack(zs,dim=1), torch.stack(ss,dim=1)
