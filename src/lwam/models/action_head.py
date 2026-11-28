import torch
from torch import nn

class ActionHead(nn.Module):
    def __init__(self, latent_dim=384,state_dim=2,action_dim=2,hidden_dim=512,horizon=5):
        super().__init__()
        self.horizon,self.action_dim=horizon,action_dim
        self.net=nn.Sequential(nn.Linear(2*latent_dim+state_dim,hidden_dim),nn.GELU(),
                               nn.Linear(hidden_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,horizon*action_dim))
    def forward(self,z,state,goal):
        return self.net(torch.cat((z,state,goal),dim=-1)).reshape(-1,self.horizon,self.action_dim)
