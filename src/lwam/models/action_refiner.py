import torch
from torch import nn

class ActionRefiner(nn.Module):
    def __init__(self,latent_dim=384,state_dim=2,action_dim=2,hidden_dim=512,horizon=5):
        super().__init__()
        # Share temporal projection to keep the refiner small.
        self.project=nn.Sequential(nn.Linear(latent_dim+state_dim,128),nn.GELU())
        self.net=nn.Sequential(nn.Linear(horizon*(128+action_dim)+latent_dim,hidden_dim),nn.GELU(),
                               nn.Linear(hidden_dim,horizon*action_dim))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,initial,future_z,future_state,goal):
        features=self.project(torch.cat((future_z,future_state),dim=-1))
        x=torch.cat((initial.flatten(1),features.flatten(1),goal),dim=-1)
        return initial+self.net(x).reshape_as(initial)
