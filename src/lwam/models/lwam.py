from torch import nn
from .tiny_dynamics import TinyDynamics
from .action_head import ActionHead
from .action_refiner import ActionRefiner

class LWAM(nn.Module):
    """Latent-space core. Encoder is external so training can use a cache."""
    def __init__(self,latent_dim=384,state_dim=2,action_dim=2,hidden_dim=512,horizon=5):
        super().__init__()
        self.world=TinyDynamics(latent_dim,state_dim,action_dim,hidden_dim)
        self.initial=ActionHead(latent_dim,state_dim,action_dim,hidden_dim,horizon)
        self.refiner=ActionRefiner(latent_dim,state_dim,action_dim,hidden_dim,horizon)
    def forward(self,z,state,goal):
        initial=self.initial(z,state,goal)
        future_z,future_state=self.world.rollout(z,state,initial)
        refined=self.refiner(initial,future_z,future_state,goal)
        return dict(initial=initial,refined=refined,future_z=future_z,future_state=future_state)
