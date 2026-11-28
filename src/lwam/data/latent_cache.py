"""Store one NPZ per episode. Each row is a recorded pre-action observation.
The final recorded action is dropped if no subsequent observation exists.
"""
from pathlib import Path
import numpy as np
import torch

def save_episode(path, z, states, actions, goal=None):
    z=np.asarray(z,dtype=np.float32); states=np.asarray(states,dtype=np.float32); actions=np.asarray(actions,dtype=np.float32)
    if z.ndim!=2 or states.ndim!=2 or actions.ndim!=2 or len(z)!=len(states) or len(z)!=len(actions)+1:
        raise ValueError('Expected T+1 observations and T actions')
    goal=z[-1] if goal is None else np.asarray(goal,dtype=np.float32)
    if goal.shape != z.shape[1:] or not all(np.isfinite(x).all() for x in (z,states,actions,goal)):
        raise ValueError('Invalid episode values or goal')
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path,z=z.astype(np.float16),states=states,actions=actions,goal=goal.astype(np.float16))

@torch.no_grad()
def encode_frames(encoder,frames,device,batch_size):
    result=[]
    for i in range(0,len(frames),batch_size):
        result.append(encoder(torch.stack(frames[i:i+batch_size]).to(device)).float().cpu())
    return torch.cat(result).numpy()
