from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def split_episodes(cache_dir,seed=42,train_fraction=.8,val_fraction=.1):
    paths=sorted(Path(cache_dir).glob('episode_*.npz'))
    if len(paths)<3: raise ValueError('Need at least 3 cached episodes')
    if not (0<train_fraction<1 and 0<val_fraction<1 and train_fraction+val_fraction<1):
        raise ValueError('Invalid split fractions')
    rng=np.random.default_rng(seed); rng.shuffle(paths)
    n=len(paths); nt=min(n-2,max(1,int(n*train_fraction))); nv=min(n-nt-1,max(1,int(n*val_fraction)))
    return paths[:nt],paths[nt:nt+nv],paths[nt+nv:]


def fit_stats(paths):
    arrays=[]
    for p in paths:
        with np.load(p) as e: arrays.append((e['states'],e['actions']))
    out={}
    for key,idx in [('state',0),('action',1)]:
        a=np.concatenate([x[idx] for x in arrays]); out[key+'_mean']=torch.tensor(a.mean(0)); out[key+'_std']=torch.tensor(a.std(0).clip(1e-6))
    return out

class LatentDataset(Dataset):
    def __init__(self,paths,horizon,stats):
        self.episodes=[]; self.indices=[]; self.horizon=horizon
        for p in paths:
            with np.load(p) as e:
                z=torch.tensor(e['z'],dtype=torch.float32); s=torch.tensor(e['states']); a=torch.tensor(e['actions']); g=torch.tensor(e['goal'],dtype=torch.float32)
            if len(z)!=len(a)+1 or len(s)!=len(z): raise ValueError(f'Invalid episode: {p}')
            s=(s-stats['state_mean'])/stats['state_std']; a=(a-stats['action_mean'])/stats['action_std']
            idx=len(self.episodes); self.episodes.append((z,s,a,g))
            self.indices.extend((idx,t) for t in range(len(a)-horizon+1))
        if not self.indices: raise ValueError('No complete action windows; reduce horizon or use longer episodes')
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        e,t=self.indices[i]; z,s,a,g=self.episodes[e]; h=self.horizon
        return dict(z=z[t],state=s[t],goal=g,actions=a[t:t+h],future_z=z[t+1:t+h+1],future_state=s[t+1:t+h+1])
