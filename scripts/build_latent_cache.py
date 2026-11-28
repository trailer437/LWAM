import argparse
from pathlib import Path
import numpy as np
import torch
from lwam.training.common import config,device_for
from lwam.models.vision_encoder import FrozenVisionEncoder
from lwam.data.latent_cache import save_episode,encode_frames

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/rtx4070_8gb.yaml'); p.add_argument('--max-episodes',type=int); p.add_argument('--synthetic',action='store_true',help='Test-only synthetic latent trajectories; not PushT'); a=p.parse_args()
    cfg=config(a.config); c=cfg['data']; m=cfg['model']; dest=Path(c['cache_dir']); dest.mkdir(parents=True,exist_ok=True)
    if any(dest.glob('episode_*.npz')): raise RuntimeError('Cache already contains episodes; use a new cache_dir or deliberately clear it first')
    if a.synthetic:
        rng=np.random.default_rng(cfg['seed']); projection=rng.normal(0,.1,(m['state_dim'],m['latent_dim']))
        for i in range(a.max_episodes or 12):
            s=rng.uniform(-1,1,(31,m['state_dim'])); acts=np.diff(s,axis=0); z=s@projection
            save_episode(dest/f'episode_{i:06d}.npz',z,s,acts)
        return
    try: from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError: from lerobot.datasets import LeRobotDataset
    ds=LeRobotDataset(c['repo_id'],root=c['raw_dir'],revision=c['revision'],video_backend='pyav')
    if c['goal_mode']!='final_observation': raise ValueError('Only final_observation goal mode implemented')
    if abs(float(ds.fps)-c['fps'])>1e-6: raise ValueError('Unexpected dataset FPS')
    dev=device_for(cfg); enc=FrozenVisionEncoder(cfg['encoder']['name'],cfg['encoder']['image_size']).to(dev).eval()
    frames=[]; states=[]; actions=[]; current=None; count=0; last_frame=None
    def flush():
        if len(frames)<=m['horizon']: return False
        z=encode_frames(enc,frames,dev,cfg['encoder']['batch_size']); s=np.stack(states); act=np.stack(actions[:-1])
        if z.shape[1]!=m['latent_dim'] or s.shape[1]!=m['state_dim'] or act.shape[1]!=m['action_dim']: raise ValueError('Dataset/model dimensions differ')
        save_episode(dest/f'episode_{current:06d}.npz',z,s,act); print('cached',current,len(z),flush=True); return True
    for row in ds:
        ep=int(row['episode_index']); frame=int(row['frame_index'])
        if ep!=current:
            if current is not None:
                count+=int(flush())
                if a.max_episodes and count>=a.max_episodes: return
            frames=[]; states=[]; actions=[]; current=ep; last_frame=None
        if last_frame is not None and frame!=last_frame+1: raise ValueError('Non-contiguous frames')
        last_frame=frame
        frames.append(torch.as_tensor(row[c['image_key']]).cpu()); states.append(np.asarray(row[c['state_key']],dtype=np.float32)); actions.append(np.asarray(row[c['action_key']],dtype=np.float32))
    if current is not None: flush()
if __name__=='__main__': main()
