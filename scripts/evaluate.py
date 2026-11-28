"""Held-out imitation and recorded-action dynamics metrics (not task success)."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from lwam.models.lwam import LWAM
from lwam.data.dataset import LatentDataset
from lwam.training.common import config,device_for

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/rtx4070_8gb.yaml'); p.add_argument('--checkpoint',default='checkpoints/joint.pt'); a=p.parse_args()
    c=config(a.config); d=device_for(c); ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True); m=LWAM(**ck['model_config']).to(d); m.load_state_dict(ck['model']); m.eval()
    ds=LatentDataset([Path(c['data']['cache_dir'])/x for x in ck['splits'][2]],ck['model_config']['horizon'],ck['stats']); sums=torch.zeros(4); n=0
    with torch.no_grad():
        for b in DataLoader(ds,batch_size=32):
            b={k:v.to(d) for k,v in b.items()}; o=m(b['z'],b['state'],b['goal']); z,s=m.world.rollout(b['z'],b['state'],b['actions']); k=len(z); n+=k
            sums+=torch.tensor([(o['initial']-b['actions']).square().mean().item(),(o['refined']-b['actions']).square().mean().item(),(z-b['future_z']).square().mean().item(),(s-b['future_state']).square().mean().item()])*k
    print(dict(zip(['initial_action_mse','refined_action_mse','latent_mse','state_mse'],(sums/n).tolist())))
if __name__=='__main__': main()
