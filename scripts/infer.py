"""Offline inference from two RGB images and measured robot state; no actuation."""
import argparse
import torch
import numpy as np
from PIL import Image
from lwam.models.lwam import LWAM
from lwam.models.vision_encoder import FrozenVisionEncoder
from lwam.training.common import config,device_for

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/rtx4070_8gb.yaml'); p.add_argument('--checkpoint',default='checkpoints/joint.pt'); p.add_argument('--image',required=True); p.add_argument('--goal',required=True); p.add_argument('--state',nargs='+',type=float,required=True); a=p.parse_args()
    c=config(a.config); d=device_for(c); ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True); m=LWAM(**ck['model_config']).to(d).eval(); m.load_state_dict(ck['model']); e=FrozenVisionEncoder(c['encoder']['name'],c['encoder']['image_size']).to(d).eval(); st=ck['stats']; s=torch.tensor(a.state)
    if s.numel()!=ck['model_config']['state_dim']: raise ValueError('Incorrect state dimension')
    def image(path): return torch.from_numpy(np.array(Image.open(path).convert('RGB'))).permute(2,0,1).unsqueeze(0).to(d)
    with torch.no_grad():
        o=m(e(image(a.image)),((s-st['state_mean'])/st['state_std']).unsqueeze(0).to(d),e(image(a.goal)))
    for key in ('initial','refined'): print(key, (o[key].cpu()*st['action_std']+st['action_mean']).tolist())
if __name__=='__main__': main()
