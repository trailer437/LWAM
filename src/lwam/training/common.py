import argparse, json, random
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from lwam.models.lwam import LWAM
from lwam.data.dataset import split_episodes,fit_stats,LatentDataset
from .losses import world_loss,action_loss

def config(path):
    with open(path) as f: return yaml.safe_load(f)

def device_for(cfg):
    d=cfg['device']; return torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if d=='auto' else d)

def run(stage):
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/rtx4070_8gb.yaml'); p.add_argument('--epochs',type=int); p.add_argument('--mode',choices=['refiner','joint'],default='refiner'); p.add_argument('--resume-refiner',action='store_true'); args=p.parse_args()
    cfg=config(args.config); tr=cfg['training']; d=device_for(cfg); seed=cfg['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    paths=split_episodes(cfg['data']['cache_dir'],seed,cfg['data']['train_fraction'],cfg['data']['val_fraction']); stats=fit_stats(paths[0]); h=cfg['model']['horizon']
    loaders=[DataLoader(LatentDataset(s,h,stats),batch_size=tr['batch_size'],shuffle=i==0,num_workers=tr['num_workers']) for i,s in enumerate(paths[:2])]
    model=LWAM(**cfg['model']).to(d); root=Path(tr['checkpoint_dir']); root.mkdir(parents=True,exist_ok=True)
    split_ids=[[p.name for p in s] for s in paths]
    def load(name,module=None):
        ck=torch.load(root/(name+'.pt'),map_location='cpu',weights_only=True)
        if ck['model_config']!=cfg['model'] or ck['splits']!=split_ids or any(not torch.equal(ck['stats'][k],v) for k,v in stats.items()): raise ValueError('Checkpoint configuration/data split/stats mismatch')
        if module: getattr(model,module).load_state_dict(ck['model'])
        else: model.load_state_dict(ck['model'])
    name=stage
    if stage=='world': trainable=model.world
    elif stage=='action': trainable=model.initial
    else:
        load('world','world'); load('action','initial')
        if args.resume_refiner: load('refiner')
        name=args.mode
        trainable=model.refiner if name=='refiner' else model
    model.requires_grad_(False); trainable.requires_grad_(True)
    opt=torch.optim.AdamW(trainable.parameters(),lr=tr['joint_learning_rate'] if name=='joint' else tr['learning_rate'])
    amp=d.type=='cuda' and tr['precision']=='bf16' and torch.cuda.is_bf16_supported()
    def loss(b):
        if stage=='world': return world_loss(*model.world.rollout(b['z'],b['state'],b['actions']),b)
        if stage=='action': return action_loss(model.initial(b['z'],b['state'],b['goal']),b)
        out=model(b['z'],b['state'],b['goal']); r=action_loss(out['refined'],b)
        if name=='refiner': return r
        # Crucially, supervise world predictions only with RECORDED actions.
        w=world_loss(*model.world.rollout(b['z'],b['state'],b['actions']),b)
        return tr['world_weight']*w+tr['initial_weight']*action_loss(out['initial'],b)+tr['refined_weight']*r
    best=float('inf'); logs=[]; epochs=args.epochs if args.epochs is not None else tr['epochs']
    if epochs<1: raise ValueError('epochs must be positive')
    if d.type=='cuda': torch.cuda.reset_peak_memory_stats(d)
    for epoch in range(epochs):
        metrics=[]
        for i,loader in enumerate(loaders):
            model.train(i==0); total=0.; count=0
            with torch.set_grad_enabled(i==0):
                for b in loader:
                    b={k:v.to(d) for k,v in b.items()}
                    if i==0: opt.zero_grad(set_to_none=True)
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16) if amp else nullcontext(): l=loss(b)
                    if not torch.isfinite(l): raise RuntimeError('Non-finite loss')
                    if i==0:
                        l.backward(); torch.nn.utils.clip_grad_norm_(trainable.parameters(),tr['gradient_clip']); opt.step()
                    n=len(b['z']); total+=l.item()*n; count+=n
            metrics.append(total/count)
        record=dict(epoch=epoch+1,train_loss=metrics[0],val_loss=metrics[1]); logs.append(record); print(record,flush=True)
        if metrics[1]<best:
            best=metrics[1]; torch.save(dict(model=(trainable if stage in ('world','action') else model).state_dict(),model_config=cfg['model'],stats=stats,splits=split_ids,stage=name),root/(name+'.pt'))
    Path('outputs').mkdir(exist_ok=True); Path(f'outputs/{name}.json').write_text(json.dumps(logs,indent=2))
    if d.type=='cuda': print('Peak allocated MiB:',torch.cuda.max_memory_allocated(d)/2**20)
