import argparse
from huggingface_hub import snapshot_download
from lwam.training.common import config
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/rtx4070_8gb.yaml'); a=p.parse_args(); d=config(a.config)['data']
    print(snapshot_download(repo_id=d['repo_id'],repo_type='dataset',revision=d['revision'],local_dir=d['raw_dir']))
