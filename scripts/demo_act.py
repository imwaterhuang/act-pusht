"""Run the ACT interactive demo without downloading the training dataset."""
import argparse
import sys
from pathlib import Path
import torch
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import uvicorn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mini_wam.studio.live import register_live_routes

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/studio/models/act_step_30000.pt')
    parser.add_argument('--port',type=int,default=7860)
    args=parser.parse_args()
    if not args.checkpoint.is_file(): parser.error('Download the release checkpoint first; see README.md.')
    torch.set_num_threads(4)
    app=FastAPI()
    @app.get('/')
    def index(): return RedirectResponse('/live')
    register_live_routes(app,args.checkpoint.resolve())
    uvicorn.run(app,host='127.0.0.1',port=args.port)

if __name__=='__main__': main()
