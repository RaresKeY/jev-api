#!/usr/bin/env python3
"""Start a selected downloaded GGUF through the shared llama.cpp manager.

Standalone dependencies: Ignored GGUF plus executable ~/scripts/llm_manager.sh, Podman/CUDA image and nvidia-smi.
See docs/standalone-dependencies.md.
"""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from decision_lab.evaluation import ROOT,sha256
from decision_lab.runtime import MANAGER, require_runtime


def main():
    config=json.loads((ROOT/'configs/local-best.json').read_text())
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',nargs='?',default=config['default_gguf'],choices=[model['id'] for model in config['models']])
    parser.add_argument('--port',type=int,default=8080)
    args=parser.parse_args()
    require_runtime()
    model=next(model for model in config['models'] if model['id']==args.model)
    path=ROOT/model['path']
    if sha256(path)!=model['sha256']:parser.exit(2,'Model bytes changed; reevaluate before using saved settings.\n')
    env=dict(os.environ,LLM_CONTAINER_ENGINE='podman',LLM_CONTAINER='jev-local',LLM_PORT=str(args.port),LLM_BIND_ADDRESS='127.0.0.1',LLM_CTX='4096',LLM_PARALLEL='1',LLM_CACHE_RAM='0',LLM_BATCH='512',LLM_CACHE_TYPE_K=model['cache'],LLM_CACHE_TYPE_V=model['cache'],LLM_N_GPU_LAYERS='999',LLM_FLASH_ATTN='on',LLM_REASONING='off')
    for name in ('JEV_API','TYPESAFE_API_KEY','LLAMA_API_KEY'):env.pop(name,None)
    os.execve(MANAGER,[str(MANAGER),str(path)],env)

if __name__=='__main__':main()
