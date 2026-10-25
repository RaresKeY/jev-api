"""Launch through the configured shared manager; stop only the owned container."""
import json
import os
from pathlib import Path
import subprocess
import signal
import socket
import shutil
import uuid
import time
from .http import request_json

MANAGER = Path.home() / 'scripts/llm_manager.sh'


def require_runtime():
    """Check external workstation prerequisites before creating run artifacts."""
    missing = []
    if not MANAGER.is_file() or not os.access(MANAGER, os.X_OK):
        missing.append('executable ~/scripts/llm_manager.sh (not included)')
    missing.extend(command for command in ('podman', 'nvidia-smi') if not shutil.which(command))
    if missing:
        raise RuntimeError('Missing local runtime dependencies: '+', '.join(missing)+
                           '. See docs/standalone-dependencies.md')


class ManagedServer:
    def __init__(self,model,output,cache='q8_0',port=None,image=None,batch=512,parallel=1,ctx=4096,ubatch=None):
        self.model = Path(model).resolve()
        self.output = Path(output)
        self.cache = cache
        self.image = image
        self.batch = batch
        self.parallel = parallel
        self.ctx = ctx
        self.ubatch = ubatch
        if port is None:
            with socket.socket() as probe:
                probe.bind(('127.0.0.1',0))
                port=probe.getsockname()[1]
        self.port = port
        self.name = 'jev-ab-'+str(os.getpid())+'-'+uuid.uuid4().hex[:8]
        self.process = None
        self.logfile = None
        self.env = None

    def __enter__(self):
        require_runtime()
        self.output.mkdir(parents=True,exist_ok=True)
        if (self.output/'server.log').exists():
            self.output=self.output/('attempt-'+uuid.uuid4().hex[:8])
            self.output.mkdir()
        settings = dict(LLM_CONTAINER_ENGINE='podman',LLM_CONTAINER=self.name,LLM_PORT=str(self.port),LLM_CTX='4096',LLM_PARALLEL='1',LLM_CACHE_RAM='0',LLM_BATCH='512',LLM_CACHE_TYPE_K=self.cache,LLM_CACHE_TYPE_V=self.cache,LLM_N_GPU_LAYERS='999',LLM_FLASH_ATTN='on',LLM_REASONING='off',LLM_LOG_VERBOSITY='10',LLM_BIND_ADDRESS='127.0.0.1',TERM='xterm')
        if self.image:
            settings['LLM_IMAGE'] = self.image
        settings['LLM_BATCH'] = str(self.batch)
        settings['LLM_PARALLEL'] = str(self.parallel)
        settings['LLM_CTX'] = str(self.ctx)
        if self.ubatch is not None:
            settings['LLM_UBATCH'] = str(self.ubatch)
        self.env = dict(os.environ,**settings)
        # No cloud credentials are propagated to the manager process.
        for key in ('JEV_API','TYPESAFE_API_KEY','LLAMA_API_KEY'):
            self.env.pop(key,None)
        self.logfile = open(self.output/'server.log','w')
        self.process = subprocess.Popen([str(MANAGER),str(self.model)],env=self.env,stdout=self.logfile,stderr=subprocess.STDOUT,start_new_session=True)
        base = f'http://127.0.0.1:{self.port}'
        try:
            for _ in range(120):
                if self.process.poll() is not None:
                    raise RuntimeError('Manager exited before readiness; inspect server.log')
                # Confirm this container loaded its model before trusting the port.
                # A prior container's port-forwarder can briefly remain reachable.
                log_result = subprocess.run(['podman','logs',self.name],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False)
                startup = log_result.stdout.decode() if log_result.returncode == 0 else ''
                try:
                    if log_result.returncode == 0 and request_json(base+'/health',timeout=1).get('status')=='ok':
                        models = request_json(base+'/v1/models',timeout=2)
                        if any(self.model.name in item['id'] for item in models['data']):
                            break
                except RuntimeError:
                    pass
                time.sleep(.5)
            else:
                raise RuntimeError('Manager readiness timeout')
            text = subprocess.check_output(['podman','logs',self.name],stderr=subprocess.STDOUT).decode()
            (self.output/'startup.log').write_text(text)
            pid=subprocess.check_output(['podman','inspect',self.name,'--format','{{.State.Pid}}']).decode().strip()
            gpu_lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader']).decode().splitlines()
            gpu_processes=[line.strip() for line in gpu_lines if line.split(',')[0].strip()==pid]
            if not gpu_processes:
                raise RuntimeError('The actual server PID has no CUDA GPU allocation')
            gpu_proof=dict(server_pid=pid,allocations=gpu_processes,offload_log_available='offloaded' in text,requested_gpu_layers=999)
            from .evaluation import write_json
            write_json(self.output/'gpu.json',gpu_proof)
            self.props = request_json(base+'/props')
            self.models = request_json(base+'/v1/models')
            self.base_url = base
            from .evaluation import write_json
            write_json(self.output/'runtime.json',dict(settings=settings,props=self.props,models=self.models,gpu_proof=gpu_proof))
            return self
        except Exception:
            self.__exit__(None,None,None)
            raise

    def __exit__(self,*args):
        # Let Podman stop the container and its GPU context before terminating
        # the manager/logger process group. Then recheck this exact name.
        if self.env:
            subprocess.run([str(MANAGER),'stop'],env=self.env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
        if self.process and self.process.poll() is None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid,signal.SIGTERM)
                self.process.wait(timeout=10)
        if self.env:
            subprocess.run([str(MANAGER),'stop'],env=self.env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
        if self.logfile:
            self.logfile.close()
