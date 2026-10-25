"""Recreate only the Mica runtime cache from the retained pinned image, offline.

Standalone dependencies: Podman, pinned native image, ../tooling/godot-podman runner, Mica assets and optional runtime packages.
See docs/standalone-dependencies.md.
"""
import json
from pathlib import Path
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_lab.evaluation import ROOT, sha256, write_json


def main():
    manifest = json.loads((ROOT/'docs/downloads/mica-runtime-20261015.json').read_text())
    image = manifest['image']
    metadata = json.loads(subprocess.check_output(['podman','image','inspect',image],text=True))[0]
    if metadata['Config']['Labels']['org.opencontainers.image.revision'] != 'df18bee11e96557b07aa7b12db0976f3c7ac9165':
        raise RuntimeError('Retained runtime image revision changed')
    output = ROOT/'.local/llama-df18bee-runtime'
    if not output.exists():
        output.mkdir(parents=True)
        cid = subprocess.check_output(['podman','create','--name','jev-mica-extract-'+uuid.uuid4().hex[:8],
            '--label','io.decision-lab.project=jev-api','--label','io.decision-lab.purpose=mica-library-extraction',
            '--entrypoint','/bin/true',image],text=True).strip()
        try:
            subprocess.run(['podman','cp',cid+':/usr/local/lib/.',str(output)],check=True)
        finally:
            subprocess.run(['podman','rm',cid],check=True,stdout=subprocess.DEVNULL)
            if subprocess.run(['podman','container','exists',cid]).returncode != 1:
                raise RuntimeError('Owned extraction container remains')
    for record in manifest['files']:
        path=output/record['name']
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise RuntimeError('Extracted library identity differs from validated runtime')
    write_json(output/'image.json',dict(image=image, image_id=metadata['Id'],
               retention='ephemeral extracted runtime libraries'))
    runner=ROOT.parent/'tooling/godot-podman/bin/godot-podman'
    subprocess.run([str(runner),'run','--project',str(ROOT),'--purpose','mica-abi-prepare',
        '--image','localhost/asset-pipeline:local','--display','none','--','bash','-c',
        'cc -I vendored/llama-abi-df18bee scripts/mica_abi_probe.c -o .local/mica-abi-probe && .local/mica-abi-probe > .local/mica-abi.json'],check=True)
    if json.loads((ROOT/'.local/mica-abi.json').read_text()) != manifest['abi']:
        raise RuntimeError('Compiled ABI differs from validated binding')
    print('Mica runtime cache and compiled ABI verified. No model inference or downloads.')


if __name__ == '__main__':
    main()
