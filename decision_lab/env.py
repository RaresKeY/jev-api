"""Read selected .env keys as data, never execute or log its contents."""
import os
from pathlib import Path


def load_env(path):
    allowed = {'JEV_API', 'TYPESAFE_API_KEY', 'LLAMA_API_KEY', 'LLAMA_BASE_URL', 'LLAMA_MODEL'}
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, sep, value = line.partition('=')
        name = name.removeprefix('export ').strip()
        if sep and name in allowed:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(name, value)
