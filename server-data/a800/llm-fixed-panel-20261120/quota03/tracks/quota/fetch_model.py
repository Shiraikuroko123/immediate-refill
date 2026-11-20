"""Download the pinned distillation checkpoint and verify every selected file."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from hashlib import sha256
from urllib.parse import quote
import json
import subprocess
import sys


def fetch(model_dir):
    manifest = json.loads((Path(__file__).resolve().parent/'sources/model-files.json').read_bytes())
    model_dir.mkdir(parents=True, exist_ok=True)
    def retrieve(item):
        dest = model_dir/item['Name']
        part = dest.with_suffix(dest.suffix+'.part')
        if not dest.exists():
            url = f"https://www.modelscope.cn/models/{manifest['repo']}/resolve/{item['Revision']}/{quote(item['Name'])}"
            subprocess.run(['curl','-fsSL','--retry','3','--connect-timeout','15','-C','-','-o',str(part),url],check=True)
        path = dest if dest.exists() else part
        h = sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(2**20),b''): h.update(block)
        if path.stat().st_size!=item['Size'] or h.hexdigest()!=item['Sha256']:raise ValueError(item['Name'])
        if path==part: part.replace(dest)
        print('verified',dest.name,flush=True)
    with ThreadPoolExecutor(4) as pool:list(pool.map(retrieve,manifest['files']))

if __name__=='__main__': fetch(Path(sys.argv[1]))
