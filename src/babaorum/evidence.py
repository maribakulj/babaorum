"""Content-bound evidence; atomic reports, no training dependencies."""
import fcntl
import hashlib
import importlib.metadata
import subprocess
import sys
import json
from pathlib import Path
import tempfile
import os


def pipeline_path(path):
    """Files owned by the pipeline; joint-model workers have separate receipts."""
    name=path.as_posix()
    return not (
        name.startswith('src/babaorum/experimental/') or
        name.startswith('src/babaorum/reader/') or
        name.startswith('tests/test_joint') or name.startswith('scripts/probe_joint') or
        name.startswith('scripts/analyze_joint') or
        name.startswith('tests/test_reader') or name.startswith('scripts/reader_') or
        name.startswith('config/reader') or name.startswith('docs/READER') or
        name.startswith('config/joint') or name.startswith('docs/END_TO_END') or
        name in ('docs/LITTERATURE_END_TO_END.txt','docs/PROTOTYPE_JOINT.txt',
                 'docs/PASSATION_END_TO_END.txt','provenance/literature.json')
    )


def runtime_environment(root):
    """Describe the runner and the external Axel environment used for checks."""
    root = Path(root)
    environment = {
        'python': sys.version,
        'executable': str(Path(sys.executable).resolve()),
        'packages': {},
    }
    for name in ('rapidfuzz', 'numpy', 'pillow', 'lxml', 'pytest'):
        try:
            environment['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            environment['packages'][name] = None
    provenance = root / 'provenance/ported.json'
    if provenance.is_file():
        source = Path(json.loads(provenance.read_text())['repository'])
        lockfile = source / 'uv.lock'
        environment['axel_uv_lock_sha256'] = (
            hashlib.sha256(lockfile.read_bytes()).hexdigest() if lockfile.is_file() else None
        )
    return environment


def signature(root, *, scope='all'):
    if scope not in ('all','pipeline'):raise ValueError('unknown evidence scope')
    root=Path(root); h=hashlib.sha256()
    h.update(('babaorum-evidence-v2:'+scope).encode())
    paths=[root/'.gitignore',root/'pyproject.toml']
    for folder in ('src','tests','scripts','provenance','config','docs'):
        paths.extend(p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for path in sorted(paths):
        if scope=='pipeline' and not pipeline_path(path.relative_to(root)):continue
        if path.is_file():
            h.update(str(path.relative_to(root)).encode());h.update(b'\0');h.update(path.read_bytes());h.update(b'\0')
    environment = runtime_environment(root)
    provenance=root/'provenance/ported.json'
    if provenance.is_file():
        source=json.loads(provenance.read_text())['repository']
        environment['axel_head']=subprocess.check_output(['git','-C',source,'rev-parse','HEAD'],text=True).strip()
        environment['axel_status']=subprocess.check_output(['git','-C',source,'status','--porcelain','--untracked-files=normal'],text=True)
        environment['axel_diff_sha256']=hashlib.sha256(subprocess.check_output(['git','-C',source,'diff','--binary','HEAD'])).hexdigest()
    h.update(json.dumps(environment,sort_keys=True).encode())
    return h.hexdigest()


def pipeline_signature(root):
    return signature(root,scope='pipeline')


def current(receipt,root, *, scope='all'):
    return receipt.get('signature')==signature(root,scope=scope) and receipt.get('passed') is True


def atomic_write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix(path.suffix+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        name=None
        try:
            with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False,encoding='utf-8') as stream:
                name=stream.name
                json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
                stream.flush();os.fsync(stream.fileno())
            os.replace(name,path)
            directory_fd=os.open(path.parent,os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if name and os.path.exists(name):os.unlink(name)
