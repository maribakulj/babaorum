"""Hash only read/, lines/, alto/ recursively (including alto/indep).
Does not cover other E22 subtrees or establish any OCR result.
"""
from pathlib import Path
import sys, subprocess
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from babaorum.contracts import digest
from babaorum.evidence import atomic_write

def run():
    root=Path('/Users/marcel/axel/runs/e22')
    files=[]
    for sub in ('read','lines','alto'):
        for p in sorted((root/sub).rglob('*')):
            if p.is_file():
                stat=p.stat()
                files.append(dict(path=str(p),relative_path=str(p.relative_to(root)),bytes=stat.st_size,sha256=digest(p),mtime_ns=stat.st_mtime_ns))
    result={'source':str(root),'source_commit':subprocess.check_output(['git','-C',str(root.parents[1]),'rev-parse','HEAD'],text=True).strip(),'scope':__doc__,'files':files,'count':len(files),'total_bytes':sum(f['bytes'] for f in files)}
    atomic_write(ROOT/'reports/e22_stable_inventory.json',result)
    print(result['count'],result['total_bytes'])
if __name__=='__main__':run()
