"""Bounded lightweight G0/G1 loop. Never installs, downloads weights or runs models.
Watch mode observes prerequisites; later research gates require new implementation.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from babaorum.contracts import digest
from babaorum.g0_receipt import ReceiptError, verify_metadata_receipt
from babaorum.g0_collections import CollectionRegistryError, eligibility, verify_collections, verify_coverage
from babaorum.g0_seal import SealStop, check_headline_measure


from babaorum.evidence import (pipeline_signature as code_signature, pipeline_path,
                               atomic_write as write, current, runtime_environment)


def head(repo):
    return subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()


def inventory(axel):
    out={'axel_head':head(axel),'free_bytes':shutil.disk_usage(ROOT).free,'manifests':[],'note':'Metadata only; image and GT files not opened.'}
    for rel in ('runs/e12/pages.json','data/prepared/olr/newseye_as.json','data/prepared/olr/newseye_as.cross.json','data/prepared/seg/manifest.json'):
        path=axel/rel
        item={'path':str(path),'exists':path.exists()}
        if path.exists():item.update(sha256=digest(path),bytes=path.stat().st_size)
        out['manifests'].append(item)
    out['c7_doc_key_present']=(axel/'src/axel/data/doc_key.py').is_file()
    out['e22_log_metadata']=[{'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in sorted((axel/'runs/e22/logs').glob('*.log'))]
    out['environment']=runtime_environment(ROOT)
    return out


def verify_port():
    manifest=json.loads((ROOT/'provenance/ported.json').read_text())
    for item in manifest['files']:
        source=subprocess.check_output(['git','-C',manifest['repository'],'show',manifest['commit']+':'+item['source']])
        if hashlib.sha256(source).hexdigest()!=item['source_sha256']:raise ValueError('source hash mismatch')
        target=(ROOT/item['target']).read_bytes()
        if target!=source.replace(b'from axel.',b'from babaorum.legacy.') or hashlib.sha256(target).hexdigest()!=item['target_sha256']:
            raise ValueError('port differs beyond namespace: '+item['target'])
    c7=json.loads((ROOT/'provenance/c7.json').read_text())
    source=subprocess.check_output(['git','-C',c7['repository'],'show',c7['commit']+':'+c7['source']])
    if source!=(ROOT/c7['target']).read_bytes() or hashlib.sha256(source).hexdigest()!=c7['sha256']:
        raise ValueError('C7 port mismatch')
    return {'verified_files':len(manifest['files'])+1,'commit':manifest['commit'],'c7_commit':c7['commit']}


def e13_status(valid):
    return {'role':'historical_diagnostic','gt_assisted':True,'current':bool(valid)}


def g0_receipt_current(receipt,axel):
    try:
        if receipt.get('metadata_current') is not True:return False
        # Metadata freshness is deliberately separate from split release:
        # a complete pending inventory can be current without being admissible.
        verify_metadata_receipt(receipt, expected_axel_root=axel)
        state={'head':head(axel),
               'status_porcelain':subprocess.check_output(['git','-C',str(axel),'status','--porcelain','--untracked-files=normal'],text=True),
               'tracked_diff_sha256':hashlib.sha256(subprocess.check_output(['git','-C',str(axel),'diff','--binary','HEAD'])).hexdigest()}
        before,after=receipt.get('axel_worktree_before'),receipt.get('axel_worktree_after')
        return (isinstance(before,dict) and before==after==state and receipt.get('axel_head')==state['head'])
    except (OSError,KeyError,TypeError,ValueError,ReceiptError,subprocess.CalledProcessError):return False


def manifest_eligibility(specs,registry_path):
    """Every page of each ROLE:PATH manifest goes through eligibility(); one refusal fails the loop."""
    refused=[];checked=0
    try:
        registry=json.loads(Path(registry_path).read_text())
        for spec in specs:
            role,_,path=spec.partition(':')
            for page in json.loads(Path(path).read_text())['pages']:
                result=eligibility(registry,role,page_key=page.get('page_key'),issue_key=page.get('issue_key'),title_key=page.get('title_key'))
                checked+=1
                if not result['allowed']:refused.append({'manifest':path,'role':role,'page':page,'refusals':result['refusals']})
    except (OSError,ValueError,KeyError,TypeError,CollectionRegistryError) as exc:
        return {'ok':False,'checked':checked,'error':str(exc),'refused_count':len(refused),'refused':refused[:20]}
    return {'ok':not refused,'checked':checked,'refused_count':len(refused),'refused':refused[:20]}


def headline_check(paths,seal_path):
    """A measure presented as the one that counts may only contain headline=true pages of the seal."""
    checked=0
    try:
        sealed=json.loads(Path(seal_path).read_text())['pages']
        for path in paths:
            pages=json.loads(Path(path).read_text())['pages']
            check_headline_measure(pages,sealed);checked+=len(pages)
    except (OSError,ValueError,KeyError,TypeError,SealStop) as exc:
        return {'ok':False,'checked':checked,'error':str(exc)}
    return {'ok':True,'checked':checked}


def cycle(axel,run_checks,manifests=(),headline_measures=()):
    if axel.resolve()!=Path('/Users/marcel/axel'):
        raise ValueError('alternate Axel root is not supported by the frozen evidence')
    info=inventory(axel);write(ROOT/'reports/inventory.json',info)
    g0_path=ROOT/'reports/g0_inventory.json'
    g0_inventory=json.loads(g0_path.read_text()) if g0_path.exists() else {}
    g0_refreshed=False
    if not g0_receipt_current(g0_inventory,axel):
        proc=subprocess.run([sys.executable,str(ROOT/'scripts/g0_inventory.py')],cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=60)
        if proc.returncode:raise RuntimeError('G0 metadata inventory failed: '+proc.stderr[-2000:])
        g0_inventory=json.loads(g0_path.read_text());g0_refreshed=True
    g0_current=g0_receipt_current(g0_inventory,axel)
    g0_receipt_sha256=digest(g0_path) if g0_path.is_file() else None
    try:collections={'verified':True,**verify_collections(ROOT/'config/g0_collections_v2_2.json')}
    except CollectionRegistryError as exc:collections={'verified':False,'error':str(exc)}
    try:coverage={'verified':True,**verify_coverage(ROOT/'config/g0_coverage_v2_2.json',ROOT/'config/g0_collections_v2_2.json')}
    except (CollectionRegistryError,OSError,KeyError) as exc:coverage={'verified':False,'error':str(exc)}
    manifest_check=manifest_eligibility(manifests,ROOT/'config/g0_collections_v2_2.json') if manifests else None
    headline=headline_check(headline_measures,ROOT/'config/sealed/presse_e7_test_v2_1.json') if headline_measures else None
    port=verify_port();write(ROOT/'reports/port_verification.json',port)
    signature=code_signature(ROOT);receipt=ROOT/'reports/checks.json'
    prior=json.loads(receipt.read_text()) if receipt.exists() else {}
    if run_checks and (prior.get('signature')!=signature or not prior.get('passed')):
        tests=[str(p.relative_to(ROOT)) for p in sorted((ROOT/'tests').glob('test_*.py')) if pipeline_path(p.relative_to(ROOT))]
        proc=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*tests],cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=120)
        stable=code_signature(ROOT)==signature
        prior={'signature':signature,'scope':'pipeline','source_stable_during_checks':stable,
               'runner_environment':runtime_environment(ROOT),
               'passed':proc.returncode==0 and stable,'exit_code':proc.returncode,
               'tests':tests,'output':proc.stdout+proc.stderr}
        write(receipt,prior)
    parity=ROOT/'reports/replay_dev.json'
    replay=None
    if parity.exists():
        replay=json.loads(parity.read_text())
        # A changed input invalidates evidence rather than silently rerunning it.
        replay_valid=replay.get('signature')==signature and replay.get('all_identical') and all(Path(p).is_file() and digest(p)==sha for p,sha in replay['inputs'].items())
    else:replay_valid=False
    e13_path=ROOT/'reports/replay_e13.json'
    e13_valid=False
    if e13_path.exists():
        e13=json.loads(e13_path.read_text())
        e13_valid=e13.get('signature')==signature and e13.get('all_identical') is True and e13.get('role')=='historical_diagnostic' and e13.get('gt_assisted') is True and all(Path(p).is_file() and digest(p)==sha for p,sha in e13['inputs'].items())
    checks_current=current(prior,ROOT,scope='pipeline')
    checked_at=datetime.now(timezone.utc).isoformat()
    state={'scope':'G0/G1 support loop; not a training scheduler','checks_passed':checks_current,
           'checked_at_utc':checked_at,'updated_at_utc':checked_at,'pipeline_signature':signature,'pid':os.getpid(),
           'runner_environment':runtime_environment(ROOT),
           'historical_dev_replay_current':bool(replay_valid),'historical_e13_replay_current':bool(e13_valid),'g0_metadata_inventory_current':g0_current,'g0_inventory_refreshed':g0_refreshed,'free_bytes':info['free_bytes'],
           'g0_receipt_sha256':g0_receipt_sha256,'g0_collections':collections,'g0_coverage':coverage,'g0_manifests':manifest_check,'g0_headline_measure':headline,
           'missing_exposure_sources':g0_inventory.get('missing_exposure_sources'),
           'g0_inventory_counts':g0_inventory.get('counts') if g0_current else None,
           'gates':{
               'G0':{'status':'partial','metadata_receipt_sha256':g0_receipt_sha256,'metadata_current':g0_current,
                     'done':['C7 canonical page/issue identity port verified','metadata-only source inventory and overlap report'], 'requires':['pretraining/exposure audit of candidate issues','new split by issue and title','GT and image-duplicate audit','Claude review']},
               'G1':{'status':'partial','e13':e13_status(e13_valid),'done':['namespace-only metric port verified','page_graph.v1 output contract and traceable PAGE/ALTO export bundle implemented','line_geometry.v1 and paired ratio comparison metrics implemented']+(['regression tests pass'] if checks_current else [])+(['E22 five dev pages numerically reproduced'] if replay_valid else [])+(['E13 seven historical pages x two sources reproduced'] if e13_valid else []),
                     'requires':([] if e13_valid else ['E13 frozen replay current'])+['PERO checkpoint and pretraining provenance','independent review of page/ALTO and geometry metrics','Claude review']},
               'G2':{'status':'waiting','requires':['G0/G1 criteria','exact PERO checkpoint and execution resources']},
               'G3':{'status':'waiting','priority':'parallel to G2','requires':['clean segmentation dataset','E22/E23 evidence','available storage and compute budget']},
               'G4':{'status':'conditional','requires':['G3 diagnosis','predeclared useful improvement target','compute budget']}}}
    if not checks_current:state['gates']['G1']['status']='checks_pending_or_failed'
    if not g0_current:state['gates']['G0']['status']='stale_metadata'
    elif not collections['verified']:state['gates']['G0']['status']='collections_registry_invalid'
    elif not coverage['verified']:state['gates']['G0']['status']='coverage_invalid'
    if manifest_check is not None and not manifest_check['ok']:
        state['checks_passed']=False;state['gates']['G0']['status']='manifest_refused'
    if headline is not None and not headline['ok']:
        state['checks_passed']=False;state['gates']['G0']['status']='headline_measure_invalid'
    write(ROOT/'reports/loop_status.json',state)
    return state


def safe_cycle(axel,run_checks,manifests=(),headline_measures=()):
    try:
        return cycle(axel,run_checks,manifests,headline_measures)
    except Exception as exc:
        receipt=ROOT/'reports/g0_inventory.json'
        try:
            raw=receipt.read_bytes();snapshot=json.loads(raw)
            receipt_sha=hashlib.sha256(raw).hexdigest()
            missing=snapshot.get('missing_exposure_sources')
        except (OSError,ValueError,TypeError):
            receipt_sha=None;missing=None
        now=datetime.now(timezone.utc).isoformat()
        write(ROOT/'reports/loop_status.json',{'scope':'G0/G1 support loop','status':'failed','checks_passed':False,
              'historical_dev_replay_current':False,'historical_e13_replay_current':False,
              'g0_metadata_inventory_current':False,'g0_receipt_sha256':receipt_sha,
              'missing_exposure_sources':missing,'pipeline_signature':code_signature(ROOT),
              'checked_at_utc':now,'updated_at_utc':now,'error':str(exc)})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--axel',type=Path,default=Path('/Users/marcel/axel'))
    parser.add_argument('--watch',action='store_true');parser.add_argument('--interval',type=int,default=60)
    parser.add_argument('--cycles',type=int,default=120);parser.add_argument('--check',action='store_true')
    parser.add_argument('--headline-measure',action='append',default=[],metavar='PATH',help='JSON {"pages":[page_id]} presented as the measure that counts: every page must be headline=true in the presse_e7 test seal')
    parser.add_argument('--manifest',action='append',default=[],metavar='ROLE:PATH',help='JSON {"pages":[{page_key,issue_key,title_key}]} checked with eligibility() for ROLE (train, dev or test)')
    args=parser.parse_args()
    if args.interval<10 or args.cycles<1:parser.error('interval >=10 and cycles >=1 required')
    (ROOT/'reports').mkdir(exist_ok=True)
    with (ROOT/'reports/loop.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:parser.exit(2,'another loop owns the lock\n')
        for i in range(args.cycles if args.watch else 1):
            result=safe_cycle(args.axel,args.check,args.manifest,args.headline_measure)
            if result.get('g0_headline_measure') and not result['g0_headline_measure']['ok']:print(json.dumps({'headline_measure_invalid':result['g0_headline_measure'].get('error')}),flush=True);parser.exit(3,'a headline measure contains a page that is not headline=true\n')
            if result.get('g0_manifests') and not result['g0_manifests']['ok']:print(json.dumps({'manifest_refused':result['g0_manifests']['refused_count']}),flush=True);parser.exit(3,'a manifest page is not eligible\n')
            print(json.dumps({'cycle':i+1,'checks':result['checks_passed'],'dev_parity':result['historical_dev_replay_current'],'gates':{k:v['status'] for k,v in result['gates'].items()}}),flush=True)
            if args.watch and i+1<args.cycles:time.sleep(args.interval)

if __name__=='__main__':main()
