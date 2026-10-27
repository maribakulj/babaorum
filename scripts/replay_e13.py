"""Historical E13 text replay, after Claude's explicit E22 completion notice.
Uses the pinned original registration functions (GT-assisted, diagnostic ONLY)
and Babaorum's namespace-only metric/XML ports. No inference, no image loads.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from babaorum.contracts import digest,parse
from babaorum.evidence import atomic_write,pipeline_signature as signature
from babaorum.legacy.data.alto_read import parse_alto
from babaorum.legacy.layout.gt import GtPage,load_page
from babaorum.legacy.layout.pagetext import norm,score_page


def read_pinned(path, frozen, inputs):
    path=Path(path)
    if str(path) not in frozen:raise ValueError('input outside frozen set: '+str(path))
    raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    if frozen.get(str(path))!=sha:raise ValueError('frozen input mismatch: '+str(path))
    inputs[str(path)]=sha
    return raw


def run():
    config=parse((ROOT/'config/replay_e13.json').read_bytes())
    if config.get('role')!='historical_diagnostic' or config.get('gt_assisted') is not True:raise ValueError('diagnostic labels required')
    frozen=config['inputs'];inputs={}
    pages=parse(read_pinned(config['manifest'],frozen,inputs))
    selected=[p for p in pages if p['split']=='test' and p['group']=='transkribus_1937']
    if len(selected)!=7 or {p['page'] for p in selected}!=set(config['pages']):raise ValueError('wrong historical page set')
    def _registration_helpers(raw):
        import numpy as np
        from rapidfuzz import fuzz,process
        names={'_trimmed_affine','register','register_lines','apply'}
        nodes=[node for node in ast.parse(raw).body if isinstance(node,ast.FunctionDef) and node.name in names]
        if {node.name for node in nodes}!=names:raise ValueError('registration functions missing')
        namespace={'np':np,'fuzz':fuzz,'process':process,'GtPage':GtPage,'Any':Any,'norm':norm}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned_e13_registration','exec'),namespace)
        return namespace
    helpers=_registration_helpers(read_pinned(config['registration_source'],frozen,inputs))
    reference=parse(read_pinned(config['reference'],frozen,inputs))
    expected={(r['src'],r['page']):r for r in reference['rows']}
    rows=[];sums={}
    for src in ('gallica','tesseract'):
        sums[src]={'chars':0,'d212':0.,'d217':0.}
        for p in selected:
            alto_path=Path(config['raw'])/(f"{p['ark']}.f{p['folio']}.alto.xml" if src=='gallica' else f"tesseract/{p['page']}.xml")
            alto=parse_alto(read_pinned(alto_path,frozen,inputs))
            # load_page is the unchanged path-based legacy loader; check before/after.
            read_pinned(p['xml'],frozen,inputs);gt=load_page(Path(p['xml']))
            if digest(p['xml'])!=frozen[p['xml']]:raise ValueError('GT changed during replay')
            reg=helpers['register'](alto,gt) if src=='gallica' else {'m':[[1.,0.],[0.,1.],[0.,0.]],'fitted':False}
            texts=[]
            for line in alto['lines']:
                points=helpers['apply'](line['box'],reg)
                cx=sum(x for x,_ in points)/4;cy=sum(y for _,y in points)/4
                if 0<=cx<=gt.width and 0<=cy<=gt.height:texts.append(line['text'])
            score=score_page(texts,[line.text for line in gt.lines()])
            row={'src':src,'page':p['page'],'chars':score.chars,'d212':score.cer_reordered_d212,'d217':score.cer_reordered}
            old=expected[(src,p['page'])]
            row['identical_to_c6']=all(abs(row[k]-old[k])<1e-12 for k in ('chars','d212','d217'))
            if not row['identical_to_c6']:raise ValueError('historical parity mismatch: '+p['page'])
            rows.append(row);sums[src]['chars']+=score.chars
            for k in ('d212','d217'):sums[src][k]+=row[k]*score.chars
    if inputs!=frozen:raise ValueError('input coverage differs from frozen configuration')
    for path,sha in frozen.items():
        if digest(path)!=sha:raise ValueError('input changed: '+path)
    aggregates={src:{'chars':s['chars'],**{k:s[k]/s['chars'] for k in ('d212','d217')}} for src,s in sums.items()}
    for src,agg in aggregates.items():
        for k in ('chars','d212','d217'):
            if abs(agg[k]-config['expected'][src][k])>1e-12:raise ValueError('aggregate mismatch')
    return {'role':'historical_diagnostic','gt_assisted':True,'signature':signature(ROOT),'all_identical':True,'inputs':inputs,'rows':rows,'aggregates':aggregates,
            'scope':'historical E13; 7 pages x 2 sources; fractions, not percent; no new OCR inference',
            'limitation':'Gallica registration and both sources cropping to GT page dimensions use GT as in E13; not a deployable pipeline or an independent test',
            'metric_versions':{'d212':'D-212 historical unit reordering','d217':'D-217 run reordering + minimum with ordered CER'}}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/replay_e13.json');args=parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT/'reports') and args.output.exists():parser.error('external output must not already exist')
    result=run();atomic_write(args.output,result);print(json.dumps(result['aggregates']))
