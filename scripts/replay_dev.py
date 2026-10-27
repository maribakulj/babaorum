"""Replay only whitelisted E22 development pages, without inference or source writes."""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from babaorum.contracts import digest
from babaorum.evidence import pipeline_signature as signature, atomic_write
from babaorum.legacy.layout import chains
from babaorum.legacy.layout.gt import load_page
from babaorum.legacy.layout.pagetext import score_page


def run(axel):
    axel=Path(axel).resolve()
    frozen=json.loads((ROOT/'config/replay_dev.json').read_text())
    if (frozen.get('schema_version')!='babaorum.historical_replay.v1' or
            frozen.get('role')!='historical_diagnostic' or
            frozen.get('dataset')!='E22' or frozen.get('split')!='dev' or
            frozen.get('chain')!='c1'):
        raise ValueError('structured historical replay metadata mismatch')
    if axel!=Path(frozen['repository']).resolve():raise ValueError('alternate --axel root forbidden by frozen replay')
    manifest=axel/'runs/e12/pages.json'
    if digest(manifest)!=frozen['manifest_sha256']:raise ValueError('historical manifest changed')
    for path,sha in frozen['inputs'].items():
        if digest(path)!=sha:raise ValueError('frozen input changed: '+path)
    pages=[p for p in json.loads(manifest.read_text()) if p['split']=='dev']
    if len(pages)!=5 or {p['page'] for p in pages}!=set(frozen['pages']):
        raise ValueError('expected frozen five-page historical dev')
    inputs={str(manifest):digest(manifest)}
    rows=[]; chars=0; sums={'cer_reordered':0.,'cer_reordered_d212':0.}
    chains.RUNS=axel/'runs/e12'
    for page in pages:
        name=page['page']
        assert page['split']=='dev'
        gt=Path(page['xml'])
        reference=axel/'runs/c6/e22/e12/dev'/f'{name}.json'
        expected=json.loads(reference.read_text())['chains']['c1']['text']
        base=chains.detector_chain('ppdl3_z',name,chains.FROZEN['ppdl3_z'])
        if base is None:
            raise ValueError(f'missing base chain: {name}')
        read=axel/'runs/e22/read/ppdl3_z'/f'{name}.json'
        if not read.is_file():raise ValueError('frozen E22 read missing; fallback forbidden')
        texts=json.loads(read.read_text())['texts']
        for path in (gt,reference,read,axel/'runs/e12/lines/ppdl3_z'/f'{name}.json',axel/'runs/e12/read/ppdl3_z'/f'{name}.json'):
            inputs[str(path)]=digest(path)
        score=score_page([texts.get(unit.id,'') for unit in base.units],[ln.text for ln in load_page(gt).lines()])
        actual={**asdict(score),'order':score.order}
        identical=set(actual)==set(expected) and all(abs(actual[k]-expected[k])<1e-12 for k in actual)
        rows.append({'page':name,'identical_to_c6':identical,'chars':score.chars,**{k:actual[k] for k in sums}})
        chars+=score.chars
        for k in sums:sums[k]+=actual[k]*score.chars
    if inputs!=frozen['inputs']:raise ValueError('read input hashes differ from frozen evidence')
    for path,sha in inputs.items():
        if digest(path)!=sha:raise ValueError('input changed during replay: '+path)
    result={'schema_version':frozen['schema_version'],'role':frozen['role'],
            'dataset':frozen['dataset'],'split':frozen['split'],'chain':frozen['chain'],
            'signature':signature(ROOT),'scope':'five historical dev pages ONLY; no inference; fractions not percent','inputs':inputs,'rows':rows,
            'all_identical':all(r['identical_to_c6'] for r in rows),'aggregates':{k:v/chars for k,v in sums.items()}}
    if not result['all_identical']:raise ValueError('legacy parity failure')
    if any(abs(result['aggregates'][k]-v)>1e-12 for k,v in frozen['expected_aggregates'].items()):raise ValueError('frozen aggregate mismatch')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--axel',default='/Users/marcel/axel')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/replay_dev.json')
    args=parser.parse_args()
    result=run(args.axel)
    if args.output.resolve().is_relative_to(Path(args.axel).resolve()):parser.error('output must not write in axel')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_write(args.output,result)
    print(json.dumps({'all_identical':result['all_identical'],'aggregates':result['aggregates']}))
