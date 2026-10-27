import argparse
import json
from pathlib import Path
from .evidence import atomic_write
from .contracts import ContractError, load, validate_manifest


def main():
    parser=argparse.ArgumentParser(description='Babaorum: cached evaluation only, no model launch')
    sub=parser.add_subparsers(dest='command',required=True)
    v=sub.add_parser('validate');v.add_argument('manifest')
    vpage=sub.add_parser('validate-page');vpage.add_argument('graph')
    e=sub.add_parser('evaluate-lines');e.add_argument('manifest');e.add_argument('predictions');e.add_argument('--output',required=True)
    c=sub.add_parser('compare-lines');c.add_argument('manifest');c.add_argument('reference');c.add_argument('candidate');c.add_argument('--output',required=True)
    c.add_argument('--resamples',type=int,default=5000);c.add_argument('--seed',type=int,default=0)
    xp=sub.add_parser('export-page',help='Export a graph with required sidecar and loss report; no model launch')
    xp.add_argument('graph');xp.add_argument('--format',choices=('page','alto'),required=True)
    xp.add_argument('--image-filename',required=True);xp.add_argument('--physical-img-nr',type=int)
    xp.add_argument('--mode',choices=('strict','best_effort'),default='strict')
    xp.add_argument('--quantization',choices=('reject','nearest_pixel'),default='reject')
    xp.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    try:
        if args.command=='validate':
            result=validate_manifest(load(args.manifest));print(f'valid: {len(result["lines"])} lines')
        elif args.command=='export-page':
            from .export_bundle import export_bundle
            result=export_bundle(args.graph,args.output_dir,format=args.format,
                image_filename=args.image_filename,physical_img_nr=args.physical_img_nr,
                mode=args.mode,quantization=args.quantization)
            print(json.dumps({'accepted':result['accepted'],'xsd_validated':result['xsd_validated'],
                              'output_dir':str(Path(args.output_dir)),'losses':len(result['losses'])},ensure_ascii=False))
            if not result['accepted']:
                parser.exit(2,'projection rejected; source graph and loss report are preserved in output-dir\n')
        elif args.command=='validate-page':
            from .page_graph import validate_page_graph
            graph=load(args.graph);validate_page_graph(graph)
            print(f'valid page graph: {len(graph["lines"])} emitted lines; structural validation only')
        else:
            output=Path(args.output)
            inputs=[args.manifest,args.predictions] if args.command=='evaluate-lines' else [args.manifest,args.reference,args.candidate]
            if output.resolve() in {Path(p).resolve() for p in inputs}:
                raise ContractError('output would overwrite input')
            if args.command=='evaluate-lines':
                from .benchmark import evaluate
                result=evaluate(args.manifest,args.predictions)
            else:
                from .comparison import compare
                result=compare(args.manifest,args.reference,args.candidate,resamples=args.resamples,seed=args.seed)
            output.parent.mkdir(parents=True,exist_ok=True)
            atomic_write(output,result)
            print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    except (ValueError,OSError,KeyError,TypeError) as exc:
        parser.exit(2,f'error: {exc}\n')

if __name__=='__main__':
    main()
