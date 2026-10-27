"""PERO process_lines bridge with a fail-closed real-weight provenance gate.
Fixture manifests are allowed only with fake engines defined in this repository's
tests. Real inference requires a lazy factory and audited local artifacts.
API source: DCGM/pero-ocr 4301bdffe9205dff11579dae27b3f80458c27e89.
"""
from __future__ import annotations
import hashlib
import io
import inspect
import json
import math
from pathlib import Path
import subprocess
from ..contracts import ContractError,digest,load,parse,require,validate_manifest,validate_predictions,allow_development
from ..contracts import sha,string


def _fixture_fake_only(engine):
    require(hasattr(engine,'process_lines') and not callable(engine),
            'fixture manifests require a fake engine instance, not a factory/model')
    try:
        source=Path(inspect.getfile(type(engine))).resolve()
        test_root=Path(__file__).resolve().parents[3]/'tests'
        source.relative_to(test_root)
    except (OSError,TypeError,ValueError):
        raise ContractError('fixture inference is restricted to fake engines defined under tests/')


def _code_revision(settings):
    repository=settings.get('local_source_repository')
    require(string(repository),'PERO source checkout path is not frozen')
    path=Path(repository)
    require(path.is_dir(),'PERO source checkout is missing')
    try:
        revision=subprocess.run(['git','-C',str(path),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
        dirty=subprocess.run(['git','-C',str(path),'status','--porcelain'],check=True,capture_output=True,text=True).stdout
    except (OSError,subprocess.CalledProcessError) as exc:
        raise ContractError(f'cannot verify PERO source checkout: {exc}') from exc
    require(not dirty,'PERO source checkout has uncommitted changes')
    return revision


def _alphabet_digest(config_bytes):
    try:
        config=parse(config_bytes)
    except Exception as exc:
        raise ContractError(f'invalid PERO OCR config: {exc}') from exc
    characters=config.get('characters') if isinstance(config,dict) else None
    require(isinstance(characters,list) and bool(characters) and all(isinstance(c,str) for c in characters),
            'PERO OCR config has no valid ordered character list')
    canonical=json.dumps(characters,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
    return hashlib.sha256(canonical).hexdigest()


def _checkpoint_from_config(config_bytes, config_path, settings):
    """Match the pinned PytorchEngineLineOCR resolution, including CPU suffix."""
    require(settings.get('device') == 'cpu', 'PERO device must be explicitly frozen to cpu')
    require(settings.get('checkpoint_resolution') == 'pytorch_cpu_suffix_v1',
            'PERO checkpoint resolution policy is not frozen')
    config = parse(config_bytes)
    require(isinstance(config, dict) and string(config.get('checkpoint')),
            'PERO OCR config has no checkpoint reference')
    path = Path(config['checkpoint'])
    if not path.is_absolute():
        # BaseEngineLineOCR resolves the relative base before the CPU suffix.
        path = (config_path.parent / path).resolve()
    return Path(str(path) + '.cpu').resolve()


def _preflight_real(settings,model):
    """Verify audited metadata and actual small/config and checkpoint bytes before engine creation."""
    require(isinstance(settings,dict),'PERO provenance config must be an object')
    require(isinstance(model,dict),'PERO model provenance must be an object')
    require(settings.get('weights_license_audit_status')=='reviewed',
            'PERO weights license audit is not reviewed')
    require(string(settings.get('weights_license')) and settings['weights_license'].strip().lower()!='unknown',
            'PERO weights license is unknown')
    require(settings['weights_license'].strip().casefold() not in {'none','tbd','fixture'},
            'PERO weights license is a placeholder')
    require(string(settings.get('weights_license_evidence')) and
            settings['weights_license_evidence'].strip().casefold() not in {'unknown','none','tbd','fixture'},
            'PERO weights license evidence is missing')
    require(model.get('weights_license')==settings['weights_license'],
            'model weight license differs from audited PERO weight license')
    require(model.get('weights_license_audit_status')=='reviewed',
            'model weights license audit is not reviewed')
    require(settings.get('code_audit_status')=='reviewed','PERO code audit is not reviewed')
    require(settings.get('config_audit_status')=='reviewed','PERO config audit is not reviewed')
    require(type(settings.get('batch_size')) is int and settings['batch_size']>0,
            'PERO batch size not frozen')

    checkpoint=Path(settings.get('checkpoint_path') or '')
    config_path=Path(settings.get('config_path') or '')
    require(checkpoint.is_file(),'PERO checkpoint path is not a local file')
    require(config_path.is_file(),'PERO OCR config path is not a local file')
    for name in ('checkpoint_sha256','config_sha256','alphabet_sha256'):
        require(sha(settings.get(name)),f'PERO {name} is not frozen')
        require(sha(model.get(name)),f'model {name} is missing or invalid')
        require(model[name]==settings[name],f'model {name} differs from frozen PERO config')
    require(digest(checkpoint)==settings['checkpoint_sha256'],'PERO checkpoint bytes differ from frozen hash')
    config_bytes=config_path.read_bytes()
    require(hashlib.sha256(config_bytes).hexdigest()==settings['config_sha256'],
            'PERO OCR config bytes differ from frozen hash')
    require(_alphabet_digest(config_bytes)==settings['alphabet_sha256'],
            'PERO alphabet in OCR config differs from frozen hash')
    require(_checkpoint_from_config(config_bytes, config_path, settings)==checkpoint.resolve(),
            'PERO OCR config resolves to a different checkpoint than the audited artifact')

    revision=_code_revision(settings)
    require(string(settings.get('code_revision')) and revision==settings['code_revision'],
            'PERO source revision differs from reviewed revision')
    require(model.get('code_revision')==revision,'model code revision differs from audited PERO source')
    require(string(model.get('license')),'PERO code/source license metadata is missing')
    require(model.get('pretraining_audit')=='reviewed','PERO pretraining exposure is not reviewed')
    return {'checkpoint_sha256':settings['checkpoint_sha256'],
            'config_sha256':settings['config_sha256'],
            'alphabet_sha256':settings['alphabet_sha256'],
            'code_revision':revision,
            'checkpoint_path':str(checkpoint.resolve()),
            'device':settings['device']}


def _verify_loaded_engine(engine,expected,settings):
    require(engine.batch_size==settings['batch_size'],'PERO batch size differs from frozen config')
    require(getattr(getattr(engine,'device',None),'type',None)==expected['device'],
            'PERO loaded engine device differs from frozen config')
    actual_checkpoint=getattr(engine,'checkpoint',None)
    require(string(actual_checkpoint) and Path(actual_checkpoint).resolve()==Path(expected['checkpoint_path']),
            'PERO loaded engine checkpoint path differs from audited artifact')
    loaded=getattr(engine,'babaorum_provenance',None)
    require(isinstance(loaded,dict),'PERO engine must expose babaorum_provenance for loaded artifacts')
    require(loaded.get('loader')=='torchscript_bytes_v1',
            'PERO engine was not built by the audited snapshot loader')
    for key,value in expected.items():
        require(loaded.get(key)==value,f'PERO engine loaded {key} differs from audited artifact')


def recognize(engine=None, manifest_path=None, crop_paths=None, model=None):
    require(manifest_path is not None and crop_paths is not None and model is not None,
            'manifest_path, crop_paths and model are required')
    manifest_bytes=Path(manifest_path).read_bytes()
    manifest=validate_manifest(parse(manifest_bytes))
    allow_development(manifest)
    settings=None
    if manifest['exposure']=='fixture':
        _fixture_fake_only(engine)
    else:
        settings=load(Path(__file__).resolve().parents[3]/'config/pero.json')
        expected=_preflight_real(settings,model)
        require(engine is None,
                'real PERO inference accepts no caller-provided engine or factory')
    manifest_hash=hashlib.sha256(manifest_bytes).hexdigest()
    draft={'schema_version':'0.1.0','manifest_sha256':manifest_hash,'model':model,'rows':[]}
    validate_predictions(draft,manifest,manifest_hash)
    if settings is not None:
        # This constructor owns the CPU load hook and hands TorchScript the
        # captured, audited checkpoint bytes. Caller factories are barred.
        from .pero_snapshot import build_engine_from_snapshot
        engine=build_engine_from_snapshot(settings,model)
        # Detect persistent changes during loading. This does not prove that a
        # mutable source remains unchanged after snapshot loading.
        require(_preflight_real(settings,model)==expected,
                'PERO artifacts changed while the engine was loading')
        _verify_loaded_engine(engine,expected,settings)
    require(set(crop_paths)=={x['id'] for x in manifest['lines']},'crop file coverage mismatch')
    from PIL import Image
    import numpy as np
    arrays=[];preprocessing=[]
    for line in manifest['lines']:
        path=Path(crop_paths[line['id']])
        crop_bytes=path.read_bytes()
        require(hashlib.sha256(crop_bytes).hexdigest()==line['crop_sha256'],'crop bytes mismatch')
        with Image.open(io.BytesIO(crop_bytes)) as source:
            width,height=source.size
            target_height=int(engine.line_px_height)
            require(width>0 and height>0 and target_height>0,'invalid dimensions')
            target_width=max(1,round(width*target_height/height))
            # Avoid the silent right-edge crop in upstream process_lines.
            padded=math.ceil(target_width/32)*32+2*int(engine.line_padding_px)
            if engine.model_type=='transformer':
                padded=min(math.ceil(target_width/32)*32,int(engine.max_line_width)+2*int(engine.line_padding_px))+2*int(engine.line_padding_px)
            require(padded<=int(engine.max_input_horizontal_pixels),'PERO would truncate this line; explicit long-line policy required')
            rgb=np.array(source.convert('RGB').resize((target_width,target_height),Image.Resampling.BICUBIC))
            arrays.append(rgb[..., ::-1].copy())
        preprocessing.append({'id':line['id'],'original_crop_size':[width,height],'network_size':[target_width,target_height],
                              'conversion':'BGR; bicubic; aspect ratio retained','geometry':'no word positions exported by this bridge'})
    transcripts,logits,coords=engine.process_lines(arrays,no_logits=True)
    require(len(transcripts)==len(arrays) and len(logits)==len(arrays) and len(coords)==len(arrays),'PERO result cardinality mismatch')
    for line,text in zip(manifest['lines'],transcripts):
        require(text is None or isinstance(text,str),'invalid transcript type')
        draft['rows'].append({'id':line['id'],'crop_sha256':line['crop_sha256'],'status':'error' if text is None else 'ok','text':text or ''})
    require(digest(manifest_path)==manifest_hash,'manifest changed during inference')
    validate_predictions(draft,manifest,manifest_hash)
    return draft,preprocessing
