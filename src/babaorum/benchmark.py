"""Same-crop adapter contract: reader-independent cached output evaluation."""
import hashlib
from pathlib import Path
from .contracts import ContractError, parse, validate_manifest, validate_predictions, allow_development


def evaluate(manifest_path, prediction_path):
    manifest_bytes=Path(manifest_path).read_bytes()
    manifest_hash=hashlib.sha256(manifest_bytes).hexdigest()
    manifest = validate_manifest(parse(manifest_bytes))
    allow_development(manifest)
    prediction_bytes=Path(prediction_path).read_bytes()
    prediction=parse(prediction_bytes)
    rows = validate_predictions(prediction, manifest, manifest_hash)
    from .legacy.reader.metric import score, NORMS
    reports = {}
    for split in ('train','dev','test'):
        lines = [line for line in manifest['lines'] if line['split']==split]
        if not lines:
            continue
        missing = sum(line['id'] not in rows for line in lines)
        errors = sum(rows.get(line['id'],{}).get('status')=='error' for line in lines)
        pairs = [(rows.get(line['id'],{}).get('text',''),line['text']) for line in lines]
        measured=score(pairs)
        for mode,normalizer in NORMS.items():
            support=sum(len(normalizer(text)) for _,text in pairs)
            measured['support_chars_'+mode]=support
            if support==0:measured['cer_'+mode]=None
        reports[split] = {**measured, 'missing_predictions':missing, 'failed_predictions':errors,
                          'coverage':(len(lines)-missing-errors)/len(lines),
                          'all_predictions_present_and_successful':missing+errors==0}
    return {'schema_version':'0.1.0','task':'same_crops','manifest_sha256':manifest_hash,
            'predictions_sha256':hashlib.sha256(prediction_bytes).hexdigest(),'model':prediction['model'],
            'exposure':manifest['exposure'],'units':'CER in percent; coverage in [0,1]',
            'metric_source':'axel.reader.metric at pinned provenance/ported.json',
            'limitation':'raw is NFC plus whitespace compaction, not exact diplomatic codepoint CER; no geometry score here',
            'splits':reports}
