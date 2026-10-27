"""Strict same-crop benchmark contract; no model or image loads."""
from __future__ import annotations
import hashlib
import json
import math
import re
from .identity import canonical_page, canonical_issue, identities, IdentityError
from .g0_receipt import ReceiptError, verify_metadata_receipt, verify_receipt, verify_page_release
from pathlib import Path

EXPECTED_AXEL_ROOT = Path('/Users/marcel/axel')


class ContractError(ValueError):
    pass


def _object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ContractError(f'duplicate JSON key: {key}')
        out[key] = value
    return out


def parse(raw):
    def bad(value):
        raise ContractError(f'non-finite JSON constant: {value}')
    return json.loads(raw, object_pairs_hook=_object, parse_constant=bad)


def load(path):
    return parse(Path(path).read_bytes())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ContractError(message)


def keys(obj, expected, label):
    require(isinstance(obj, dict), f'{label}: expected object')
    require(set(obj) == set(expected.split()), f'{label}: invalid fields')


def string(value):
    return isinstance(value, str) and bool(value.strip())


def sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def reject_gt_assisted(data):
    if isinstance(data, dict):
        require(not data.get('gt_assisted'), 'GT-assisted outputs are diagnostic only')
        require(data.get('role')!='historical_diagnostic', 'historical diagnostic forbidden in benchmark')
        for value in data.values():reject_gt_assisted(value)
    elif isinstance(data, list):
        for value in data:reject_gt_assisted(value)


def _fixture_ok(manifest):
    if manifest.get('exposure')!='fixture':
        return False
    allowlist=load(Path(__file__).resolve().parents[2]/'config/fixtures_allowlist.json')
    return fixture_fingerprint(manifest) in {
        item['canonical_manifest_sha256'] for item in allowlist['fixtures']}


def _canonical_line_keys(line, fixture=False):
    values=[]
    for value, normalizer in ((line['page_id'],canonical_page),
                              (line['document_key'],canonical_issue)):
        try:
            values.append(normalizer(value))
        except IdentityError as exc:
            if not fixture:
                raise ContractError(str(exc)) from exc
            values.append(value)
    page,issue=values
    if page.startswith('gallica:') and issue.startswith('gallica:'):
        require(page.split('/f',1)[0]==issue, 'page/document issue mismatch')
    return page,issue


def _g0_admissibility(manifest, registry):
    require(registry.get('status')=='frozen', 'real benchmark blocked pending frozen C7 registry')
    descriptor=registry.get('g0_audit')
    require(isinstance(descriptor,dict) and set(descriptor)=={'path','sha256'},
            'G0 audit link missing; frozen registry alone cannot authorize a benchmark')
    require(string(descriptor['path']) and Path(descriptor['path']).is_absolute(),
            'G0 audit path must be absolute')
    require(sha(descriptor['sha256']), 'G0 audit SHA-256 missing or invalid')
    try:
        raw=Path(descriptor['path']).read_bytes()
    except OSError as exc:
        raise ContractError('G0 audit file unavailable') from exc
    require(hashlib.sha256(raw).hexdigest()==descriptor['sha256'], 'G0 audit hash mismatch')
    audit=parse(raw)  # Hash and parsing concern the exact same bytes.
    require(isinstance(audit,dict), 'invalid G0 audit role')
    axel_root = EXPECTED_AXEL_ROOT
    try:
        verify_metadata_receipt(audit, expected_axel_root=axel_root)
    except ReceiptError as exc:
        raise ContractError(str(exc)) from exc
    release_descriptor = registry.get('g0_release')
    require(isinstance(release_descriptor, dict) and set(release_descriptor) == {'path', 'sha256'},
            'separate independent G0 release artifact is required')
    release_path = Path(__file__).resolve().parents[2] / 'config/g0_release.json'
    require(release_descriptor.get('path') == str(release_path),
            'G0 release artifact path is not the pinned repository artifact')
    require(sha(release_descriptor.get('sha256')), 'G0 release artifact SHA-256 missing or invalid')
    try:
        release_raw = release_path.read_bytes()
    except OSError as exc:
        raise ContractError('independent G0 release artifact unavailable') from exc
    require(hashlib.sha256(release_raw).hexdigest() == release_descriptor['sha256'],
            'G0 release artifact hash mismatch')
    release_artifact = parse(release_raw)
    require(isinstance(release_artifact, dict), 'invalid independent G0 release artifact')
    try:
        verify_receipt(audit,
                       expected_axel_root=axel_root,
                       release_artifact=release_artifact,
                       receipt_sha256=hashlib.sha256(raw).hexdigest())
    except ReceiptError as exc:
        raise ContractError(str(exc)) from exc
    for line in manifest['lines']:
        page,issue=_canonical_line_keys(line)
        try:
            verify_page_release(audit,page,issue,manifest.get('exposure'))
        except ReceiptError as exc:
            raise ContractError(str(exc)) from exc


def validate_heldout(manifest, registry):
    reject_gt_assisted(manifest)
    exclusions=registry.get('heldout_exclusions', {})
    try:
        pages={canonical_page(v) for v in exclusions.get('page_ids', [])}
        issues={canonical_issue(v) for v in exclusions.get('document_keys', [])}
    except IdentityError as exc:
        raise ContractError(str(exc)) from exc
    unresolved=[]
    for line in manifest['lines']:
        page=None
        issue=None
        try:
            page=canonical_page(line['page_id'])
            if page.startswith('gallica:'):
                issue=page.split('/f',1)[0]
        except IdentityError:
            unresolved.append(line['page_id'])
        require(page not in pages and issue not in issues,
                'historically exposed page/document forbidden in held-out')
        try:
            issue=canonical_issue(line['document_key'])
        except IdentityError:
            unresolved.append(line['document_key'])
        require(issue not in issues, 'historically exposed page/document forbidden in held-out')
    if manifest.get('exposure')=='fixture':
        require(_fixture_ok(manifest), 'fixture manifest not allowlisted: unrecognized or modified identity/content')
        return
    require(not unresolved, 'unrecognized document/page identity or historical issue')
    _g0_admissibility(manifest,registry)


def validate_manifest(data):
    reject_gt_assisted(data)
    keys(data, 'schema_version task dataset_id exposure key_policy coordinate_frame crop_recipe lines', 'manifest')
    require(data['schema_version'] == '0.1.0' and data['task'] == 'same_crops', 'unsupported contract')
    require(string(data['dataset_id']), 'missing dataset identity')
    require(data['key_policy'] in ('fixture_explicit','declared_document_v0'), 'unsupported identity policy')
    require(data['coordinate_frame']=='original_encoded_pixels_top_left_xy', 'unsupported coordinate frame')
    require(data['crop_recipe']=='axis_aligned_xyxy_no_rectification', 'unsupported crop recipe')
    require(data['exposure'] in ('historical', 'development', 'sealed_test', 'fixture'), 'unknown exposure')
    require(isinstance(data['lines'], list) and bool(data['lines']), 'empty benchmark')
    ids, documents, images, crops = set(), {}, {}, {}
    page_identity, image_identity, regions = {}, {}, set()
    for line in data['lines']:
        keys(line, 'id document_key page_id split image_sha256 crop_sha256 image_size box text', 'line')
        require(all(string(line[k]) for k in ('id','document_key','page_id')), 'missing line identity')
        require(line['id'] not in ids, f'duplicate line: {line["id"]}')
        ids.add(line['id'])
        require(line['split'] in ('train','dev','test'), 'invalid split')
        require(data['exposure']!='development' or line['split']!='test','test split under development exposure')
        require(data['exposure']!='sealed_test' or line['split']=='test','sealed test must contain only test')
        require(isinstance(line['text'], str), 'text must be a string')
        require(sha(line['image_sha256']) and sha(line['crop_sha256']), 'invalid content hash')
        size, box = line['image_size'], line['box']
        require(isinstance(size,list) and len(size)==2 and all(type(x) is int and x>0 for x in size), 'invalid image size')
        require(isinstance(box,list) and len(box)==4 and all(number(x) for x in box), 'invalid box')
        require(0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1], 'box outside original scan')
        page_key,document_key=_canonical_line_keys(line,fixture=data['exposure']=='fixture')
        page_meta=(line['image_sha256'],tuple(size),document_key)
        require(page_key not in page_identity or page_identity[page_key]==page_meta,'inconsistent page identity')
        page_identity[page_key]=page_meta
        image_meta=(page_key,document_key)
        require(line['image_sha256'] not in image_identity or image_identity[line['image_sha256']]==image_meta,'image reused under different page/document')
        image_identity[line['image_sha256']]=image_meta
        region=(page_key,tuple(box))
        require(region not in regions,'duplicate page crop region')
        require(line['crop_sha256'] not in crops,'duplicate crop bytes')
        regions.add(region)
        for registry, identity in ((documents,document_key),(images,line['image_sha256']),(crops,line['crop_sha256'])):
            require(identity not in registry or registry[identity]==line['split'], 'cross-split identity/content leak')
            registry[identity] = line['split']
    if data['exposure']=='sealed_test':
        validate_heldout(data,load(Path(__file__).resolve().parents[2]/'config/sealed_registry.json'))
    return data


def validate_predictions(data, manifest, manifest_hash):
    reject_gt_assisted(data)
    keys(data, 'schema_version manifest_sha256 model rows', 'predictions')
    require(data['schema_version']=='0.1.0' and data['manifest_sha256']==manifest_hash, 'wrong benchmark hash')
    required_model=set('name code_revision checkpoint_sha256 config_sha256 alphabet_sha256 license pretraining_audit'.split())
    weights_fields={'weights_license','weights_license_audit_status'}
    require(isinstance(data['model'],dict) and required_model <= set(data['model']) and
            set(data['model']) <= required_model|weights_fields, 'model: invalid fields')
    if manifest['exposure']!='fixture':
        require(weights_fields <= set(data['model']), 'audited weights license provenance required')
        license_value=data['model']['weights_license']
        require(string(license_value) and license_value.strip().casefold() not in
                {'unknown','none','tbd','fixture','pending','unreviewed','unavailable','n/a'},
                'weights license must be explicit and audited')
        require(data['model']['weights_license_audit_status']=='reviewed', 'weights license audit required')
    require(all(string(data['model'][k]) for k in ('name','code_revision','license')), 'missing model provenance')
    require(all(sha(data['model'][k]) for k in ('checkpoint_sha256','config_sha256','alphabet_sha256')), 'invalid model hash')
    require(manifest['exposure']=='fixture' or data['model']['pretraining_audit']=='reviewed', 'pretraining audit required outside fixtures')
    require(data['model']['pretraining_audit'] in ('unknown','reviewed','fixture'), 'invalid pretraining audit')
    require(isinstance(data['rows'],list), 'rows must be list')
    expected={line['id']:line for line in manifest['lines']}
    found={}
    for row in data['rows']:
        keys(row, 'id crop_sha256 status text', 'prediction')
        require(string(row['id']) and row['id'] in expected, 'unknown prediction id')
        require(row['id'] not in found, 'duplicate prediction id')
        require(row['crop_sha256']==expected[row['id']]['crop_sha256'], 'prediction from different crop')
        require(row['status'] in ('ok','error') and isinstance(row['text'],str), 'invalid result')
        require(row['status']=='ok' or row['text']=='', 'failed result must not contain accepted text')
        found[row['id']]=row
    return found


def fixture_fingerprint(manifest):
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def allow_development(manifest, registry_path=None):
    reject_gt_assisted(manifest)
    require(manifest['exposure']!='sealed_test', 'sealed test requires independent release')
    require(all(line['split']=='dev' for line in manifest['lines']), 'evaluation is dev-only; no test/train release')
    registry_path=Path(registry_path) if registry_path else Path(__file__).resolve().parents[2]/'config/sealed_registry.json'
    registry=load(registry_path)
    for line in manifest['lines']:
        require(line['document_key'] not in registry.get('document_keys',[]) and
                line['image_sha256'] not in registry.get('image_sha256',[]) and
                line['crop_sha256'] not in registry.get('crop_sha256',[]),'sealed identity encountered')
    # Keep the pending registry gate early, before touching any real audit path.
    if manifest['exposure']!='fixture':
        require(registry.get('status')=='frozen', 'real benchmark blocked pending frozen C7 registry')
    try:
        protected_issues={canonical_issue(value) for value in registry.get('document_keys',[])}
    except IdentityError as exc:
        raise ContractError(f'unrecognized sealed registry document identity: {exc}') from exc
    for line in manifest['lines']:
        try:
            issue=canonical_issue(line['document_key'])
        except IdentityError:
            issue=None
        require(issue not in protected_issues,'sealed identity encountered through canonical document alias')
    validate_heldout(manifest,registry)
