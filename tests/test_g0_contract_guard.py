"""G0 delivery gates and canonical aliases, with synthetic metadata only."""
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from babaorum.contracts import (ContractError, allow_development, validate_heldout,
                               validate_manifest, validate_predictions)
from babaorum.g0_receipt import ReceiptError, verify_metadata_receipt, verify_page_release
from babaorum.identity import canonical_page, canonical_issue, identities, IdentityError
from babaorum.legacy.data.doc_key import gallica_ark
from test_contracts import manifest, predictions


def real_manifest():
    out = manifest()
    out.update(exposure='development', key_policy='declared_document_v0')
    out['lines'][0].update(page_id='gallica:7000001/f2', document_key='gallica:7000001')
    return out


def audit_record(page_key='gallica:7000001/f2', **changes):
    record = {'source': 'read', 'page': 'read_L_OEuvre_7000001_T0000002',
              'source_split': 'train', 'source_excluded': None,
              'page_key': page_key, 'issue_key': 'gallica:7000001',
              'title': "L'Œuvre", 'title_key': 'l oeuvre', 'title_origin': 'seg_manifest',
              'protected_page_or_issue': [], 'protected_title': [], 'contacts': [],
              'identity_gaps': [], 'press_same_issue': [], 'exposure': [],
              'source_excluded': None, 'eligible_for_split': False,
              'blind_test_eligible': False, 'title_ood_test_eligible': False,
              'heldout_blockers': []}
    record.update(changes)
    return record


def release(tmp_path, records=None, **changes):
    tmp_path.mkdir(parents=True, exist_ok=True)
    records = records if records is not None else [audit_record()]
    axel_root = tmp_path / 'axel'
    babaorum_root = tmp_path / 'babaorum'
    val_dir = tmp_path / 'external' / 'ATR_ValidationSet_BnF_Newseye_M2+'
    val_dir.mkdir(parents=True)
    (val_dir / 'metadata.xml').write_text('<metadata/>')

    def put(path, raw=b'{}'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return str(path)

    source_rows = [{'source': r['source'], 'page': r['page'], 'split': r['source_split'],
                    'excluded': r['source_excluded'], 'title': r['title'],
                    **({'date': r['date']} if 'date' in r else {})} for r in records]
    source_path = axel_root / 'data/prepared/seg/manifest.json'
    source_raw = json.dumps({'pages': source_rows}, ensure_ascii=False).encode()
    put(source_path, source_raw)
    c7_audit = put(axel_root / 'runs/c7/audit.json')
    e12 = put(axel_root / 'runs/e12/pages.json', b'{"pages": []}')
    e22 = put(axel_root / 'runs/e22/bench/runs/e12/pages.json', b'{"pages": []}')
    test_manifest_path = put(axel_root / 'data/prepared/presse_test_manifest.json', b'{"sets": []}')
    e7_dir = axel_root / 'data/prepared/presse_e7'
    e7_bnf = put(e7_dir / 'pages_bnf.jsonl', b'')
    e7_impact = put(e7_dir / 'pages_impact.jsonl', b'')
    e7_pages = put(e7_dir / 'pages.jsonl', b'')
    as_path = put(axel_root / 'data/prepared/olr/newseye_as.json', b'{"pages": []}')
    press_train = put(axel_root / 'data/prepared/press/train.jsonl', b'')
    press_dev = put(axel_root / 'data/prepared/press/dev.jsonl', b'')
    code_paths = []
    for relative in ('scripts/g0_inventory.py', 'src/babaorum/identity.py',
                     'src/babaorum/legacy/data/doc_key.py', 'src/babaorum/contracts.py'):
        path = babaorum_root / relative
        put(path, b'# synthetic test metadata source\n')
        code_paths.append(path)
    config_path = babaorum_root / 'config/g0_metadata_supplements.json'
    crosswalk_path = babaorum_root / 'reports/g0_wikisource_crosswalk_investigation.json'
    e7_pages_hash = hashlib.sha256(Path(e7_pages).read_bytes()).hexdigest()
    crosswalk_raw = json.dumps({'local_inputs': {e7_pages: e7_pages_hash}}).encode()
    put(crosswalk_path, crosswalk_raw)
    lm_manifest_path = tmp_path / 'external' / 'french-pd-sample' / 'manifest.json'
    lm_manifest = {'source': 'PleIAs/French-PD-Newspapers (datasets-server)',
                   'min_ocr': 95, 'seed': 2, 'chars': 58032354,
                   'protected_ids': 370, 'skipped': {'ocr': 378, 'protected': 0},
                   'kept_file_ids': [f'gallica:{700000 + i}' for i in range(322)]}
    lm_manifest_raw = json.dumps(lm_manifest).encode()
    put(lm_manifest_path, lm_manifest_raw)
    config = {'schema_version': 'babaorum.g0_metadata_supplements.v2',
              'wikisource_crosswalk': {'path': str(crosswalk_path), 'pages_metadata_path': e7_pages,
                                       'sha256': hashlib.sha256(crosswalk_raw).hexdigest(),
                                       'pages_metadata_sha256': e7_pages_hash},
              'french_pd_lm_manifest': {'path': str(lm_manifest_path),
                                        'sha256': hashlib.sha256(lm_manifest_raw).hexdigest()}}
    put(config_path, json.dumps(config).encode())
    source_inputs = {
        'C7': [str(source_path), c7_audit], 'E12': [e12], 'E22': [e22],
        'E7': [test_manifest_path, e7_bnf, e7_impact, e7_pages, str(config_path), str(crosswalk_path)],
        'NewsEye_validation': [str(val_dir)], 'AS': [as_path],
        'E6_E8_press': [press_train, press_dev], 'E14': [], 'E15': [], 'E24': [], 'C2': [], 'E1_BnL': [],
        'R2_LM_PleIAs_French_PD': [str(lm_manifest_path)],
    }
    inputs = {}
    for paths in source_inputs.values():
        for name in paths:
            if Path(name).is_file():
                inputs[name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    inputs.update({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths})
    listed_dir = val_dir
    nonexcluded = [r for r in records if r['source_split'] == 'train' and r['source_excluded'] in (None, False, '')]
    audit = {'schema_version': '0.4.0', 'role': 'metadata_inventory', 'metadata_current': True,
             'exposure_identity_complete': False, 'unresolved_exposures': [],
             'split_release_status': 'blocked_missing_exposure_sources',
             'missing_exposure_sources': ['C2', 'E14', 'E15', 'E1_BnL', 'E24'],
             'exposure_sources': {key: {'status': ('missing_metadata_manifest' if key in
                                  {'C2', 'E14', 'E15', 'E24', 'E1_BnL'} else 'metadata_loaded')} for key in
                                  ('C7', 'E12', 'E22', 'E7', 'NewsEye_validation', 'AS',
                                   'E6_E8_press', 'E14', 'E15', 'E24', 'C2', 'E1_BnL',
                                   'R2_LM_PleIAs_French_PD')},
             'created_utc': datetime.now(timezone.utc).isoformat(),
             'axel_head': 'synthetic-head',
             'axel_worktree_before': {'head': 'synthetic-head', 'status_porcelain': '',
                                      'tracked_diff_sha256': 'a' * 64},
             'axel_worktree_after': {'head': 'synthetic-head', 'status_porcelain': '',
                                     'tracked_diff_sha256': 'a' * 64},
             'inputs': inputs,
             'directory_inputs': {str(listed_dir): {'pattern': '*.xml', 'names': ['metadata.xml']}},
             'inventory_source': {'path': str(source_path), 'sha256': inputs[str(source_path)]},
             'source_inputs': source_inputs,
             'counts': {'unresolved_exposure_records': 0, 'unresolved_exposures_by_label': {},
                        'missing_exposure_sources': 5,
                        'seg_manifest_pages': len(records), 'E12_pages': 0, 'E22_independent_pages': 0,
                        'nonexcluded_source_train_pages': len(nonexcluded),
                        'nonexcluded_train_pages_without_known_overlap': sum(
                            not r['protected_page_or_issue'] and not r['contacts'] for r in nonexcluded),
                        'known_overlaps': sum(bool(r['protected_page_or_issue']) for r in records),
                        'source_excluded_pages': sum(r['source_excluded'] not in (None, False, '') for r in records),
                        'text_contact_pages': sum(bool(r['contacts']) for r in records),
                        'missing_page_identity': sum(r['page_key'] is None for r in records),
                        'missing_issue_identity': sum(r['issue_key'] is None for r in records),
                        'nonexcluded_train_missing_title': sum(not r.get('title_key') for r in nonexcluded),
                        'nonexcluded_train_title_seen_in_dev_test': sum(bool(r['protected_title']) for r in nonexcluded),
                        'nonexcluded_train_in_press_train': sum('E6_E8_press:train' in r['exposure'] for r in nonexcluded),
                        'r2_lm_pretraining_issue_count': 322,
                        'r2_lm_pretraining_candidate_overlap_pages': 0,
                        'blind_test_eligible': sum(r['blind_test_eligible'] is True for r in records),
                        'title_ood_test_eligible': sum(r['title_ood_test_eligible'] is True for r in records)},
             'pages': records, 'protected_titles': {}}
    audit.update(changes)
    raw = json.dumps(audit, ensure_ascii=False).encode()
    audit_path = tmp_path / 'audit.json'
    audit_path.write_bytes(raw)
    registry = {'status': 'frozen', 'document_keys': [], 'image_sha256': [], 'crop_sha256': [],
                'g0_audit': {'path': str(audit_path), 'sha256': hashlib.sha256(raw).hexdigest()}}
    registry_path = tmp_path / 'registry.json'
    registry_path.write_text(json.dumps(registry))
    return registry, registry_path, audit_path


def rewrite_receipt(registry, registry_path, audit_path, mutate):
    audit = json.loads(audit_path.read_text())
    mutate(audit)
    raw = json.dumps(audit, ensure_ascii=False).encode()
    audit_path.write_bytes(raw)
    registry['g0_audit']['sha256'] = hashlib.sha256(raw).hexdigest()
    registry_path.write_text(json.dumps(registry))


def read_receipt(audit_path):
    return json.loads(audit_path.read_text())


def test_frozen_status_alone_cannot_release_real_benchmark(tmp_path):
    registry = {'status': 'frozen', 'document_keys': [], 'image_sha256': [], 'crop_sha256': []}
    path = tmp_path / 'registry.json'
    path.write_text(json.dumps(registry))
    with pytest.raises(ContractError, match='G0 audit link'):
        allow_development(real_manifest(), path)
    for page in ('kb_00531071', 'kb_00531074', 'gallica:268644'):
        data = real_manifest()
        data['lines'][0].update(page_id=page, document_key=page)
        with pytest.raises(ContractError):
            validate_heldout(data, registry)


def test_metadata_receipt_can_be_current_without_authorizing_a_release(tmp_path):
    registry, path, _ = release(tmp_path)
    _, _, audit_path = release(tmp_path / 'second')
    verify_metadata_receipt(read_receipt(audit_path))
    with pytest.raises(ContractError, match='G0'):
        validate_heldout(real_manifest(), registry)
    with pytest.raises(ReceiptError, match='not explicitly authorized'):
        from babaorum.g0_receipt import verify_receipt
        verify_receipt(read_receipt(audit_path))


@pytest.mark.parametrize('field,value', [
    ('protected_titles', {'le figaro': ['E7:test']}),
    ('unresolved_exposures', []),
    ('exposure_sources', {'E7': {'status': 'missing_metadata_manifest'}}),
    ('counts', {'blind_test_eligible': 1}),
])
def test_recomputed_inventory_must_match_every_derived_exposure_claim(field, value):
    from babaorum.g0_receipt import _compare_derived_fields
    receipt = {'protected_titles': {}, 'unresolved_exposures': [{'name': 'page'}],
               'exposure_sources': {'E7': {'status': 'metadata_loaded'}}, 'counts': {'blind_test_eligible': 0},
               'pages': [{'protected_page_or_issue': [], 'protected_title': [], 'contacts': [],
                          'press_same_issue': [], 'exposure': []}]}
    recomputed = json.loads(json.dumps(receipt))
    receipt[field] = value
    with pytest.raises(ReceiptError, match='derived evidence differs'):
        _compare_derived_fields(receipt, recomputed)


@pytest.mark.parametrize('records,changes,match', [
    ([], {}, 'pages missing or empty'),
    ([audit_record(), audit_record(page_key='gallica:7000001/f2')], {}, 'exactly one'),
    ([audit_record(eligible_for_split=False)], {}, 'not released'),
    ([audit_record(eligible_for_split=1)], {}, 'not released'),
    ([audit_record(heldout_blockers=['quarantine'])], {}, 'held-out blockers'),
    ([audit_record(title_key=None)], {}, 'title identity'),
    ([audit_record(issue_key=None)], {}, 'issue identity'),
    ([audit_record(issue_key='gallica:7000002')], {}, 'issue does not match'),
    (None, {'metadata_current': False}, 'not current'),
    (None, {'exposure_identity_complete': True}, 'completeness flag'),
    (None, {'unresolved_exposures': [{'page': 'ws_unresolved'}]}, 'split status'),
])
def test_g0_absence_ambiguity_blockers_and_unresolved_exposure_refused(tmp_path, records, changes, match):
    registry, _, _ = release(tmp_path, records, **changes)
    audit = json.loads(registry['g0_audit']['path'] and Path(registry['g0_audit']['path']).read_text())
    if records is not None and not changes:
        page = records[0] if records else None
        if page is not None:
            if type(page.get('eligible_for_split')) is not bool:
                with pytest.raises(ReceiptError, match='explicit boolean'):
                    verify_metadata_receipt(audit)
                return
            if match != 'not released':
                audit['pages'][0]['eligible_for_split'] = True
            with pytest.raises(ReceiptError, match=match):
                verify_page_release(audit, 'gallica:7000001/f2', 'gallica:7000001', 'development')
            return
    if changes.get('metadata_current') is False:
        verify_metadata_receipt(audit)  # freshness validation intentionally does not authorize or require current=true
        with pytest.raises(ReceiptError, match='not current'):
            from babaorum.g0_receipt import verify_receipt
            verify_receipt(audit)
        return
    with pytest.raises(ReceiptError, match=match):
        verify_metadata_receipt(audit)


def test_audit_hash_covers_bytes_actually_parsed(tmp_path):
    registry, _, path = release(tmp_path)
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ContractError, match='hash mismatch'):
        validate_heldout(real_manifest(), registry)


def test_receipt_rehashes_input_files_and_directory_members_at_call_time(tmp_path):
    registry, registry_path, audit_path = release(tmp_path)
    receipt = json.loads(audit_path.read_text())
    Path(receipt['inventory_source']['path']).write_text('{"pages":[]}')
    with pytest.raises(ReceiptError, match='input hash changed'):
        verify_metadata_receipt(receipt)

    registry, registry_path, audit_path = release(tmp_path / 'dir-case')
    directory = Path(next(iter(json.loads(audit_path.read_text())['directory_inputs'])))
    (directory / 'unexpected.xml').write_text('<metadata/>')
    with pytest.raises(ReceiptError, match='directory listing changed'):
        verify_metadata_receipt(read_receipt(audit_path))


def test_inventory_truncation_cannot_be_hidden_by_rewriting_counts(tmp_path):
    records = [audit_record(), audit_record(page_key='gallica:7000001/f3',
                                           page='read_L_OEuvre_7000001_T0000003')]
    registry, registry_path, audit_path = release(tmp_path, records)
    rewrite_receipt(registry, registry_path, audit_path,
                    lambda r: (r['pages'].pop(), r['counts'].__setitem__('seg_manifest_pages', 1)))
    with pytest.raises(ReceiptError, match='multiset'):
        verify_metadata_receipt(read_receipt(audit_path))


@pytest.mark.parametrize('field,value', [
    ('protected_page_or_issue', ['E7:test']),
    ('protected_title', ['E7:test']),
    ('contacts', ['AS:test']),
    ('identity_gaps', ['unresolved']),
    ('press_same_issue', ['E6_E8_press:train']),
    ('exposure', ['E21:train']),
    ('source_excluded', True),
])
def test_release_flags_do_not_override_nonempty_page_evidence(tmp_path, field, value):
    registry, _, _ = release(tmp_path, [audit_record(**{field: value})])
    receipt = read_receipt(release(tmp_path / field, [audit_record(**{field: value})])[2])
    # Metadata verification permits blocked pages but requires their facts to be explicit.
    with pytest.raises(ReceiptError, match='explicitly empty|source-excluded'):
        verify_page_release(receipt, 'gallica:7000001/f2', 'gallica:7000001', 'development')


@pytest.mark.parametrize('timestamp,match', [
    ('2020-01-01T00:00:00Z', 'predates'),
    ('2999-01-01T00:00:00Z', 'future'),
    ('2026-10-07T12:00:00', 'timezone'),
])
def test_receipt_creation_date_is_tz_aware_fresh_and_not_future(tmp_path, timestamp, match):
    registry, registry_path, audit_path = release(tmp_path)
    rewrite_receipt(registry, registry_path, audit_path,
                    lambda r: r.__setitem__('created_utc', timestamp))
    with pytest.raises(ReceiptError, match=match):
        verify_metadata_receipt(read_receipt(audit_path))


def test_receipt_requires_explicit_empty_catalogue_and_release_state(tmp_path):
    registry, registry_path, audit_path = release(tmp_path)
    for key, value in (('missing_exposure_sources', ['E14']),
                       ('split_release_status', 'pending_independent_audit'),
                       ('exposure_identity_complete', True)):
        rewrite_receipt(registry, registry_path, audit_path,
                        lambda r, key=key, value=value: r.__setitem__(key, value))
        with pytest.raises(ReceiptError):
            verify_metadata_receipt(read_receipt(audit_path))
        registry, registry_path, audit_path = release(tmp_path / key.replace('/', '_'))


def test_source_role_omission_and_title_protection_cannot_be_hidden(tmp_path):
    registry, registry_path, audit_path = release(tmp_path)
    rewrite_receipt(registry, registry_path, audit_path,
                    lambda r: r['source_inputs']['E7'].pop())
    with pytest.raises(ContractError, match='file list incomplete'):
        validate_heldout(real_manifest(), registry)

    record = audit_record(eligible_for_split=True)
    registry, registry_path, audit_path = release(tmp_path / 'title', [record])
    rewrite_receipt(registry, registry_path, audit_path,
                    lambda r: r['protected_titles'].__setitem__('l oeuvre', ['E7:test']))
    with pytest.raises(ReceiptError, match='protected title registry'):
        verify_metadata_receipt(read_receipt(audit_path))


def test_inventory_title_and_date_must_agree_with_source_metadata(tmp_path):
    record = audit_record()
    record['date'] = '1937'
    registry, registry_path, audit_path = release(tmp_path, [record])
    rewrite_receipt(registry, registry_path, audit_path,
                    lambda r: r['pages'][0].__setitem__('date', '1938'))
    with pytest.raises(ContractError, match='outside expected Axel root'):
        validate_heldout(real_manifest(), registry)


def test_g0_admission_uses_pinned_axel_root_even_if_repository_is_copied():
    import babaorum.contracts as contracts
    assert contracts.EXPECTED_AXEL_ROOT == Path('/Users/marcel/axel')


def test_canonical_page_key_must_match_source_page_metadata(tmp_path):
    registry, registry_path, audit_path = release(tmp_path)
    rewrite_receipt(registry, registry_path, audit_path,
                    lambda r: r['pages'][0].__setitem__('page_key', 'gallica:7000001/f1'))
    with pytest.raises(ContractError, match='outside expected Axel root'):
        validate_heldout(real_manifest(), registry)


def test_sealed_manifest_uses_same_g0_gate(tmp_path, monkeypatch):
    import babaorum.contracts as contracts
    registry, _, _ = release(tmp_path, [audit_record(eligible_for_split=False)])
    original = contracts.load
    monkeypatch.setattr(contracts, 'load', lambda path: registry if Path(path).name=='sealed_registry.json' else original(path))
    data = real_manifest()
    data['exposure'] = 'sealed_test'
    data['lines'][0]['split'] = 'test'
    with pytest.raises(ContractError, match='G0'):
        validate_manifest(data)


def test_canonical_aliases_detect_cross_split_and_duplicate_region():
    data = real_manifest()
    data['exposure'] = 'historical'
    first = data['lines'][0]
    first['split'] = 'train'
    alias = copy.deepcopy(first)
    alias.update(id='b', page_id=gallica_ark('7000001')+'.f2', document_key=gallica_ark('7000001'),
                 split='test', crop_sha256='c'*64, box=[40,0,80,20])
    data['lines'].append(alias)
    with pytest.raises(ContractError, match='cross-split'):
        validate_manifest(data)
    alias.update(split='train', box=first['box'])
    with pytest.raises(ContractError, match='duplicate page crop region'):
        validate_manifest(data)


def test_registry_document_protection_recognizes_ark_alias(tmp_path):
    registry, path, _ = release(tmp_path)
    registry['document_keys'] = ['gallica:7000001']
    path.write_text(json.dumps(registry))
    data = real_manifest()
    data['lines'][0]['document_key'] = gallica_ark('7000001')
    with pytest.raises(ContractError, match='sealed identity'):
        allow_development(data, path)


def test_page_and_issue_are_distinct_and_ark_checksum_is_strict():
    ark = gallica_ark('512130')
    for value in ('gallica:512130', ark, 'bnf_512130'):
        assert canonical_issue(value)=='gallica:512130'
        with pytest.raises(IdentityError):
            canonical_page(value)
    assert canonical_page(ark+'.f3')=='gallica:512130/f3'
    assert canonical_issue(ark+'.f3')=='gallica:512130'
    assert identities('kb:00531072')==('kb:00531072',None)
    with pytest.raises(IdentityError):
        canonical_issue('kb:00531072')
    for bad in (ark[:-1]+('b' if ark[-1]!='b' else 'c'), 'bpt6k512130'):
        with pytest.raises(IdentityError):
            canonical_page(bad+'.f3')
        with pytest.raises(IdentityError):
            canonical_issue(bad)


def test_nonfixture_prediction_requires_audited_weights_license():
    data = predictions()
    data['model']['pretraining_audit'] = 'reviewed'
    with pytest.raises(ContractError, match='weights license'):
        validate_predictions(data, real_manifest(), 'c'*64)
    data['model'].update(weights_license='unknown', weights_license_audit_status='reviewed')
    with pytest.raises(ContractError, match='weights license'):
        validate_predictions(data, real_manifest(), 'c'*64)
    data['model'].update(weights_license='CC-BY-4.0', weights_license_audit_status='pending')
    with pytest.raises(ContractError, match='weights license'):
        validate_predictions(data, real_manifest(), 'c'*64)
    data['model']['weights_license_audit_status']='reviewed'
    validate_predictions(data, real_manifest(), 'c'*64)


@pytest.mark.parametrize('blind_flag', [False, None, 1])
def test_sealed_test_requires_explicit_blind_release(tmp_path, blind_flag):
    registry, _, _ = release(tmp_path, [audit_record(eligible_for_split=True, blind_test_eligible=blind_flag)])
    data = real_manifest()
    data['exposure'] = 'sealed_test'
    data['lines'][0]['split'] = 'test'
    with pytest.raises(ContractError, match='G0'):
        validate_heldout(data, registry)
    registry, _, _ = release(tmp_path / 'otherwise-clean', [audit_record(eligible_for_split=True,
                                                                          blind_test_eligible=True)])
    with pytest.raises(ContractError, match='G0'):
        validate_heldout(data, registry)
