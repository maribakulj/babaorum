"""Adversarial metadata cases from Claude's independent G0 review."""
import importlib.util
import hashlib
import json
from pathlib import Path
import pytest
from babaorum.identity import canonical_page, identities, IdentityError
from babaorum.contracts import ContractError, validate_heldout

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('g0_inventory', ROOT / 'scripts/g0_inventory.py')
g0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g0)


def row(name, source='read', **kw):
    return dict(page=name, source=source, split='train', excluded=None, **kw)


def seg_manifest(rows):
    return {'pages': rows, 'protected_documents': {}, 'protected_issues': {}}


def inventory(rows, as_rows=None, e12=None, press=None, lm_pretraining_issues=()):
    return g0.build_inventory(seg_manifest(rows), {'pages': as_rows or []}, e12 or [], [], [],
                             press or [], [], {'sets': []}, {},
                             lm_pretraining_issues=lm_pretraining_issues)


def test_e12_kb_dispatch_and_no_none_protection():
    out = inventory([row('kb_00531072', 'kb'), row('kb_00531073', 'kb'), row('bnf_Rouge_001', 'bnf')],
                    e12=[{'page': 'kb_00531072', 'split': 'test'}, {'page': 'kb_00531073', 'split': 'test'}])
    a, b, unknown = out['pages']
    assert a['page_key'] == 'kb:00531072' and b['page_key'] == 'kb:00531073'
    assert a['protected_page_or_issue'] == b['protected_page_or_issue'] == ['E12:test']
    assert unknown['page_key'] is None and not unknown['protected_page_or_issue']
    assert a['issue_key'] is None and 'unresolved_historical_issue_identity' in a['heldout_blockers']


def test_contact_and_missing_metadata_cannot_be_heldout():
    out = inventory([row('kb_00531071', 'kb', contacts=['e7test:kb_00531074:16']),
                     row('bnf_752262-004', 'bnf', title="L'Aurore", contacts=['icdar2017.test:fr_periodical-21:633'])])
    assert not out['candidate_train_pages']
    for item in out['pages']:
        assert item['blind_test_eligible'] is False
        assert 'text_contact_requires_review' in item['heldout_blockers']
    assert 'KB_title_date_concordance_required' in out['pages'][0]['identity_gaps']


def test_as_crosswalk_preserves_gallica_issue_and_training():
    as_rows = [{'page': '18840315_1-0003', 'read_page': 'Le_Matin_0551841_T0000003',
                'title': 'Le Matin', 'split': 'train', 'in_press_train': True},
               {'page': '18840315_1-0004', 'read_page': None, 'title': 'Le Matin', 'split': 'train'}]
    out = inventory([row(x['page'], 'as') for x in as_rows], as_rows=as_rows,
                    e12=[{'page': 'Le_Matin_0551841_T0000007', 'split': 'test'}])
    linked, unlinked = out['pages']
    assert linked['page_key'] == 'gallica:551841/f3' and linked['issue_key'] == 'gallica:551841'
    assert linked['protected_page_or_issue'] == ['AS:train', 'E12:test']
    assert 'E6_E8_press:train' in linked['exposure']
    assert unlinked['issue_key'] is None and unlinked['protected_title']
    assert 'AS_Gallica_crosswalk_missing' in unlinked['identity_gaps']


def test_as_train_pages_are_protected_even_without_candidate_train_allocation():
    as_rows = [{'page': '18840315_1-0003', 'read_page': 'Le_Matin_0551841_T0000003',
                'title': 'Le Matin', 'split': 'train'}]
    out = inventory([row('read_551841-003', title='Le Matin')], as_rows=as_rows)
    item = out['pages'][0]
    assert item['protected_page_or_issue'] == ['AS:train']
    assert 'known_exposed_page_or_issue' in item['heldout_blockers']
    assert out['exposure_sources']['AS']['status'] == 'metadata_loaded'


def test_title_and_training_exposure_are_separate_from_page_overlap():
    out = inventory([row('read_L_OEuvre_1234567_T0000001', title="L'Œuvre")],
                    e12=[{'page': 'L_OEuvre_4618409_T0000008', 'split': 'test'}],
                    press=[{'page': 'L_OEuvre_1234567_T0000001', 'corpus': 'newseye_read', 'split': 'train'}])
    item = out['pages'][0]
    assert not item['protected_page_or_issue'] and item['protected_title']
    assert 'E6_E8_press:train' in item['exposure']
    assert out['counts']['nonexcluded_train_title_seen_in_dev_test'] == 1
    assert out['counts']['nonexcluded_train_in_press_train'] == 1


def test_identity_aliases_and_dot_folios():
    assert canonical_page('kb_00531072') == canonical_page('kb:00531072') == 'kb:00531072'
    for alias in ('bpt6k5121304.f3', 'bpt6k5121304.f3.alto.xml', 'ark:/12148/bpt6k5121304/f3'):
        assert identities(alias) == ('gallica:512130/f3', 'gallica:512130')


@pytest.mark.parametrize('value', ['Rouge_001', '18680715_1-0001', 'as:18680715_1',
                                  'junk-bpt6k5121304.f3', 'bpt6k5121304.f3.bad',
                                  'kb:arbitrary', 'gallica:512130/f0', None])
def test_identity_fails_closed(value):
    with pytest.raises(IdentityError):
        identities(value)


def test_unresolved_real_identity_fails_heldout():
    with pytest.raises(ContractError, match='unrecognized'):
        validate_heldout({'lines': [{'page_id': 'Rouge_001', 'document_key': 'gallica:123456'}]}, {})


def test_source_exclusion_is_not_reported_as_overlap():
    out = inventory([dict(page='photo_unknown', source='read', split='train', excluded='photographies 1983, hors domaine')])
    assert out['counts']['known_overlaps'] == 0
    assert out['counts']['source_excluded_pages'] == 1


def test_unicode_kb_cannot_bypass_ascii_exclusions():
    from babaorum.contracts import validate_heldout
    registry = {'heldout_exclusions': {'page_ids': ['kb:00531072']}}
    for alias in ('kb:00531072', 'kb_00531072', 'kb:００５３１０７２', 'kb_００５３１０７２'):
        with pytest.raises(ContractError):
            validate_heldout({'lines': [{'page_id': alias, 'document_key': alias}]}, registry)
    for invalid in ('kb:００５３１０７２', 'kb_１２３４５６７', 'kb_1234567'):
        with pytest.raises(IdentityError):
            identities(invalid)
    assert canonical_page('read_１２３４５６７-００１') == 'gallica:1234567/f1'


def test_duplicate_or_conflicting_as_crosswalk_is_rejected():
    base = {'page': '18840315_1-0003', 'read_page': 'Le_Matin_0551841_T0000003',
            'title': 'Le Matin', 'split': 'train'}
    with pytest.raises(ValueError, match='duplicate AS page'):
        inventory([], as_rows=[base, dict(base, read_page='Le_Matin_0551841_T0000004')])
    with pytest.raises(ValueError, match='conflicting AS pages'):
        inventory([], as_rows=[base, dict(base, page='18840315_1-0004')])
    with pytest.raises(ValueError, match='inconsistent AS crosswalk'):
        inventory([row(base['page'], 'as', doc='gallica:551841/f4')], as_rows=[base])


def test_source_pool_cannot_be_consumed_as_authorized_split():
    out = inventory([row('read_1234567-001', title='New title')])
    assert out['candidate_train_pages']
    assert all(r['eligible_for_split'] is False for r in out['pages'] + out['candidate_train_pages'])


def test_unresolved_exposure_is_recorded_and_blocks_all_candidates():
    e7 = [dict(page_id='ws_Le_Polonais_1833_p0288', title='Le Polonais', split='test'),
          dict(page_id='bnus_industriel_alsacien_1835', title="L'Industriel alsacien", split='dev')]
    out = g0.build_inventory(seg_manifest([row('read_7000001-002', title='Unexposed')]),
                            {'pages': []}, [], [], e7, [], [], {'sets': []}, {})
    assert out['exposure_identity_complete'] is False
    assert out['counts']['unresolved_exposure_records'] == 2
    assert out['counts']['unresolved_exposures_by_label'] == {'E7_page_metadata:dev': 1, 'E7_page_metadata:test': 1}
    assert [r['name'] for r in out['unresolved_exposures']] == [r['page_id'] for r in e7]
    assert out['split_release_status'] == 'blocked_unresolved_exposures'
    assert 'unresolved_exposure_identities_global' in out['pages'][0]['heldout_blockers']
    assert out['pages'][0]['eligible_for_split'] is False


def test_issue_without_folio_protects_issue_but_does_not_invent_page_one():
    assert g0.known_page('bnf', 'bnf_268644', 'ark:/12148/bpt6k268644s') is None
    out = g0.build_inventory(seg_manifest([row('bnf_268644-003', source='bnf')]),
                            {'pages': []}, [], [],
                            [dict(page_id='bnf_268644', ark='ark:/12148/bpt6k268644s', title='Le Figaro')],
                            [], [], {'sets': []}, {})
    assert out['counts']['unresolved_exposure_records'] == 1
    assert out['pages'][0]['protected_page_or_issue'] == ['E7_page_metadata:selection_unresolved:issue_only']
    assert out['pages'][0]['page_key'] == 'gallica:268644/f3'


def test_explicit_title_alias_and_collection_placeholder():
    out = g0.build_inventory(seg_manifest([row('bnf_461682-001', source='bnf', title='JDPL')]),
                            {'pages': []}, [], [],
                            [dict(page_id='bnf_438752-001', title='Journal des débats')], [], [], {'sets': []}, {})
    item = out['pages'][0]
    assert item['title_key'] == 'journal des debats'
    assert item['protected_title'] == ['E7_page_metadata:selection_unresolved']
    assert not item['protected_page_or_issue']
    assert g0.title_key('presse luxembourgeoise bnl titres non nommes') is None
    assert g0.title_key('Journal de la province de Limbourg') != item['title_key']


def test_quarantine_follows_canonical_key_and_bad_ark_is_unresolved():
    out = inventory([row('kb:00531071', 'kb')])
    assert 'G0_review_B1_KB_quarantine' in out['pages'][0]['heldout_blockers']
    assert g0.known_page('read', 'read_512130-003', 'bpt6k512130') is None


def test_missing_folio_keeps_known_issue_separate_from_unknown_page():
    out = inventory([row('bnf_268644', 'bnf')])
    record = out['pages'][0]
    assert record['page_key'] is None
    assert record['issue_key'] == 'gallica:268644'
    assert 'unresolved_page_identity' in record['heldout_blockers']
    assert record['eligible_for_split'] is False


def ws_fixture(two_sources=False):
    page = {'page_id': 'ws_fixture_p0002', 'split': 'test', 'index': 'Livre:fixture.djvu'}
    credits = [{'ark': 'bpt6k15240480', 'document_key': 'gallica:1524048'}]
    if two_sources:
        credits.append({'ark': 'bpt6k5121304', 'document_key': 'gallica:512130'})
    report = {
        'schema_version': 'babaorum.wikimedia_metadata_investigation.v1',
        'groups': [{'index': page['index'], 'pages': [dict(page)], 'selected_page_count': 1,
                    'splits': {'test': 1}, 'source_documents': credits}],
        'counts': {'input_pages': 1, 'input_indexes': 1, 'declared_gallica_source_pages': 1,
                   'declared_gallica_source_indexes': 1, 'unique_declared_gallica_documents': len(credits)},
    }
    return page, report


def with_ws(rows, page, report):
    return g0.build_inventory(seg_manifest(rows), {'pages': []}, [], [], [page], [], [],
                              {'sets': []}, {}, ws_crosswalk=report)


def test_ws_credit_protects_other_folio_without_resolving_or_releasing_page():
    page, report = ws_fixture()
    rows = [row('gallica:1524048/f999'), row('gallica:7000001/f1')]
    out = with_ws(rows, page, report)
    original = with_ws(rows, page, None)
    assert out['unresolved_exposures'] == original['unresolved_exposures']
    assert out['counts']['unresolved_exposure_records'] == 1
    assert out['pages'][0]['protected_page_or_issue'] == ['E7_Wikisource_source_credit:test']
    assert not out['pages'][1]['protected_page_or_issue']
    assert out['counts']['wikisource_credited_source_documents'] == 1
    assert out['counts']['wikisource_credited_source_pages'] == 1
    assert all(r['eligible_for_split'] is False and r['blind_test_eligible'] is False for r in out['pages'])


def test_ws_montage_protects_union_of_all_credited_sources():
    page, report = ws_fixture(two_sources=True)
    out = with_ws([row('gallica:1524048/f999'), row('gallica:512130/f777')], page, report)
    assert all(r['protected_page_or_issue'] == ['E7_Wikisource_source_credit:test'] for r in out['pages'])
    assert out['counts']['wikisource_credited_source_documents'] == 2
    assert out['counts']['wikisource_credited_source_pages'] == 1


@pytest.mark.parametrize('mutation', ['ark', 'page', 'split', 'index', 'count', 'missing_page'])
def test_ws_crosswalk_inconsistency_is_rejected(mutation):
    page, report = ws_fixture()
    if mutation == 'ark':
        report['groups'][0]['source_documents'][0]['document_key'] = 'gallica:512130'
    elif mutation in ('page', 'split'):
        report['groups'][0]['pages'][0]['page_id' if mutation == 'page' else 'split'] = 'wrong'
    elif mutation == 'index':
        report['groups'][0]['index'] = 'Livre:other.djvu'
    elif mutation == 'count':
        report['counts']['input_pages'] = 2
    else:
        report['groups'] = []
    with pytest.raises(ValueError):
        with_ws([], page, report)


def test_ws_supplement_hashes_bind_exact_report_and_current_page_metadata(tmp_path):
    page, report = ws_fixture()
    pages_path, report_path, config_path = [tmp_path / name for name in ('pages.jsonl', 'report.json', 'config.json')]
    pages_path.write_text(json.dumps(page) + '\n')
    pages_sha = hashlib.sha256(pages_path.read_bytes()).hexdigest()
    report['local_inputs'] = {str(pages_path): pages_sha}
    report_path.write_text(json.dumps(report))
    config = {'schema_version': 'babaorum.g0_metadata_supplements.v2', 'wikisource_crosswalk': {
        'path': str(report_path), 'sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'pages_metadata_path': str(pages_path), 'pages_metadata_sha256': pages_sha}}
    config_path.write_text(json.dumps(config))
    assert g0.load_ws_crosswalk(g0.Inputs(), config_path, pages_path) == report
    report_path.write_text(json.dumps(report) + ' ')
    with pytest.raises(ValueError, match='report SHA-256 mismatch'):
        g0.load_ws_crosswalk(g0.Inputs(), config_path, pages_path)
    report_path.write_text(json.dumps(report))
    pages_path.write_text(json.dumps(dict(page, split='dev')) + '\n')
    with pytest.raises(ValueError, match='pages metadata SHA-256 mismatch'):
        g0.load_ws_crosswalk(g0.Inputs(), config_path, pages_path)
    report_path.unlink()
    with pytest.raises(FileNotFoundError):
        g0.load_ws_crosswalk(g0.Inputs(), config_path, pages_path)


def test_validation_directory_must_exist_and_contain_metadata_filenames(tmp_path):
    with pytest.raises(ValueError, match='missing'):
        g0.Inputs().filenames(tmp_path / 'absent', '*.xml')
    with pytest.raises(ValueError, match='empty'):
        g0.Inputs().filenames(tmp_path, '*.xml')
    (tmp_path / 'fixture.xml').touch()
    inputs = g0.Inputs()
    assert inputs.filenames(tmp_path, '*.xml') == ['fixture']
    assert inputs.current()
    (tmp_path / 'fixture.xml').unlink()
    assert not inputs.current()


@pytest.mark.parametrize('missing', ['protected_documents', 'protected_issues', 'sets'])
def test_required_exposure_maps_cannot_disappear(missing):
    seg, manifest = seg_manifest([]), {'sets': []}
    del (manifest if missing == 'sets' else seg)[missing]
    with pytest.raises(ValueError, match='required'):
        g0.build_inventory(seg, {'pages': []}, [], [], [], [], [], manifest, {})


def test_all_press_corpora_counted_and_unknown_identities_block():
    press = [{'page': '000013', 'corpus': 'bnl_pack', 'split': 'dev', 'line_count': 3},
             {'page': 'kb_00531071', 'corpus': 'other', 'split': 'train', 'line_count': 2}]
    out = inventory([row('kb_00531071', 'kb')], press=press)
    assert out['counts']['press_metadata_lines'] == 5
    assert out['counts']['press_metadata_local_page_identifiers'] == 2
    assert out['counts']['press_metadata_by_corpus_split']['bnl_pack']['dev'] == {
        'local_page_identifiers': 1, 'lines': 3}
    assert out['counts']['unresolved_exposure_records'] == 2
    assert out['exposure_identity_complete'] is False
    assert out['pages'][0]['exposure'] == ['E6_E8_press:train', 'allocated_to_E21_train_manifest_not_proof_of_training']


def test_press_same_issue_blocks_even_without_source_train_allocation():
    candidate = row('gallica:512130/f9')
    candidate['split'] = 'dev'
    out = inventory([candidate], press=[{'page': 'gallica:512130/f3', 'corpus': 'newseye_read', 'split': 'train'}])
    item = out['pages'][0]
    assert item['exposure'] == []
    assert item['press_same_issue'] == ['E6_E8_press:train']
    assert 'press_same_historical_issue_exposure' in item['heldout_blockers']


def test_missing_experiment_exposure_manifests_prevent_complete_inventory():
    out = inventory([row('gallica:7000001/f1')])
    assert out['unresolved_exposures'] == []
    assert out['missing_exposure_sources'] == ['C2', 'E14', 'E15', 'E1_BnL', 'E24']
    assert out['exposure_identity_complete'] is False
    assert out['split_release_status'] == 'blocked_missing_exposure_sources'
    assert 'missing_exposure_metadata_sources_global' in out['pages'][0]['heldout_blockers']


def test_r2_lm_pretraining_issue_blocks_candidate_issue():
    from babaorum.identity import canonical_issue
    issue = canonical_issue('bpt6k5121304')
    out = inventory([row('read_L_Oeuvre_0512130_T0000002', title="L'Œuvre")],
                    lm_pretraining_issues={issue})
    item = out['pages'][0]
    assert item['issue_key'] == issue
    assert 'R2_LM_PleIAs_French_PD:pretraining' in item['protected_page_or_issue']
    assert 'R2_LM_PleIAs_French_PD:pretraining' in item['exposure']
    assert item['pretraining_exposure'] == 'known_R2_LM_PleIAs_French_PD'
    assert 'known_exposed_page_or_issue' in item['heldout_blockers']
    assert out['counts']['r2_lm_pretraining_issue_count'] == 1
    assert out['counts']['r2_lm_pretraining_candidate_overlap_pages'] == 1
