"""Conservative G0 inventory of metadata, never an authorization to split a test.

Reads manifests and filenames; never opens PAGE/ALTO XML, images, or test GT.
Existing contact counts are evidence inherited from C7, not recomputed here.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from babaorum.legacy.data.doc_key import doc_key
from babaorum.contracts import digest
from babaorum.evidence import atomic_write
from babaorum.identity import canonical_page, canonical_issue, IdentityError


def title_key(value):
    value = (value or '').casefold().replace('œ', 'oe').replace('æ', 'ae')
    value = ''.join(c for c in unicodedata.normalize('NFKD', value) if not unicodedata.combining(c))
    key = ' '.join(re.findall(r'[a-z0-9]+', value)) or None
    # Explicit alias from the BnF manifest, confirmed by E7 issue 438752 metadata.
    # Not fuzzy matching: ambiguous/unknown titles remain unaudited.
    if key == 'jdpl':
        return 'journal des debats'
    if key in {'presse luxembourgeoise bnl titres non nommes', 'titres non nommes'}:
        return None
    return key


def known_page(source, name, ark=None):
    """Strict page identity; unresolved exposure is recorded by expose_record.

    A number/ARK without an explicit folio must never manufacture page f1.
    C7 remains pinned as historical provenance, not as a permissive fallback.
    """
    if source in ('e12', 'e22'):
        source = 'kb' if name.startswith(('kb_', 'kb:')) else 'read'
    if source not in ('kb', 'read', 'bnf', 'as', 'readval'):
        return None
    try:
        page = canonical_page(name)
        if ark and canonical_issue(ark) != real_issue(page):
            return None
        return page
    except IdentityError:
        return None


def real_issue(page):
    # C7's kb:<page> identifies a page, NOT a historical newspaper issue.
    return page.split('/f', 1)[0] if page and page.startswith('gallica:') else None


def known_issue(source, name, ark=None):
    """Keep an explicitly known issue even when its page folio is absent."""
    page = known_page(source, name, ark)
    if page:
        return real_issue(page)
    if source == 'kb' or str(name).startswith(('kb:', 'kb_')):
        return None
    try:
        return canonical_issue(ark or name)
    except IdentityError:
        return None


def title_from_read_name(name):
    match = re.fullmatch(r'(?:read_)?(.+)_\d{6,7}_[DT]\d+', name)
    return match[1].replace('_', ' ') if match else None


def wikisource_protections(crosswalk, e7_metadata):
    """Union credited source documents, never infer a Gallica page from a DJVU view."""
    if crosswalk is None:
        return {'documents': [], 'credited_pages': 0, 'credited_indexes': 0}
    if crosswalk.get('schema_version') != 'babaorum.wikimedia_metadata_investigation.v1':
        raise ValueError('unsupported Wikisource crosswalk schema')
    local = {}
    for row in e7_metadata:
        if row.get('page_id', '').startswith('ws_'):
            key = row['page_id']
            if key in local:
                raise ValueError(f'duplicate Wikisource metadata page: {key}')
            local[key] = (row['split'], row['index'])
    seen_pages, seen_indexes, credited_pages = set(), set(), set()
    documents = {}
    credited_indexes = 0
    for group in crosswalk['groups']:
        index = group['index']
        if not isinstance(index, str) or not index or index in seen_indexes:
            raise ValueError('missing or duplicate Wikisource crosswalk index')
        seen_indexes.add(index)
        if not group['pages'] or group['selected_page_count'] != len(group['pages']):
            raise ValueError(f'Wikisource selected page count mismatch: {index}')
        if group['splits'] != dict(Counter(page['split'] for page in group['pages'])):
            raise ValueError(f'Wikisource split count mismatch: {index}')
        for page in group['pages']:
            key = page['page_id']
            if key in seen_pages or local.get(key) != (page['split'], index):
                raise ValueError(f'Wikisource page/split/index mismatch: {key}')
            seen_pages.add(key)
        credited = set()
        for credit in group['source_documents']:
            issue = canonical_issue(credit['ark'])
            if issue != credit['document_key'] or not issue.startswith('gallica:'):
                raise ValueError(f'Wikisource ARK/document mismatch: {credit["ark"]}')
            credited.add(issue)
        if credited:
            credited_indexes += 1
            credited_pages.update(page['page_id'] for page in group['pages'])
        for issue in credited:
            entry = documents.setdefault(issue, {'indexes': set(), 'pages': set(), 'labels': set()})
            entry['indexes'].add(index)
            entry['pages'].update(page['page_id'] for page in group['pages'])
            entry['labels'].update('E7_Wikisource_source_credit:' + page['split'] for page in group['pages'])
    if seen_pages != set(local):
        raise ValueError('Wikisource crosswalk does not cover the current metadata pages exactly')
    measured = {'input_pages': len(local), 'input_indexes': len(seen_indexes),
                'declared_gallica_source_pages': len(credited_pages),
                'declared_gallica_source_indexes': credited_indexes,
                'unique_declared_gallica_documents': len(documents)}
    if any(crosswalk['counts'].get(key) != value for key, value in measured.items()):
        raise ValueError('Wikisource crosswalk aggregate counts mismatch')
    return {'credited_pages': len(credited_pages), 'credited_indexes': credited_indexes,
            'documents': [dict(document_key=issue, indexes=sorted(entry['indexes']),
                               credited_page_count=len(entry['pages']), labels=sorted(entry['labels']))
                          for issue, entry in sorted(documents.items())]}


def build_inventory(seg, as_manifest, e12, e22, e7_metadata, press_metadata,
                    readval_names, test_manifest, c7_audit, ws_crosswalk=None,
                    lm_pretraining_issues=()):
    for key in ('protected_documents', 'protected_issues'):
        if key not in seg or not isinstance(seg[key], dict):
            raise ValueError(f'required segmentation protection map missing or invalid: {key}')
    if 'sets' not in test_manifest or not isinstance(test_manifest['sets'], list):
        raise ValueError('required E7 exposure sets missing or invalid')
    ws_protections = wikisource_protections(ws_crosswalk, e7_metadata)
    # These experiments are known exposure sources, not audited by the manifests
    # currently wired into this inventory. Missing is never treated as empty.
    exposure_sources = {name: {'status': 'metadata_loaded'} for name in
                        ('C7', 'E12', 'E22', 'E7', 'NewsEye_validation', 'AS',
                         'E6_E8_press', 'R2_LM_PleIAs_French_PD')}
    # E1 BnL manifests are known to exist but remain unintegrated: local page
    # IDs lack a reviewed bridge to the candidate corpus. Missing here means
    # “not represented in this auditable inventory”, never “no exposure”.
    missing_exposure_sources = sorted(('E14', 'E15', 'E24', 'C2', 'E1_BnL'))
    exposure_sources.update({name: {'status': 'missing_metadata_manifest'} for name in missing_exposure_sources})
    as_pages = {}
    mapped_as_pages = {}
    for row in as_manifest['pages']:
        if row['page'] in as_pages:
            raise ValueError(f"duplicate AS page in crosswalk: {row['page']}")
        if row.get('read_page'):
            key = known_page('as', row['read_page'])
            if key is None:
                raise ValueError(f"unrecognized AS READ crosswalk: {row['page']}")
            if key in mapped_as_pages:
                raise ValueError(f"conflicting AS pages for READ identity: {key}")
            mapped_as_pages[key] = row['page']
        as_pages[row['page']] = row
    for row in seg['pages']:
        if row['source'] == 'as' and row.get('doc'):
            matched = as_pages.get(row['page'], {}).get('read_page')
            key = known_page('as', matched) if matched else None
            if canonical_page(row['doc']) != key:
                raise ValueError(f"inconsistent AS crosswalk and source identity: {row['page']}")
    exposed_pages, exposed_issues, exposed_titles = defaultdict(set), defaultdict(set), defaultdict(set)
    lm_pretraining_issues = set(lm_pretraining_issues)
    if any(not isinstance(issue, str) or not issue.startswith('gallica:') for issue in lm_pretraining_issues):
        raise ValueError('LM pretraining issue identities must be canonical Gallica issue keys')
    for issue in lm_pretraining_issues:
        exposed_issues[issue].add('R2_LM_PleIAs_French_PD:pretraining')
    metadata_titles = defaultdict(set)
    unresolved_exposures = []

    def expose(page, title, label):
        if page:
            exposed_pages[page].add(label)
            issue = real_issue(page)
            if issue:
                exposed_issues[issue].add(label)
        if title_key(title):
            exposed_titles[title_key(title)].add(label)

    def expose_record(source, name, title, label, ark=None, metadata_source=None):
        page = known_page(source, name, ark)
        if page is None:
            # Keep the unresolved contact itself; a title alone cannot establish
            # cross-source disjointness (Wikisource DJVU may originate at Gallica).
            unresolved_exposures.append({
                'source': source, 'name': name, 'label': label,
                'metadata_source': metadata_source, 'ark': ark,
                'title_key': title_key(title), 'reason': 'unresolved_exposure_page_identity',
            })
            # An explicit number can still protect every page of that issue.
            try:
                issue = canonical_issue(ark or name)
            except IdentityError:
                issue = None
            if issue and issue.startswith('gallica:'):
                exposed_issues[issue].add(label + ':issue_only')
        expose(page, title, label)
        remember_title(page, title)
        return page

    def remember_title(page, title):
        if page and title_key(title):
            metadata_titles[real_issue(page) or page].add(title)

    # C7 inherited protections remain explicit; independent metadata checks follow.
    for page, why in seg['protected_documents'].items():
        expose(canonical_page(page), None, 'C7_document:' + why)
    for gid, why in seg['protected_issues'].items():
        exposed_issues['gallica:' + str(int(gid))].add('C7_issue:' + why)
    for entry in ws_protections['documents']:
        exposed_issues[entry['document_key']].update(entry['labels'])
    for rows, label in ((e12, 'E12'), (e22, 'E22_independent')):
        for row in rows:
            title = row.get('title') or title_from_read_name(row['page'])
            expose_record('e12', row['page'], title, label + ':' + row.get('split', 'test'), row.get('ark'))
    for name in readval_names:
        expose_record('readval', name, title_from_read_name(name), 'NewsEye_validation_test')
    for row in e7_metadata:
        name = row.get('page_id', '')
        source = 'kb' if name.startswith('kb_') else 'bnf'

        # Metadata page tables sometimes predate final E7 selection. Conservatively
        # mark them exposed, without asserting they all reached the final test.
        expose_record(source, name, row.get('title'), 'E7_page_metadata:' + row.get('split', 'selection_unresolved'),
                      row.get('ark'), row.get('_metadata_source'))
    for dataset in test_manifest['sets']:
        for title in dataset.get('titles', []):
            expose(None, title, 'E7_test_manifest:' + dataset['name'])
    for row in seg['pages']:
        page = known_page(row['source'], row['page'], row.get('ark'))
        if row['source'] == 'as':
            match = as_pages.get(row['page'], {}).get('read_page')
            page = known_page('as', match) if match else None
        remember_title(page, row.get('title'))
    # All AS splits are exposure: train is used to fit downstream segmentation
    # models, while dev/test have already informed selection and evaluation.
    # Uncrosswalked PAGE names remain unresolved and block release globally.
    for row in as_manifest['pages']:
        page = known_page('as', row['read_page']) if row.get('read_page') else None
        expose_record('as', row.get('read_page') or row['page'], row.get('title'), 'AS:' + row['split'])

    # Reconcile explicit titles across aliases of one issue, never across dates alone.
    for issue, labels in exposed_issues.items():
        for title in metadata_titles.get(issue, ()):
            exposed_titles[title_key(title)].update(labels)
    press_pages, press_issues = defaultdict(set), defaultdict(set)
    press_counts = {}
    for row in press_metadata:
        corpus, split = row['corpus'], row['split']
        counts = press_counts.setdefault(corpus, {}).setdefault(split, {'local_page_identifiers': 0, 'lines': 0})
        counts['local_page_identifiers'] += 1
        counts['lines'] += row.get('line_count', 1)
        page = known_page('read', row['page'])
        label = 'E6_E8_press:' + split
        if page is None or real_issue(page) is None:
            unresolved_exposures.append({
                'source': 'press:' + corpus, 'name': row['page'], 'label': label,
                'metadata_source': row.get('_metadata_source'), 'ark': None, 'title_key': None,
                'line_count': row.get('line_count', 1),
                'reason': 'unresolved_press_page_or_historical_issue_identity',
            })
            issue = known_issue('read', row['page'])
            if issue:
                press_issues[issue].add(label)
        if page:
            press_pages[page].add(label)
            if real_issue(page):
                press_issues[real_issue(page)].add(label)

    rows = []
    for item in seg['pages']:
        source, name = item['source'], item['page']
        match = as_pages.get(name, {}) if source == 'as' else {}
        page = known_page(source, name, item.get('ark'))
        if source == 'as':
            page = known_page('as', match['read_page']) if match.get('read_page') else None
        issue = real_issue(page) or known_issue(source, match.get('read_page') or name, item.get('ark'))
        title = item.get('title') or match.get('title') or title_from_read_name(name)
        titles = metadata_titles.get(issue or page, set()) if page else set()
        title_origin = 'source_manifest_or_page_name' if title else None
        if not title and len({title_key(t) for t in titles}) == 1:
            title = sorted(titles)[0]
            title_origin = 'same_canonical_issue_metadata'
        title_id = title_key(title)
        contacts = list(item.get('contacts') or [])
        contacts += ['AS_other:' + str(x) for x in match.get('touches_other', [])]
        exposure = []
        if item['split'] == 'train' and not item.get('excluded'):
            exposure.append('allocated_to_E21_train_manifest_not_proof_of_training')
        exposure += sorted(press_pages.get(page, set())) if page else []
        if issue in lm_pretraining_issues:
            exposure.append('R2_LM_PleIAs_French_PD:pretraining')
        if match.get('in_press_train'):
            exposure.append('E6_E8_press:train')
        if match.get('touches_press_train'):
            exposure.append('AS_contact_with_E6_E8_press_train')
        protection = sorted(exposed_pages.get(page, set()) | exposed_issues.get(issue, set()))
        title_contacts = sorted(exposed_titles.get(title_id, set())) if title_id else []
        issues = []
        if not page:
            issues.append('unresolved_page_identity')
        if not issue:
            issues.append('unresolved_historical_issue_identity')
        if not title_id:
            issues.append('missing_title')
        if len({title_key(t) for t in titles}) > 1:
            issues.append('conflicting_titles_for_issue')
        if source == 'as' and not match.get('read_page'):
            issues.append('AS_Gallica_crosswalk_missing')
        if source == 'kb':
            issues.append('KB_title_date_concordance_required')
        reasons = list(issues)
        if protection:
            reasons.append('known_exposed_page_or_issue')
        if title_contacts:
            reasons.append('title_seen_in_existing_dev_or_test')
        if contacts:
            reasons.append('text_contact_requires_review')
        if item.get('excluded'):
            reasons.append('source_excluded')
        if exposure:
            reasons.append('known_training_allocation_or_exposure')
        if press_issues.get(issue):
            reasons.append('press_same_historical_issue_exposure')
        if page in ('kb:00531071', 'kb:00531087', 'kb:00531166', 'kb:00531192'):
            reasons.append('G0_review_B1_KB_quarantine')
        if unresolved_exposures:
            reasons.append('unresolved_exposure_identities_global')
        if missing_exposure_sources:
            reasons.append('missing_exposure_metadata_sources_global')
        reasons += ['model_pretraining_audit_pending', 'GT_and_image_duplicate_audit_pending',
                    'independent_split_release_pending']
        rows.append({
            'source': source, 'page': name, 'source_split': item['split'],
            'page_key': page, 'issue_key': issue,
            'source_local_issue': item.get('issue'), 'title': title, 'title_key': title_id,
            'title_origin': title_origin, 'date': item.get('date') or match.get('date'),
            'license': item.get('license'), 'uses': item.get('uses', []),
            'source_excluded': item.get('excluded'), 'exposure': sorted(set(exposure)),
            'pretraining_exposure': ('known_R2_LM_PleIAs_French_PD' if issue in lm_pretraining_issues
                                     else 'unknown_requires_audit'),
            'protected_page_or_issue': protection, 'protected_title': title_contacts,
            'press_same_issue': sorted(press_issues.get(issue, set())),
            'contacts': sorted(set(contacts)), 'identity_gaps': issues,
            'eligible_for_split': False, 'blind_test_eligible': False, 'title_ood_test_eligible': False,
            'heldout_blockers': sorted(set(reasons)),
        })
    nonexcluded = [r for r in rows if r['source_split'] == 'train' and not r['source_excluded']]
    # This pool describes source training rows only. It authorizes neither training
    # nor held-out selection; all blockers stay attached to every row.
    potential = [r for r in nonexcluded if not r['protected_page_or_issue'] and not r['contacts']
                 and 'G0_review_B1_KB_quarantine' not in r['heldout_blockers']]
    known = [r for r in rows if r['protected_page_or_issue']]
    gaps = [{'source': r['source'], 'page': r['page']} for r in rows if not r['issue_key']]
    count_source = lambda subset: dict(sorted(Counter(r['source'] for r in subset).items()))
    return {
        'schema_version': '0.4.0', 'role': 'metadata_inventory',
        'exposure_identity_complete': not unresolved_exposures and not missing_exposure_sources,
        'unresolved_exposures': unresolved_exposures,
        'exposure_sources': exposure_sources, 'missing_exposure_sources': missing_exposure_sources,
        'split_release_status': ('blocked_unresolved_exposures' if unresolved_exposures else
                                 'blocked_missing_exposure_sources' if missing_exposure_sources else
                                 'pending_independent_audit'),
        'purpose': 'Conservative inventory; no row released for blind or title-OOD test',
        'counts': {
            'unresolved_exposure_records': len(unresolved_exposures),
            'unresolved_exposures_by_label': dict(sorted(Counter(r['label'] for r in unresolved_exposures).items())),
            'unresolved_exposures_by_source': dict(sorted(Counter(r['source'] for r in unresolved_exposures).items())),
            'missing_exposure_sources': len(missing_exposure_sources),
            'wikisource_credited_source_documents': len(ws_protections['documents']),
            'wikisource_credited_source_pages': ws_protections['credited_pages'],
            'wikisource_credited_source_indexes': ws_protections['credited_indexes'],
            'press_metadata_by_corpus_split': press_counts,
            'press_metadata_lines': sum(r.get('line_count', 1) for r in press_metadata),
            'press_metadata_local_page_identifiers': len(press_metadata),
            'seg_manifest_pages': len(rows), 'E12_pages': len(e12), 'E22_independent_pages': len(e22),
            'nonexcluded_source_train_pages': len(nonexcluded),
            'nonexcluded_train_pages_without_known_overlap': len(potential),
            'known_overlaps': len(known), 'source_excluded_pages': sum(bool(r['source_excluded']) for r in rows),
            'text_contact_pages': sum(bool(r['contacts']) for r in rows),
            'missing_page_identity': sum(r['page_key'] is None for r in rows),
            'missing_issue_identity': len(gaps),
            'nonexcluded_train_missing_title': sum(not r['title_key'] for r in nonexcluded),
            'nonexcluded_train_title_seen_in_dev_test': sum(bool(r['protected_title']) for r in nonexcluded),
            'nonexcluded_train_in_press_train': sum('E6_E8_press:train' in r['exposure'] for r in nonexcluded),
            'r2_lm_pretraining_issue_count': len(lm_pretraining_issues),
            'r2_lm_pretraining_candidate_overlap_pages': sum(r['issue_key'] in lm_pretraining_issues
                                                             for r in rows if r['issue_key']),
            'blind_test_eligible': 0, 'title_ood_test_eligible': 0,
        },
        'missing_issue_identity_by_source': count_source([r for r in rows if r['issue_key'] is None]),
        'missing_page_identity_by_source': count_source([r for r in rows if r['page_key'] is None]),
        'sources': seg.get('summary', {}), 'pages': rows,
        'wikisource_source_protections': ws_protections,
        'candidate_train_pages': potential, 'candidate_scope': 'unreleased source training pool; not held-out candidates',
        'known_overlaps': [{'source': r['source'], 'page': r['page'], 'page_key': r['page_key'],
                            'issue_key': r['issue_key'], 'flags': r['protected_page_or_issue']} for r in known],
        'missing_issue_identity': gaps,
        'protected_titles': {k: sorted(v) for k, v in sorted(exposed_titles.items())},
        'p3_audit': c7_audit.get('p3', {'status': 'missing'}),
        'limitations': [
            'No image, PAGE/ALTO XML or test transcription was opened.',
            'Train/dev transcriptions occur in press JSONL files read; they are neither used nor emitted.',
            'Existing 8-gram contacts are inherited; absence is not proof of independence.',
            'KB concordance workbook is absent locally; no date or issue inferred from page id.',
            'AS train/dev/test pages without READ crosswalk remain unresolved across sources.',
            'E7 page metadata may precede final selection; exposed conservatively, final GT unopened.',
            'Wikisource credits protect the union of all source documents; no Gallica folio is inferred.',
            'Press local page identifiers, especially bnl_pack, are not proven historical pages.',
            'E14/E15/E24/C2/E1_BnL exposure metadata is missing or not integrated; exposure completeness stays false.',
            'R2 French-PD LM manifest pins 322 pretraining issues; its protected-ID list and token/8-gram exclusion evidence are not embedded for independent recomputation.',
            'E21 train membership is allocation, not proof a segmentation model trained on it.',
            'C7 P3 blocklist reported unavailable; no cloud placeholder downloaded.',
            'No model pretraining, GT completeness, near-duplicate or image audit completed.',
        ],
    }


class Inputs:
    def __init__(self):
        self.hashes = {}
        self.directories = {}
        self.supplement_paths = {}

    def read(self, path, jsonl=False):
        path = Path(path)
        raw = path.read_bytes()
        self.hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return [json.loads(line) for line in raw.splitlines() if line.strip()] if jsonl else json.loads(raw)

    def filenames(self, directory, pattern):
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f'required metadata filename directory is missing: {directory}')
        names = sorted(p.name for p in directory.glob(pattern) if p.is_file())
        if not names:
            raise ValueError(f'required metadata filename directory is empty: {directory}')
        self.directories[str(directory)] = {'pattern': pattern, 'names': names}
        return [Path(n).stem for n in names]

    def current(self):
        return (all(digest(Path(p)) == sha for p, sha in self.hashes.items()) and
                all(Path(d).is_dir() and sorted(p.name for p in Path(d).glob(v['pattern']) if p.is_file()) == v['names']
                    for d, v in self.directories.items()))


def load_ws_crosswalk(inputs, config_path, pages_metadata_path):
    """Load one explicitly pinned supplement and its exact local page-table binding."""
    config = inputs.read(config_path)
    if config.get('schema_version') != 'babaorum.g0_metadata_supplements.v2':
        raise ValueError('unsupported G0 metadata supplements schema')
    descriptor = config['wikisource_crosswalk']
    report_path = Path(descriptor['path'])
    pages_path = Path(pages_metadata_path)
    if not report_path.is_absolute() or not pages_path.is_absolute():
        raise ValueError('Wikisource supplement paths must be absolute')
    if descriptor['pages_metadata_path'] != str(pages_path):
        raise ValueError('Wikisource supplement metadata path mismatch')
    for key in ('sha256', 'pages_metadata_sha256'):
        if not re.fullmatch(r'[0-9a-f]{64}', descriptor[key]):
            raise ValueError(f'invalid Wikisource supplement {key}')
    report = inputs.read(report_path)
    if inputs.hashes[str(report_path)] != descriptor['sha256']:
        raise ValueError('Wikisource supplement report SHA-256 mismatch')
    # Main has already parsed these exact bytes; direct tests can load them here.
    if str(pages_path) not in inputs.hashes:
        inputs.read(pages_path, jsonl=True)
    actual = inputs.hashes[str(pages_path)]
    if (actual != descriptor['pages_metadata_sha256'] or
            report['local_inputs'].get(str(pages_path)) != actual or digest(pages_path) != actual):
        raise ValueError('Wikisource supplement pages metadata SHA-256 mismatch')
    inputs.supplement_paths['wikisource_crosswalk'] = str(report_path)
    return report


def load_r2_lm_manifest(inputs, config_path):
    """Load only the R2 corpus manifest's issue IDs; it contains no OCR text."""
    config = inputs.read(config_path)
    descriptor = config.get('french_pd_lm_manifest') if isinstance(config, dict) else None
    if not isinstance(descriptor, dict) or set(descriptor) != {'path', 'sha256'}:
        raise ValueError('R2 French-PD LM manifest descriptor missing or invalid')
    expected_path = Path('/Users/marcel/corpus-vt/french-pd-sample/manifest.json')
    path = Path(descriptor['path'])
    if path != expected_path or not path.is_absolute():
        raise ValueError('R2 French-PD LM manifest path is not the pinned local manifest')
    if not re.fullmatch(r'[0-9a-f]{64}', descriptor['sha256']):
        raise ValueError('invalid R2 French-PD LM manifest SHA-256')
    manifest = inputs.read(path)
    if inputs.hashes[str(path)] != descriptor['sha256']:
        raise ValueError('R2 French-PD LM manifest SHA-256 mismatch')
    if (not isinstance(manifest, dict) or
            manifest.get('source') != 'PleIAs/French-PD-Newspapers (datasets-server)' or
            manifest.get('min_ocr') != 95 or manifest.get('seed') != 2 or
            manifest.get('chars') != 58032354 or manifest.get('protected_ids') != 370 or
            manifest.get('skipped') != {'ocr': 378, 'protected': 0}):
        raise ValueError('R2 French-PD LM manifest metadata differs from the reviewed R2 receipt')
    file_ids = manifest.get('kept_file_ids')
    if (not isinstance(file_ids, list) or len(file_ids) != 322 or
            any(not isinstance(value, str) for value in file_ids) or
            len(set(file_ids)) != len(file_ids)):
        raise ValueError('R2 French-PD LM manifest file IDs must be 322 unique strings')
    issues = []
    for file_id in file_ids:
        try:
            issue = canonical_issue(file_id)
        except IdentityError as exc:
            raise ValueError(f'unrecognized R2 French-PD Gallica issue ID: {file_id}') from exc
        if not issue or not issue.startswith('gallica:'):
            raise ValueError(f'unrecognized R2 French-PD Gallica issue ID: {file_id}')
        issues.append(issue)
    if len(set(issues)) != len(file_ids):
        raise ValueError('R2 French-PD manifest contains duplicate canonical issue identities')
    return manifest, set(issues)


def git_state(axel):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(axel), *args])
    return {'head': git('rev-parse', 'HEAD').decode().strip(),
            'status_porcelain': git('status', '--porcelain', '--untracked-files=normal').decode(),
            'tracked_diff_sha256': hashlib.sha256(git('diff', '--binary', 'HEAD')).hexdigest()}


def compute_inventory(axel=Path('/Users/marcel/axel')):
    """Recompute the complete metadata receipt from pinned sources in memory.

    This pure result is also used by the admission verifier to compare every
    derived protection field against a fresh run over current, hashed inputs.
    It does not write reports or touch Axel.
    """
    axel = Path(axel).resolve()
    if axel != Path('/Users/marcel/axel'):
        raise ValueError('alternate Axel root is not supported by the frozen evidence')
    inputs = Inputs()
    before = git_state(axel)
    seg = inputs.read(axel / 'data/prepared/seg/manifest.json')
    as_manifest = inputs.read(axel / 'data/prepared/olr/newseye_as.json')
    e12 = inputs.read(axel / 'runs/e12/pages.json')
    e22 = inputs.read(axel / 'runs/e22/bench/runs/e12/pages.json')
    test_manifest = inputs.read(axel / 'data/prepared/presse_test_manifest.json')
    c7_audit = inputs.read(axel / 'runs/c7/audit.json')
    e7_metadata = []
    for name in ('pages_bnf.jsonl', 'pages_impact.jsonl', 'pages.jsonl'):
        e7_metadata += [dict(row, _metadata_source=name) for row in
                        inputs.read(axel / 'data/prepared/presse_e7' / name, jsonl=True)]
    ws_crosswalk = load_ws_crosswalk(inputs, ROOT / 'config/g0_metadata_supplements.json',
                                    axel / 'data/prepared/presse_e7/pages.jsonl')
    r2_lm_manifest, r2_lm_issues = load_r2_lm_manifest(inputs, ROOT / 'config/g0_metadata_supplements.json')
    press_metadata = []
    for split in ('train', 'dev'):
        # Only page/corpus/split survive; no transcription used or emitted.
        rows = inputs.read(axel / f'data/prepared/press/{split}.jsonl', jsonl=True)
        if any(row['split'] != split for row in rows):
            raise ValueError(f'press metadata split does not match its source file: {split}')
        counts = Counter((row['page'], row['corpus']) for row in rows)
        press_metadata += [{'page': page, 'corpus': corpus, 'split': split, 'line_count': count,
                            '_metadata_source': f'press/{split}.jsonl'}
                           for (page, corpus), count in sorted(counts.items())]
    val = inputs.filenames(Path('/Users/marcel/corpus-vt/newseye/val/ATR_ValidationSet_BnF_Newseye_M2+'), '*.xml')
    out = build_inventory(seg, as_manifest, e12, e22, e7_metadata, press_metadata, val, test_manifest,
                          c7_audit, ws_crosswalk=ws_crosswalk, lm_pretraining_issues=r2_lm_issues)
    lm_path = str(ROOT.parent / 'corpus-vt/french-pd-sample/manifest.json')
    out['r2_lm_pretraining'] = {
        'source': r2_lm_manifest['source'], 'manifest_path': lm_path,
        'manifest_sha256': inputs.hashes[lm_path], 'kept_issue_count': len(r2_lm_issues),
        'chars': r2_lm_manifest['chars'], 'min_ocr': r2_lm_manifest['min_ocr'],
        'seed': r2_lm_manifest['seed'], 'protected_ids_reported': r2_lm_manifest['protected_ids'],
        'protected_id_list_embedded': False,
        'skipped_reported': r2_lm_manifest['skipped'],
    }
    seg_path = str(axel / 'data/prepared/seg/manifest.json')
    out['inventory_source'] = {'path': seg_path, 'sha256': inputs.hashes[seg_path]}
    out['source_inputs'] = {
        'C7': [seg_path, str(axel / 'runs/c7/audit.json')],
        'E12': [str(axel / 'runs/e12/pages.json')],
        'E22': [str(axel / 'runs/e22/bench/runs/e12/pages.json')],
        'E7': [str(axel / 'data/prepared/presse_test_manifest.json')] +
              [str(axel / 'data/prepared/presse_e7' / name) for name in
               ('pages_bnf.jsonl', 'pages_impact.jsonl', 'pages.jsonl')] +
              [str(ROOT / 'config/g0_metadata_supplements.json'),
               inputs.supplement_paths['wikisource_crosswalk']],
        'NewsEye_validation': list(inputs.directories),
        'AS': [str(axel / 'data/prepared/olr/newseye_as.json')],
        'E6_E8_press': [str(axel / f'data/prepared/press/{split}.jsonl') for split in ('train', 'dev')],
        'R2_LM_PleIAs_French_PD': [str(ROOT.parent / 'corpus-vt/french-pd-sample/manifest.json')],
        **{name: [] for name in out['missing_exposure_sources']},
    }
    for path in (Path(__file__), ROOT / 'src/babaorum/identity.py',
                 ROOT / 'src/babaorum/legacy/data/doc_key.py', ROOT / 'src/babaorum/contracts.py'):
        inputs.hashes[str(path)] = digest(path)
    after = git_state(axel)
    out.update(created_utc=datetime.now(timezone.utc).isoformat(), inputs=inputs.hashes,
               directory_inputs=inputs.directories, axel_head=before['head'],
               axel_worktree_before=before, axel_worktree_after=after,
               metadata_current=inputs.current() and before == after)
    if not out['metadata_current']:
        out['limitations'].append('Sources changed during inventory; receipt is stale and must not be consumed.')
    return out


def main():
    out = compute_inventory()
    atomic_write(ROOT / 'reports/g0_inventory.json', out)
    print(json.dumps(out['counts'], ensure_ascii=False))


if __name__ == '__main__':
    main()
