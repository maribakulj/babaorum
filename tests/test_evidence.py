import hashlib,importlib.util,json,subprocess
from pathlib import Path
import pytest
from babaorum.evidence import signature,current,atomic_write
from babaorum.contracts import allow_development,ContractError
from test_contracts import manifest
ROOT=Path(__file__).resolve().parents[1]


def test_receipt_expires_on_config_change(tmp_path):
    (tmp_path/'config').mkdir();p=tmp_path/'config/model.json';p.write_text('{}')
    receipt={'signature':signature(tmp_path),'passed':True}
    assert current(receipt,tmp_path)
    p.write_text('{"changed":true}')
    assert not current(receipt,tmp_path)


def test_reports_replace_atomically(tmp_path):
    p=tmp_path/'report.json';atomic_write(p,{'a':1});atomic_write(p,{'a':2})
    assert json.loads(p.read_text())=={'a':2}


def test_legacy_data_not_gitignored():
    result=subprocess.run(['git','check-ignore','src/babaorum/legacy/data/records.py','src/babaorum/legacy/data/alto_read.py'],cwd=ROOT,capture_output=True)
    assert result.returncode==1


def test_no_test_release_even_if_relabelled_historical():
    m=manifest();m['exposure']='historical';m['lines'][0]['split']='test'
    with pytest.raises(ContractError,match='dev-only'):allow_development(m)


def test_pending_registry_blocks_real_benchmark():
    m=manifest();m['exposure']='development'
    with pytest.raises(ContractError,match='registry'):allow_development(m)


def test_sealed_identity_rejected_in_fixture_too(tmp_path):
    p=tmp_path/'registry.json';p.write_text(json.dumps({'status':'frozen','document_keys':['doc'],'image_sha256':[],'crop_sha256':[]}))
    with pytest.raises(ContractError,match='sealed identity'):allow_development(manifest(),p)


def test_replay_expected_values_frozen():
    data=json.loads((ROOT/'config/replay_dev.json').read_text())
    assert data['schema_version']=='babaorum.historical_replay.v1'
    assert (data['role'],data['dataset'],data['split'],data['chain'])==('historical_diagnostic','E22','dev','c1')
    assert data['expected_aggregates']['cer_reordered']==0.06549731393501201
    assert data['expected_aggregates']['cer_reordered_d212']==0.07848407434653086
    assert len(data['pages'])==5 and len(set(data['pages']))==5


def test_failed_cycle_clears_previous_green_status(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('babaorum_loop_test',ROOT/'scripts/loop.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    atomic_write(tmp_path/'reports/loop_status.json',{'checks_passed':True})
    def fail(*args):raise ValueError('changed port')
    monkeypatch.setattr(module,'cycle',fail)
    with pytest.raises(ValueError):module.safe_cycle(tmp_path,False)
    report=json.loads((tmp_path/'reports/loop_status.json').read_text())
    assert report['status']=='failed' and not report['checks_passed']


def test_fixture_label_cannot_admit_unknown_real_manifest():
    m=manifest();m['lines'][0]['text']='unapproved real transcription'
    with pytest.raises(ContractError,match='not allowlisted'):allow_development(m)


def test_alternate_axel_root_rejected_before_reading(tmp_path):
    spec=importlib.util.spec_from_file_location('babaorum_replay_test',ROOT/'scripts/replay_dev.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with pytest.raises(ValueError,match='alternate --axel'):module.run(tmp_path)


def test_environment_change_invalidates_receipt(tmp_path,monkeypatch):
    import babaorum.evidence as evidence
    receipt={'signature':signature(tmp_path),'passed':True}
    monkeypatch.setattr(evidence.sys,'version','different Python')
    assert not current(receipt,tmp_path)


def test_external_axel_lockfile_is_reported_in_runtime_environment(tmp_path):
    source=tmp_path/'axel';source.mkdir()
    (source/'uv.lock').write_text('lock version one')
    provenance=tmp_path/'provenance';provenance.mkdir()
    (provenance/'ported.json').write_text(json.dumps({'repository':str(source)}))
    from babaorum.evidence import runtime_environment
    expected=hashlib.sha256(b'lock version one').hexdigest()
    assert runtime_environment(tmp_path)['axel_uv_lock_sha256']==expected
    (source/'uv.lock').write_text('lock version two')
    assert runtime_environment(tmp_path)['axel_uv_lock_sha256']==hashlib.sha256(b'lock version two').hexdigest()


def test_pipeline_receipt_ignores_joint_code_but_not_shared_code(tmp_path):
    from babaorum.evidence import pipeline_signature
    folder=tmp_path/'src/babaorum/experimental';folder.mkdir(parents=True)
    shared=folder.parent/'page_graph.py';shared.write_text('v1')
    first=pipeline_signature(tmp_path);global_first=signature(tmp_path)
    (folder/'joint_page.py').write_text('joint experiment v2')
    assert pipeline_signature(tmp_path)==first
    assert signature(tmp_path)!=global_first
    shared.write_text('v2')
    assert pipeline_signature(tmp_path)!=first


def test_pipeline_receipt_ignores_reader_owned_paths_but_global_signature_tracks_them(tmp_path):
    from babaorum.evidence import pipeline_signature
    paths = (
        'src/babaorum/reader/ctc.py', 'tests/test_reader_contract.py',
        'scripts/reader_probe.py', 'config/reader_v1.json', 'docs/READER_PLAN.txt',
    )
    first_pipeline = pipeline_signature(tmp_path)
    first_global = signature(tmp_path)
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('reader-owned v1')
    assert pipeline_signature(tmp_path) == first_pipeline
    assert signature(tmp_path) != first_global


def test_g0_false_receipt_cannot_become_green(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('loop_stale_g0_test',ROOT/'scripts/loop.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.g0_receipt_current({'metadata_current':False},tmp_path) is False
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'inventory',lambda axel:{'free_bytes':0})
    monkeypatch.setattr(module,'verify_port',lambda:{})
    monkeypatch.setattr(module,'g0_receipt_current',lambda *args:False)
    monkeypatch.setattr(module.subprocess,'run',lambda *args,**kwargs:subprocess.CompletedProcess([],0,'',''))
    atomic_write(tmp_path/'reports/g0_inventory.json',{'metadata_current':False,'counts':{}})
    result=module.cycle(Path('/Users/marcel/axel'),False)
    assert result['g0_metadata_inventory_current'] is False
    assert result['gates']['G0']['status']=='stale_metadata'
    status=json.loads((tmp_path/'reports/loop_status.json').read_text())
    assert status['g0_inventory_counts'] is None
    assert status['g0_receipt_sha256']==hashlib.sha256((tmp_path/'reports/g0_inventory.json').read_bytes()).hexdigest()
    assert 'pipeline_signature' in status and 'updated_at_utc' in status


def _pending_g0_receipt(tmp_path, monkeypatch, module):
    # Exercise loop integration independently of the detailed G0 receipt
    # fixtures: those belong to test_g0_contract_guard.py.
    path=tmp_path/'source.json';path.write_text('metadata')
    directory=tmp_path/'source-dir';directory.mkdir();(directory/'page.xml').write_text('<Page/>')
    receipt={'metadata_current':True,'split_release_status':'blocked_pending_audit',
             'inputs':{str(path):hashlib.sha256(path.read_bytes()).hexdigest()},
             'directory_inputs':{str(directory):{'pattern':'*.xml','names':['page.xml']}},
             'source_inputs':{'test':[str(path),str(directory)]}}
    def verify(candidate, *, expected_axel_root=None):
        assert Path(expected_axel_root)==tmp_path
        for name,sha in candidate['inputs'].items():
            file=Path(name)
            if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=sha:
                raise ValueError('stale file input')
        for name,entry in candidate['directory_inputs'].items():
            folder=Path(name)
            if not folder.is_dir() or sorted(p.name for p in folder.glob(entry['pattern']))!=entry['names']:
                raise ValueError('stale directory input')
        if set(candidate['source_inputs'])!={'test'}:raise ValueError('incomplete sources')
        if not set(candidate['source_inputs']['test']) <= set(candidate['inputs']) | set(candidate['directory_inputs']):
            raise ValueError('source role is not pinned')
    monkeypatch.setattr(module,'verify_metadata_receipt',verify)
    module.head=lambda axel:'test-head'
    diff=b'test diff'
    def git(*args,**kwargs):
        return 'test status' if args[0][3]=='status' else diff
    monkeypatch.setattr(module.subprocess,'check_output',git)
    state={'head':'test-head','status_porcelain':'test status','tracked_diff_sha256':hashlib.sha256(diff).hexdigest()}
    receipt['axel_head']='test-head'
    receipt['axel_worktree_before']=state
    receipt['axel_worktree_after']=state
    return receipt,[str(path),str(directory)],receipt['inputs'],receipt['directory_inputs']


def test_pending_g0_metadata_is_current_without_release(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('loop_pending_g0',ROOT/'scripts/loop.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    receipt,_,_,_=_pending_g0_receipt(tmp_path,monkeypatch,module)
    receipt['split_release_status']='blocked_missing_exposure_sources'
    receipt['exposure_identity_complete']=False
    assert module.g0_receipt_current(receipt,tmp_path) is True


def test_g0_metadata_freshness_requires_all_sources_dirs_and_hashes(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('loop_g0_metadata_complete',ROOT/'scripts/loop.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    receipt,roles,inputs,dirs=_pending_g0_receipt(tmp_path,monkeypatch,module)
    assert module.g0_receipt_current(receipt,tmp_path) is True

    for mutation in ('role_map','hash','directory_map','directory_listing'):
        bad=json.loads(json.dumps(receipt))
        if mutation=='role_map':bad['source_inputs'].pop('test')
        elif mutation=='hash':bad['inputs'][roles[0]]='0'*64
        elif mutation=='directory_map':bad['directory_inputs'].clear()
        else:bad['directory_inputs'][next(iter(dirs))]['names'].clear()
        assert module.g0_receipt_current(bad,tmp_path) is False, mutation

    stale=json.loads(json.dumps(receipt));stale['metadata_current']=False
    assert module.g0_receipt_current(stale,tmp_path) is False

    substituted=json.loads(json.dumps(receipt))
    substituted['source_inputs']['test']=[str(tmp_path/'wrong-role.json')]
    assert module.g0_receipt_current(substituted,tmp_path) is False


def test_g0_freshness_invalidates_deleted_metadata_file_and_directory(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('loop_g0_removed_inputs',ROOT/'scripts/loop.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    receipt,roles,_,dirs=_pending_g0_receipt(tmp_path,monkeypatch,module)
    Path(roles[0]).unlink()
    assert module.g0_receipt_current(receipt,tmp_path) is False

    second=tmp_path/'second';second.mkdir()
    receipt,roles,_,dirs=_pending_g0_receipt(second,monkeypatch,module)
    directory=Path(next(iter(dirs)))
    (directory/'page.xml').unlink()
    directory.rmdir()
    assert module.g0_receipt_current(receipt,second) is False


def test_pipeline_receipt_ignores_joint_analysis_scripts(tmp_path):
    # N4 (G1 second pass): end-to-end analysis scripts must not stale the
    # pipeline receipt, like probe_joint_* and test_joint_* already do.
    from babaorum.evidence import pipeline_signature
    (tmp_path/'scripts').mkdir()
    shared=tmp_path/'scripts/replay_dev.py';shared.write_text('v1')
    first=pipeline_signature(tmp_path);global_first=signature(tmp_path)
    (tmp_path/'scripts/analyze_joint_capture_v9.py').write_text('joint analysis v1')
    assert pipeline_signature(tmp_path)==first
    assert signature(tmp_path)!=global_first
    shared.write_text('v2')
    assert pipeline_signature(tmp_path)!=first
