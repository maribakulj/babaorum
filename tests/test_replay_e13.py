import importlib.util,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('replay_e13_test',ROOT/'scripts/replay_e13.py')
replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)


def test_changed_historical_input_refused(tmp_path):
    path=tmp_path/'input.xml';path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='frozen input mismatch'):
        replay.read_pinned(path,{str(path):'0'*64},{})


def test_historical_scope_and_two_metrics():
    config=json.loads((ROOT/'config/replay_e13.json').read_text())
    assert len(config['pages'])==len(set(config['pages']))==7
    assert config['expected']['gallica']['chars']==160740
    assert config['expected']['gallica']['d212']==0.0404705184335079
    assert config['expected']['gallica']['d217']==0.07238290420932177
    assert config['expected']['tesseract']['d212']==0.06540139563902257
    assert config['expected']['tesseract']['d217']==0.09580794499537545


def test_registration_not_exported():
    assert not hasattr(replay,'_registration_helpers')
    assert not hasattr(replay,'register')


def test_outside_freeze_not_opened(tmp_path):
    with pytest.raises(ValueError,match='outside frozen set'):
        replay.read_pinned(tmp_path/'nonexistent',{}, {})


def test_labels_and_actual_parity():
    result=replay.run()
    assert result['role']=='historical_diagnostic' and result['gt_assisted'] is True
    assert result['all_identical'] and len(result['rows'])==14
    config=json.loads((ROOT/'config/replay_e13.json').read_text())
    assert result['aggregates']==config['expected']
    assert config['role']==result['role'] and config['gt_assisted'] is True


def test_gt_assistance_rejected_everywhere():
    from babaorum.contracts import validate_predictions,allow_development,validate_heldout,ContractError
    data={'gt_assisted':True}
    for call in (lambda:validate_predictions(data,{},''),lambda:allow_development(data),lambda:validate_heldout(data,{})):
        with pytest.raises(ContractError,match='GT-assisted'):call()


def test_exposed_e13_pages_excluded():
    from babaorum.contracts import validate_heldout,ContractError
    registry=json.loads((ROOT/'config/sealed_registry.json').read_text())
    for page in registry['heldout_exclusions']['page_ids']:
        with pytest.raises(ContractError,match='historically exposed'):
            validate_heldout({'lines':[{'page_id':page,'document_key':'unknown'}]},registry)
    for doc in registry['heldout_exclusions']['document_keys']:
        with pytest.raises(ContractError,match='historically exposed'):
            validate_heldout({'lines':[{'page_id':'new-folio','document_key':doc}]},registry)


def test_loop_diagnostic_labels():
    spec=importlib.util.spec_from_file_location('loop_label_test',ROOT/'scripts/loop.py')
    loop=importlib.util.module_from_spec(spec);spec.loader.exec_module(loop)
    for valid in (True,False):
        assert loop.e13_status(valid)=={'role':'historical_diagnostic','gt_assisted':True,'current':valid}
