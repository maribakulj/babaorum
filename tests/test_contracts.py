import copy
import hashlib
import json
import pytest
from babaorum.contracts import ContractError,load,validate_manifest,validate_predictions
from babaorum.benchmark import evaluate


def manifest():
    return {'schema_version':'0.1.0','task':'same_crops','dataset_id':'fixture','exposure':'fixture','key_policy':'fixture_explicit','coordinate_frame':'original_encoded_pixels_top_left_xy','crop_recipe':'axis_aligned_xyxy_no_rectification',
            'lines':[{'id':'a','document_key':'doc','page_id':'p','split':'dev','image_sha256':'a'*64,'crop_sha256':'b'*64,'image_size':[100,100],'box':[0,0,30,20],'text':'ſon'}]}


def predictions():
    return {'schema_version':'0.1.0','manifest_sha256':'c'*64,'model':{'name':'fixture','code_revision':'fixture','checkpoint_sha256':'d'*64,'config_sha256':'e'*64,'alphabet_sha256':'f'*64,'license':'fixture','pretraining_audit':'fixture'},'rows':[]}


def test_valid_and_identity_leak():
    m=manifest();validate_manifest(m)
    row=copy.deepcopy(m['lines'][0]);row.update(id='b',page_id='p2',split='test',image_sha256='e'*64,crop_sha256='f'*64)
    m['lines'].append(row)
    with pytest.raises(ContractError,match='leak'):validate_manifest(m)


def test_duplicate_keys_and_nonfinite(tmp_path):
    p=tmp_path/'bad.json'
    for s in ('{"a":1,"a":2}','{"a":NaN}'):
        p.write_text(s)
        with pytest.raises(ContractError):load(p)


@pytest.mark.parametrize('box',[[0,0,float('inf'),5],[20,0,10,5],[-1,0,10,5],[0,0,101,5]])
def test_bad_geometry(box):
    m=manifest();m['lines'][0]['box']=box
    with pytest.raises(ContractError):validate_manifest(m)


def test_wrong_crop_and_extra_ids():
    m=manifest();p=predictions()
    p['rows']=[{'id':'a','crop_sha256':'e'*64,'status':'ok','text':'son'}]
    with pytest.raises(ContractError,match='different crop'):validate_predictions(p,m,'c'*64)
    p['rows'][0]['id']='unknown'
    with pytest.raises(ContractError,match='unknown'):validate_predictions(p,m,'c'*64)


def test_missing_prediction_counts_as_deletion(tmp_path):
    m=tmp_path/'m.json';p=tmp_path/'p.json';m.write_text(json.dumps(manifest()))
    data=predictions();data['manifest_sha256']=hashlib.sha256(m.read_bytes()).hexdigest();p.write_text(json.dumps(data))
    r=evaluate(m,p)['splits']['dev']
    assert r['cer_raw']==100 and r['missing_predictions']==1 and r['coverage']==0


def test_empty_truth_invention_reported(tmp_path):
    data=manifest();data['lines'][0]['text']=''
    m=tmp_path/'m.json';p=tmp_path/'p.json';m.write_text(json.dumps(data))
    pred=predictions();pred['manifest_sha256']=hashlib.sha256(m.read_bytes()).hexdigest()
    pred['rows']=[{'id':'a','crop_sha256':'b'*64,'status':'ok','text':'invented'}];p.write_text(json.dumps(pred))
    r=evaluate(m,p)['splits']['dev']
    assert r['empty_truth_output_chars_raw']==8 and r['cer_raw'] is None and r['support_chars_raw']==0


def test_sealed_test_not_exposed(tmp_path):
    data=manifest();data['exposure']='sealed_test';m=tmp_path/'m.json';m.write_text(json.dumps(data))
    with pytest.raises(ContractError,match='sealed test'):evaluate(m,tmp_path/'absent.json')


def test_development_cannot_hide_test():
    m=manifest();m['exposure']='development';m['lines'][0]['split']='test'
    with pytest.raises(ContractError,match='test split'):validate_manifest(m)


def test_crop_cannot_be_counted_twice():
    m=manifest();row=copy.deepcopy(m['lines'][0]);row['id']='b';row['box']=[0,30,30,50];m['lines'].append(row)
    with pytest.raises(ContractError,match='duplicate crop bytes'):validate_manifest(m)


def test_page_metadata_cannot_conflict():
    m=manifest();row=copy.deepcopy(m['lines'][0]);row.update(id='b',image_size=[101,100],crop_sha256='e'*64);m['lines'].append(row)
    with pytest.raises(ContractError,match='page identity'):validate_manifest(m)


def test_bad_manifest_hash_and_duplicate_predictions():
    m=manifest();p=predictions()
    with pytest.raises(ContractError,match='benchmark hash'):validate_predictions(p,m,'e'*64)
    row={'id':'a','crop_sha256':'b'*64,'status':'ok','text':'son'};p['rows']=[row,row]
    with pytest.raises(ContractError,match='duplicate prediction'):validate_predictions(p,m,'c'*64)


def test_error_result_is_counted(tmp_path):
    m=tmp_path/'m.json';p=tmp_path/'p.json';m.write_text(json.dumps(manifest()))
    pred=predictions();pred['manifest_sha256']=hashlib.sha256(m.read_bytes()).hexdigest()
    pred['rows']=[{'id':'a','crop_sha256':'b'*64,'status':'error','text':''}];p.write_text(json.dumps(pred))
    r=evaluate(m,p)['splits']['dev']
    assert r['failed_predictions']==1 and r['cer_raw']==100 and r['coverage']==0


def test_same_image_cannot_be_aliased():
    m=manifest();row=copy.deepcopy(m['lines'][0]);row.update(id='b',page_id='alias',document_key='other',crop_sha256='e'*64,box=[1,0,31,20]);m['lines'].append(row)
    with pytest.raises(ContractError,match='image reused'):validate_manifest(m)


@pytest.mark.parametrize('page',[
    'gallica:7635415/f2','read_Ce-soir_7635415_D0000002',
    'gallica:7635415/f02','ark:/12148/bpt6k76354152/f2',
    'gallica:bpt6k76354152/f2','read_7635415-002',
    'gallica:7635415/f99','7635415-099',
])
def test_benchmark_rejects_exposed_alias_before_predictions(tmp_path,page):
    m=manifest();m['lines'][0].update(page_id=page,document_key='arbitrary')
    path=tmp_path/'manifest.json';path.write_text(json.dumps(m))
    with pytest.raises(ContractError,match='historically exposed'):
        evaluate(path,tmp_path/'predictions-not-opened.json')


def test_c7_keys_ported_exactly():
    from babaorum.legacy.data.doc_key import doc_key,issue_key
    from babaorum.identity import identities
    assert doc_key('e12','Ce-soir_7635415_D0000002','bpt6k76354152')=='gallica:7635415/f2'
    assert issue_key('e12','Ce-soir_7635415_D0000002','bpt6k76354152')=='gallica:7635415'
    assert identities('gallica:07635415/f02')==('gallica:7635415/f2','gallica:7635415')
