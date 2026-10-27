import hashlib,json
from pathlib import Path
import pytest
from PIL import Image
from babaorum.adapters.pero import recognize
from babaorum.contracts import ContractError
from test_contracts import manifest,predictions

class Engine:
    line_px_height=20
    line_padding_px=32
    max_input_horizontal_pixels=200
    model_type='ctc'
    def process_lines(self,lines,no_logits):
        assert lines[0].shape==(20,30,3)
        return ['ſon'],[None],[None]


def fixture(tmp_path,width=30,color=False):
    image=tmp_path/'crop.png'
    name='pero_red' if color else ('pero_long' if width==200 else 'pero_gray')
    image.write_bytes((Path(__file__).parent/'fixtures'/(name+'.png')).read_bytes())
    m=manifest();m['lines'][0]['crop_sha256']=hashlib.sha256(image.read_bytes()).hexdigest()
    path=tmp_path/'m.json';path.write_text(json.dumps(m))
    return path,{'a':image}


def test_bridge_identity_and_text(tmp_path):
    path,crops=fixture(tmp_path)
    out,transforms=recognize(Engine(),path,crops,predictions()['model'])
    assert out['rows'][0]['text']=='ſon' and transforms[0]['network_size']==[30,20]


def test_no_silent_long_line_crop(tmp_path):
    path,crops=fixture(tmp_path,200)
    with pytest.raises(ContractError,match='truncate'):recognize(Engine(),path,crops,predictions()['model'])


def test_no_silent_short_result(tmp_path):
    class Bad(Engine):
        def process_lines(self,*args,**kwargs):return [],[],[]
    path,crops=fixture(tmp_path)
    with pytest.raises(ContractError,match='cardinality'):recognize(Bad(),path,crops,predictions()['model'])


def test_pero_receives_bgr_not_rgb(tmp_path):
    class ColorEngine(Engine):
        def process_lines(self,lines,no_logits):
            assert lines[0][0,0].tolist()==[0,0,255]
            return ['ſon'],[None],[None]
    path,crops=fixture(tmp_path,color=True)
    out,metadata=recognize(ColorEngine(),path,crops,predictions()['model'])
    assert metadata[0]['conversion'].startswith('BGR')


def test_crop_decoded_from_exact_hashed_bytes(tmp_path,monkeypatch):
    path,crops=fixture(tmp_path,color=True)
    original=Path.read_bytes
    reads=[]
    def read_once(self):
        if self==crops['a']:
            reads.append(self)
            assert len(reads)==1
            data=original(self)
            self.write_bytes(b'changed after immutable snapshot')
            return data
        return original(self)
    class RedEngine(Engine):
        def process_lines(self,lines,no_logits):
            assert lines[0][0,0].tolist()==[0,0,255]
            return ['ſon'],[None],[None]
    monkeypatch.setattr(Path,'read_bytes',read_once)
    result,_=recognize(RedEngine(),path,crops,predictions()['model'])
    assert len(reads)==1 and result['rows'][0]['status']=='ok'


def test_fixture_manifest_cannot_run_engine_outside_test_fake_boundary(tmp_path):
    path,_=fixture(tmp_path)
    with pytest.raises(ContractError,match='fake engine instance'):
        recognize(object(),path,{'a':tmp_path/'crop-does-not-exist.png'},predictions()['model'])


def test_real_preflight_rejects_unknown_license_before_factory_or_crop(tmp_path,monkeypatch):
    import babaorum.adapters.pero as adapter
    from test_contracts import manifest as make_manifest

    data=make_manifest()
    data['exposure']='historical'
    data['key_policy']='declared_document_v0'
    data['lines'][0]['page_id']='gallica:7633477/f1'
    data['lines'][0]['document_key']='gallica:7633477/f1'
    mpath=tmp_path/'historical.json';mpath.write_text(json.dumps(data))
    monkeypatch.setattr(adapter,'allow_development',lambda _:None)
    monkeypatch.setattr(adapter,'load',lambda _: {
        'weights_license':None,
        'weights_license_audit_status':'unknown',
        'weights_license_evidence':None,
    })
    calls=[]
    def factory(_settings):
        calls.append('called')
        raise AssertionError('engine factory must not run')
    with pytest.raises(ContractError,match='license audit is not reviewed'):
        recognize(factory,mpath,{'a':tmp_path/'missing-crop.png'},predictions()['model'])
    assert calls==[]


def test_real_provenance_hashes_config_alphabet_and_checkpoint_before_loading(tmp_path,monkeypatch):
    import hashlib
    import babaorum.adapters.pero as adapter

    checkpoint=tmp_path/'reader.pt.cpu';checkpoint.write_bytes(b'fixture checkpoint bytes')
    config=tmp_path/'ocr_engine.json';config.write_text(json.dumps({'characters':['a','ſ','œ'],'checkpoint':'reader.pt'}))
    raw=config.read_bytes()
    alphabet=hashlib.sha256(json.dumps(['a','ſ','œ'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    model={
        'name':'reader','code_revision':'a'*40,
        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'config_sha256':hashlib.sha256(raw).hexdigest(),
        'alphabet_sha256':alphabet,
        'license':'BSD-3-Clause',
        'weights_license':'permission grant 2026-10-07',
        'weights_license_audit_status':'reviewed',
        'pretraining_audit':'reviewed',
    }
    settings={
        'weights_license_audit_status':'reviewed',
        'weights_license':'permission grant 2026-10-07',
        'weights_license_evidence':'https://example.org/authorization',
        'code_audit_status':'reviewed','config_audit_status':'reviewed',
        'batch_size':2,'checkpoint_path':str(checkpoint),
        'device':'cpu','checkpoint_resolution':'pytorch_cpu_suffix_v1',
        'checkpoint_sha256':model['checkpoint_sha256'],
        'config_path':str(config),'config_sha256':model['config_sha256'],
        'alphabet_sha256':alphabet,'code_revision':'a'*40,
        'local_source_repository':str(tmp_path),
    }
    monkeypatch.setattr(adapter,'_code_revision',lambda _settings:'a'*40)
    expected=adapter._preflight_real(settings,model)
    assert expected['checkpoint_sha256']==model['checkpoint_sha256']
    checkpoint.write_bytes(b'changed checkpoint bytes')
    with pytest.raises(ContractError,match='checkpoint bytes differ'):
        adapter._preflight_real(settings,model)


def test_real_provenance_accepts_audited_non_spdx_permission_not_unknown(tmp_path,monkeypatch):
    import hashlib
    import babaorum.adapters.pero as adapter

    checkpoint=tmp_path/'reader.pt.cpu';checkpoint.write_bytes(b'w')
    config=tmp_path/'ocr.json';config.write_text(json.dumps({'characters':['a'],'checkpoint':'reader.pt'}))
    raw=config.read_bytes()
    model={
        'name':'reader','code_revision':'b'*40,
        'checkpoint_sha256':hashlib.sha256(b'w').hexdigest(),
        'config_sha256':hashlib.sha256(raw).hexdigest(),
        'alphabet_sha256':hashlib.sha256(b'["a"]').hexdigest(),
        'license':'BSD-3-Clause',
        'weights_license':'explicit institutional permission',
        'weights_license_audit_status':'reviewed',
        'pretraining_audit':'reviewed',
    }
    settings={
        'weights_license_audit_status':'reviewed','weights_license':'explicit institutional permission',
        'weights_license_evidence':'signed grant record ABC-2026',
        'code_audit_status':'reviewed','config_audit_status':'reviewed','batch_size':1,
        'device':'cpu','checkpoint_resolution':'pytorch_cpu_suffix_v1',
        'checkpoint_path':str(checkpoint),'checkpoint_sha256':model['checkpoint_sha256'],
        'config_path':str(config),'config_sha256':model['config_sha256'],
        'alphabet_sha256':model['alphabet_sha256'],'code_revision':'b'*40,
        'local_source_repository':str(tmp_path),
    }
    monkeypatch.setattr(adapter,'_code_revision',lambda _settings:'b'*40)
    assert adapter._preflight_real(settings,model)['code_revision']=='b'*40
    # Even a separately audited config hash cannot point at another weight file.
    config.write_text(json.dumps({'characters':['a'],'checkpoint':'other.pt'}))
    settings['config_sha256']=model['config_sha256']=hashlib.sha256(config.read_bytes()).hexdigest()
    with pytest.raises(ContractError,match='different checkpoint'):
        adapter._preflight_real(settings,model)


@pytest.mark.parametrize('absolute', [False,True])
def test_checkpoint_resolution_matches_upstream_cpu_suffix(tmp_path,absolute):
    from babaorum.adapters.pero import _checkpoint_from_config
    base=tmp_path/'reader.pt'
    raw=json.dumps({'checkpoint':str(base) if absolute else 'reader.pt'}).encode()
    settings={'device':'cpu','checkpoint_resolution':'pytorch_cpu_suffix_v1'}
    assert _checkpoint_from_config(raw,tmp_path/'ocr.json',settings)==tmp_path/'reader.pt.cpu'
    # Upstream appends even if the JSON already includes .cpu.
    raw=json.dumps({'checkpoint':'reader.pt.cpu'}).encode()
    assert _checkpoint_from_config(raw,tmp_path/'ocr.json',settings)==tmp_path/'reader.pt.cpu.cpu'


def test_loaded_engine_cannot_claim_hash_for_a_different_checkpoint(tmp_path):
    from types import SimpleNamespace
    from babaorum.adapters.pero import _verify_loaded_engine
    expected={'checkpoint_path':str(tmp_path/'audited.pt.cpu'),'device':'cpu'}
    engine=SimpleNamespace(batch_size=1,device=SimpleNamespace(type='cpu'),
                           checkpoint=str(tmp_path/'other.pt.cpu'),babaorum_provenance=expected)
    with pytest.raises(ContractError,match='checkpoint path differs'):
        _verify_loaded_engine(engine,expected,{'batch_size':1})


def test_real_bridge_rejects_any_caller_factory_before_calling_it(tmp_path,monkeypatch):
    import babaorum.adapters.pero as adapter
    data=manifest();data.update(exposure='historical',key_policy='declared_document_v0')
    data['lines'][0].update(page_id='gallica:7000001/f1',document_key='gallica:7000001')
    mpath=tmp_path/'historical.json';mpath.write_text(json.dumps(data))
    monkeypatch.setattr(adapter,'allow_development',lambda _:None)
    monkeypatch.setattr(adapter,'load',lambda _: {'real':'settings'})
    monkeypatch.setattr(adapter,'_preflight_real',lambda *_: {'checkpoint_sha256':'a'*64})
    monkeypatch.setattr(adapter,'validate_predictions',lambda *_: {})
    calls=[]
    def factory(_settings):calls.append('called')
    with pytest.raises(ContractError,match='no caller-provided engine or factory'):
        recognize(factory,mpath,{'a':tmp_path/'crop.png'},predictions()['model'])
    assert calls==[]


def test_real_bridge_accepts_native_constructor_interface_without_factory(tmp_path,monkeypatch):
    import babaorum.adapters.pero as adapter
    import babaorum.adapters.pero_snapshot as snapshot
    data=manifest();data.update(exposure='historical',key_policy='declared_document_v0')
    data['lines'][0].update(page_id='gallica:7000001/f1',document_key='gallica:7000001')
    mpath=tmp_path/'historical.json';mpath.write_text(json.dumps(data))
    monkeypatch.setattr(adapter,'allow_development',lambda _:None)
    monkeypatch.setattr(adapter,'load',lambda _: {'real':'settings'})
    monkeypatch.setattr(adapter,'_preflight_real',lambda *_: {'checkpoint_sha256':'a'*64})
    monkeypatch.setattr(adapter,'validate_predictions',lambda *_: {})
    monkeypatch.setattr(adapter,'_verify_loaded_engine',lambda *_:None)
    calls=[]
    def native(settings,model):
        calls.append((settings,model))
        return Engine()
    monkeypatch.setattr(snapshot,'build_engine_from_snapshot',native)
    with pytest.raises(FileNotFoundError):
        recognize(None,mpath,{'a':tmp_path/'crop.png'},predictions()['model'])
    assert calls==[({'real':'settings'},predictions()['model'])]
