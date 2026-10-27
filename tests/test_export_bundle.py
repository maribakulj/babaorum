import hashlib
import json
from pathlib import Path
import sys

import pytest

from babaorum.contracts import ContractError
from babaorum.export_bundle import export_bundle
from test_export_page import exportable_graph


def source(tmp_path):
    p = tmp_path / 'source.json'
    p.write_text(json.dumps(exportable_graph(),ensure_ascii=False))
    return p


def test_bundle_keeps_exact_graph_hash_and_valid_xml(tmp_path):
    src = source(tmp_path)
    out = tmp_path / 'bundle'
    receipt = export_bundle(src, out, format='page', image_filename='scan.tif', quantization='nearest_pixel')
    assert receipt['accepted'] and receipt['xsd_validated']
    assert receipt['image_bytes_verified'] is False
    assert receipt['source_graph_file_sha256'] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert receipt['sidecar_sha256'] == hashlib.sha256((out/'graph.json').read_bytes()).hexdigest()
    assert receipt['xml_sha256'] == hashlib.sha256((out/'page.xml').read_bytes()).hexdigest()
    assert json.loads((out/'graph.json').read_bytes()) == json.loads(src.read_bytes())
    assert json.loads((out/'export.json').read_bytes()) == receipt
    assert (out.stat().st_mode & 0o777) == 0o700
    assert all((p.stat().st_mode & 0o777) == 0o600 for p in out.iterdir())
    assert not list(tmp_path.glob('.bundle.tmp-*'))


def test_rejected_alto_has_reviewable_report_and_no_xml(tmp_path):
    src = source(tmp_path)
    data = json.loads(src.read_bytes())
    data['lines'][0]['text']['value'] += ', punctuation missing from word'
    src.write_text(json.dumps(data))
    out = tmp_path/'rejected'
    receipt = export_bundle(src,out,format='alto',image_filename='scan.tif',physical_img_nr=2)
    assert not receipt['accepted'] and not receipt['xsd_validated']
    assert receipt['xml_file'] is receipt['xml_sha256'] is None
    assert sorted(p.name for p in out.iterdir()) == ['export.json','graph.json']


def test_existing_outputs_and_inapplicable_options_are_refused(tmp_path):
    src = source(tmp_path)
    out = tmp_path/'bundle';out.mkdir();sentinel=out/'keep';sentinel.write_text('keep')
    with pytest.raises(ContractError,match='already exists'):
        export_bundle(src,out,format='page',image_filename='scan.tif')
    assert sentinel.read_text() == 'keep'
    with pytest.raises(ContractError,match='only to ALTO'):
        export_bundle(src,tmp_path/'bad',format='page',image_filename='scan.tif',physical_img_nr=2)
    assert not (tmp_path/'bad').exists()


def test_concurrent_nonempty_target_is_never_replaced(tmp_path, monkeypatch):
    import babaorum.export_bundle as module
    src = source(tmp_path)
    out = tmp_path / 'raced'
    original = module._rename_noreplace

    def create_target_then_publish(stage, target):
        target = Path(target)
        target.mkdir()
        sentinel = target / 'concurrent-owner.txt'
        sentinel.write_text('must survive')
        original(stage, target)

    monkeypatch.setattr(module, '_rename_noreplace', create_target_then_publish)
    with pytest.raises(ContractError, match='appeared during export'):
        export_bundle(src, out, format='page', image_filename='scan.tif', quantization='nearest_pixel')
    assert (out / 'concurrent-owner.txt').read_text() == 'must survive'
    assert sorted(p.name for p in out.iterdir()) == ['concurrent-owner.txt']
    assert not list(tmp_path.glob('.raced.tmp-*'))


def test_parent_fsync_failure_reports_bundle_already_published(tmp_path, monkeypatch):
    import babaorum.export_bundle as module
    src = source(tmp_path)
    out = tmp_path / 'durability'
    original = module._sync_dir

    def fail_parent(path):
        if Path(path) == tmp_path:
            raise OSError('simulated directory fsync failure')
        return original(path)

    monkeypatch.setattr(module, '_sync_dir', fail_parent)
    with pytest.raises(ContractError, match='bundle was published'):
        export_bundle(src, out, format='page', image_filename='scan.tif', quantization='nearest_pixel')
    assert (out / 'page.xml').is_file()
    assert (out / 'graph.json').is_file()
    assert (out / 'export.json').is_file()


def test_cli_emits_bundle_and_reports_rejected_projection(tmp_path,monkeypatch,capsys):
    from babaorum.__main__ import main
    src=source(tmp_path)
    monkeypatch.setattr(sys,'argv',['babaorum','export-page',str(src),'--format','alto',
                                   '--image-filename','scan.tif','--physical-img-nr','2',
                                   '--output-dir',str(tmp_path/'alto')])
    main()
    assert json.loads(capsys.readouterr().out)['xsd_validated'] is True
    data=json.loads(src.read_text());data['lines'][0]['text']['value']='not the word'
    src.write_text(json.dumps(data))
    sys.argv[-1]=str(tmp_path/'refused')
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert (tmp_path/'refused'/'export.json').is_file()


def test_concurrent_empty_target_is_never_replaced(tmp_path, monkeypatch):
    # A plain rename(2) would silently replace an EMPTY directory; the
    # no-replace primitive must refuse it too, and leave it untouched.
    import babaorum.export_bundle as module
    src = source(tmp_path)
    out = tmp_path / 'raced_empty'
    original = module._rename_noreplace

    def create_empty_target_then_publish(stage, target):
        Path(target).mkdir()
        original(stage, target)

    monkeypatch.setattr(module, '_rename_noreplace', create_empty_target_then_publish)
    with pytest.raises(ContractError, match='appeared during export'):
        export_bundle(src, out, format='page', image_filename='scan.tif', quantization='nearest_pixel')
    assert out.is_dir() and list(out.iterdir()) == []
    assert not list(tmp_path.glob('.raced_empty.tmp-*'))


@pytest.mark.parametrize('fmt,kwargs', [
    ('page', dict(quantization='nearest_pixel')),
    ('alto', dict(physical_img_nr=2)),
])
def test_xsd_failure_prevents_bundle_publication(tmp_path, monkeypatch, fmt, kwargs):
    # A projection that is accepted by the exporter but invalid against the
    # pinned XSD must publish nothing: no directory, no temp stage.
    from dataclasses import replace
    import babaorum.export_bundle as module
    from babaorum.export_page import ExportError

    exporter_name = 'export_pagexml' if fmt == 'page' else 'export_alto'
    exporter = getattr(module, exporter_name)

    def invalid_xml(*args, **kw):
        result = exporter(*args, **kw)
        assert result.accepted
        return replace(result, xml=result.xml.replace('>', ' bogus_attribute="1">', 1))

    monkeypatch.setattr(module, exporter_name, invalid_xml)
    src = source(tmp_path)
    out = tmp_path / 'xsd_rejected'
    with pytest.raises(ExportError, match='XSD validation failed'):
        export_bundle(src, out, format=fmt, image_filename='scan.tif', **kwargs)
    assert not out.exists()
    assert not list(tmp_path.glob('.xsd_rejected.tmp-*'))
