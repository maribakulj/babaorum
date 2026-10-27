import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from babaorum.adapters.pero_snapshot import _construct_from_bytes
from babaorum.contracts import ContractError


class Device:
    def __init__(self, name):
        self.type = name


class ScriptedStub:
    def __init__(self, data):
        self.data = data

    def to(self, device):
        self.device = device
        return self


class TorchStub:
    def __init__(self, device_factory=Device):
        self.device = device_factory
        self.loaded = []

        def load(stream, *args, **kwargs):
            data = stream.read()
            self.loaded.append((data, kwargs.get('map_location')))
            return ScriptedStub(data)

        self.jit = SimpleNamespace(load=load)


class FakePEROBase:
    def __init__(self, json_path, device, batch_size=8):
        self.config = json.loads(Path(json_path).read_bytes())
        self.device = device
        self.batch_size = batch_size
        self.checkpoint = self.config['checkpoint']
        if not Path(self.checkpoint).is_absolute():
            self.checkpoint = str((Path(json_path).parent / self.checkpoint).resolve())
        self.line_px_height = self.config.get('line_px_height', 40)
        self.line_vertical_scale = self.config.get('line_vertical_scale', 1.0)
        self.net_name = self.config.get('net_name', 'fake')
        self.characters = list(self.config['characters'])
        self._load_exported_model()
        self.characters += ['\u200b']

    def _load_exported_model(self):
        raise AssertionError('snapshot override must own the model load')


def snapshot_inputs(tmp_path, checkpoint='weights/reader.pt'):
    config_bytes = json.dumps({'checkpoint': checkpoint, 'characters': ['a', 'ſ'],
                               'line_px_height': 40, 'line_vertical_scale': 1.0,
                               'net_name': 'fake'}).encode()
    checkpoint_bytes = b'tiny synthetic scripted-model bytes'
    hashes = (hashlib.sha256(config_bytes).hexdigest(),
              hashlib.sha256(checkpoint_bytes).hexdigest())
    expected = {'config_sha256': hashes[0], 'checkpoint_sha256': hashes[1],
                'checkpoint_path': str((tmp_path / 'weights/reader.pt.cpu').resolve()),
                'device': 'cpu', 'code_revision': 'a' * 40}
    return config_bytes, checkpoint_bytes, hashes, expected


def construct(tmp_path, *, config_bytes=None, checkpoint_bytes=None,
              config_hash=None, checkpoint_hash=None, torch=None, base=FakePEROBase):
    original_config, original_checkpoint, hashes, expected = snapshot_inputs(tmp_path)
    config_bytes = original_config if config_bytes is None else config_bytes
    checkpoint_bytes = original_checkpoint if checkpoint_bytes is None else checkpoint_bytes
    config_hash = hashes[0] if config_hash is None else config_hash
    checkpoint_hash = hashes[1] if checkpoint_hash is None else checkpoint_hash
    torch = TorchStub() if torch is None else torch
    engine = _construct_from_bytes(
        engine_base_class=base, torch_module=torch,
        config_bytes=config_bytes, checkpoint_bytes=checkpoint_bytes,
        config_sha256=config_hash, checkpoint_sha256=checkpoint_hash,
        source_config_path=tmp_path/'ocr_engine.json',
        original_checkpoint=Path(expected['checkpoint_path']),
        settings={'device': 'cpu', 'batch_size': 2,
                  'checkpoint_resolution':'pytorch_cpu_suffix_v1'}, expected=expected)
    return engine, torch, expected, config_bytes, checkpoint_bytes


def test_constructs_engine_with_exact_config_and_checkpoint_bytes(tmp_path):
    engine, torch, expected, config_bytes, checkpoint_bytes = construct(tmp_path)
    assert torch.loaded == [(checkpoint_bytes, engine.device)]
    assert engine.model.data == checkpoint_bytes
    assert engine.model.device is engine.device
    assert engine.device.type == 'cpu' and engine.batch_size == 2
    assert engine.checkpoint == expected['checkpoint_path']
    assert engine.config == json.loads(config_bytes)
    assert engine.characters == ['a', 'ſ', '\u200b']
    assert engine.babaorum_provenance['checkpoint_sha256'] == expected['checkpoint_sha256']
    assert engine.babaorum_provenance['loader'] == 'torchscript_bytes_v1'


def test_bad_snapshot_hash_fails_before_engine_or_loader(tmp_path):
    config_bytes, checkpoint_bytes, hashes, expected = snapshot_inputs(tmp_path)
    torch = TorchStub()

    class NeverConstructed:
        def __init__(self, *args, **kwargs):
            pytest.fail('constructor should not run')

    with pytest.raises(ContractError, match='config snapshot hash'):
        _construct_from_bytes(engine_base_class=NeverConstructed, torch_module=torch,
                              config_bytes=config_bytes, checkpoint_bytes=checkpoint_bytes,
                              config_sha256='0' * 64, checkpoint_sha256=hashes[1],
                              source_config_path=tmp_path/'ocr_engine.json',
                              original_checkpoint=Path(expected['checkpoint_path']),
                              settings={'device': 'cpu', 'batch_size': 2,
                                        'checkpoint_resolution':'pytorch_cpu_suffix_v1'}, expected=expected)
    assert torch.loaded == []


def test_rejects_non_cpu_before_torchscript_load(tmp_path):
    # Simulate a native base constructor that does not honor the requested CPU.
    class WrongDeviceTorch(TorchStub):
        def __init__(self):
            super().__init__(lambda name: Device('cuda'))

    torch = WrongDeviceTorch()
    config_bytes, checkpoint_bytes, hashes, expected = snapshot_inputs(tmp_path)
    with pytest.raises(ContractError, match='explicitly target CPU'):
        _construct_from_bytes(engine_base_class=FakePEROBase, torch_module=torch,
                              config_bytes=config_bytes, checkpoint_bytes=checkpoint_bytes,
                              config_sha256=hashes[0], checkpoint_sha256=hashes[1],
                              source_config_path=tmp_path/'ocr_engine.json',
                              original_checkpoint=Path(expected['checkpoint_path']),
                              settings={'device': 'cpu', 'batch_size': 2,
                                        'checkpoint_resolution':'pytorch_cpu_suffix_v1'}, expected=expected)
    assert torch.loaded == []


def test_invalid_checkpoint_config_fails_closed_before_model_load(tmp_path):
    config_bytes, checkpoint_bytes, hashes, expected = snapshot_inputs(tmp_path)
    config_bytes = json.dumps({'checkpoint': 'other/reader.pt', 'characters': ['a'],
                               'line_px_height': 40, 'line_vertical_scale': 1.0,
                               'net_name': 'fake'}).encode()
    torch = TorchStub()
    with pytest.raises(ContractError, match='does not resolve to the captured checkpoint'):
        _construct_from_bytes(engine_base_class=FakePEROBase, torch_module=torch,
                              config_bytes=config_bytes, checkpoint_bytes=checkpoint_bytes,
                              config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                              checkpoint_sha256=hashes[1],
                              source_config_path=tmp_path/'ocr_engine.json',
                              original_checkpoint=Path(expected['checkpoint_path']),
                              settings={'device': 'cpu', 'batch_size': 2,
                                        'checkpoint_resolution':'pytorch_cpu_suffix_v1'}, expected=expected)
    assert torch.loaded == []


def test_engine_postload_contract_mismatch_rejected(tmp_path):
    class WrongBatchBase(FakePEROBase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.batch_size += 1

    with pytest.raises(ContractError, match='batch size differs'):
        construct(tmp_path, base=WrongBatchBase)
