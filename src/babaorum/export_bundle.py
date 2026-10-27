"""Publish a PAGE/ALTO projection together with its loss report and source graph."""
from dataclasses import asdict
import hashlib
import json
import os
import sys
import ctypes
import errno
from pathlib import Path
import shutil
import tempfile

from .contracts import ContractError, parse, require
from .export_page import export_alto, export_pagexml, validate_export_xsd


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _write_synced(path, raw):
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o600)


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _rename_noreplace(source, target):
    """Atomically publish a directory only if target does not exist."""
    libc = ctypes.CDLL(None, use_errno=True)
    source_b, target_b = os.fsencode(source), os.fsencode(target)
    if sys.platform == 'darwin':
        function = getattr(libc, 'renameatx_np', None)
        if function is None:
            raise ContractError('atomic no-replace directory publication is unsupported: renameatx_np unavailable')
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        at_fdcwd, no_replace_flag = -2, 0x00000004  # Darwin: AT_FDCWD, RENAME_EXCL
        result = function(at_fdcwd, source_b, at_fdcwd, target_b, no_replace_flag)
    elif sys.platform.startswith('linux'):
        function = getattr(libc, 'renameat2', None)
        if function is None:
            raise ContractError('atomic no-replace directory publication is unsupported: renameat2 unavailable')
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        at_fdcwd, no_replace_flag = -100, 1  # Linux: AT_FDCWD, RENAME_NOREPLACE
        result = function(at_fdcwd, source_b, at_fdcwd, target_b, no_replace_flag)
    else:
        raise ContractError(f'atomic no-replace directory publication unsupported on {sys.platform}')
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise ContractError('export output directory appeared during export; existing target preserved')
    raise OSError(error, os.strerror(error), os.fspath(target))


def export_bundle(graph_path, output_dir, *, format, image_filename,
                  physical_img_nr=None, mode='strict', quantization='reject'):
    """Write a new directory; refuse overwrite and validate any XML before publish.

    A rejected projection still has a reviewable bundle (graph + loss report),
    with xml_file=null, accepted=false. It never emits a substitute XML file.
    Source and encoded-image identity are distinct: this operation hashes the
    graph bytes, and does not claim to have opened or verified the scan image.
    """
    target = Path(output_dir)
    require(not target.exists() and not target.is_symlink(), 'export output directory already exists')
    source_bytes = Path(graph_path).read_bytes()
    graph = parse(source_bytes)
    if format == 'page':
        require(physical_img_nr is None, 'physical_img_nr applies only to ALTO')
        result = export_pagexml(graph, image_filename=image_filename, mode=mode, quantization=quantization)
        xml_name = 'page.xml'
    elif format == 'alto':
        require(quantization == 'reject', 'PAGE quantization option does not apply to ALTO')
        require(physical_img_nr is not None, 'ALTO requires explicit physical_img_nr')
        result = export_alto(graph, image_filename=image_filename, physical_img_nr=physical_img_nr, mode=mode)
        xml_name = 'alto.xml'
    else:
        raise ContractError('format must be page or alto')
    xml_bytes = None
    if result.accepted:
        validate_export_xsd(result)  # Always offline, against vendored pinned schemas.
        xml_bytes = ('<?xml version="1.0" encoding="UTF-8"?>\n' + result.xml).encode('utf-8')
    graph_bytes = result.sidecar_json.encode('utf-8')
    require(_hash(graph_bytes) == result.sidecar_sha256, 'sidecar hash inconsistent')
    receipt = {k: v for k, v in asdict(result).items() if k not in ('xml', 'sidecar_json')}
    receipt['losses'] = list(receipt['losses'])
    receipt.update(
        schema_version='babaorum.export_bundle.v1',
        role='ocr_projection_bundle', data_authorization=None,
        source_graph_file_sha256=_hash(source_bytes),
        sidecar_file='graph.json', xml_file=xml_name if xml_bytes is not None else None,
        xml_sha256=_hash(xml_bytes) if xml_bytes is not None else None,
        xsd_validated=xml_bytes is not None,
        image_bytes_verified=False,
        mode=mode, image_filename=image_filename, physical_img_nr=physical_img_nr,
    )
    report_bytes = (json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + target.name + '.tmp-', dir=target.parent))
    try:
        os.chmod(stage, 0o700)
        _write_synced(stage / 'graph.json', graph_bytes)
        if xml_bytes is not None:
            _write_synced(stage / xml_name, xml_bytes)
        _write_synced(stage / 'export.json', report_bytes)
        _sync_dir(stage)
        # Do not replace an existing bundle. The marker is written before the
        # single directory rename, so consumers never see a half-filled bundle.
        require(not target.exists() and not target.is_symlink(), 'export output directory appeared during export')
        _rename_noreplace(stage, target)
        try:
            _sync_dir(target.parent)
        except OSError as exc:
            raise ContractError(f'bundle was published at {target}, but parent-directory fsync failed: {exc}') from exc
    finally:
        if stage.exists():
            shutil.rmtree(stage)  # This function's own unpublished temporary directory only.
    return receipt
