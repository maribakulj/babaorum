# Installation

The default test environment is Python 3.12 with
`config/requirements-ci.txt`. It uses pytest, NumPy, Pillow, RapidFuzz, and lxml.

```sh
python -m pip install -r config/requirements-ci.txt
mkdir -p /tmp/babaorum-public-tests
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/babaorum-public-tests/run
```

The default test paths are listed in `pyproject.toml`. They cover synthetic
contracts, page graphs, XML export, CTC word geometry, canonical identifiers,
text-overlap calculation, conversion, and fail-closed loading of omitted data.
Pre-existing checks that load the historical sealing registry, the real BBVLM
delivery, or the external normalization reference are marked `private_data` and
excluded from the default suite, including when those resources happen to exist
on the developer's machine. The scientific source is unchanged; no synthetic
replacement of the global sealing registry is installed.

The other test modules are preserved as research source. Some require corpus
metadata, private registries, external packages, models, or historical results.
Running every module explicitly is an integration task with those prerequisites;
it is not the default source-only test suite.

Model-based experiments have separate environments, typically documented in
their source notebooks. Some import the separately maintained `axel` project.
No weights, credentials, or dataset access are supplied with this edition.

## Data-dependent admission

The G0 implementation and input schema are retained. Its real registry, usage
tables, journal attestations, and selection records are omitted. Anchors remain
part of the source; absent or mismatched inputs must continue to refuse admission.
The default tests verify that this source edition cannot admit a page from its
missing private inputs.

`config/g0_roots.local.example.json` shows the root configuration format. Use a
local `config/g0_roots.local.json` only with separately provisioned and reviewed
data. The local file is excluded from Git. Supplying a directory alone does not
satisfy the cryptographic anchors.
