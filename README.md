# Babaorum

Experimental OCR for historical newspapers and heritage documents.

Babaorum explores page layout analysis, line extraction, text recognition,
CTC-guided decoding, abstention, and ALTO XML export.

## Research directions

- Detecting lines and reconstructing their geometry.
- Recognizing text with line encoders and Connectionist Temporal Classification (CTC).
- Guiding language-model decoding with CTC scores.
- Studying abstention and unsupported transcriptions.
- Evaluating geometry, recognized text, and exported documents separately.

## Getting started

Use Python 3.12 and the lightweight dependencies:

```sh
python -m pip install -r config/requirements-ci.txt
python -m pytest -q -p no:cacheprovider
```

The default suite uses synthetic examples. It does not download models or read
the private corpora, registries, or reference transcriptions.

See [installation](docs/INSTALLATION.md) and the [source edition scope](docs/SOURCE_EDITION.md).

## Project status

Babaorum is experimental and under active development. Inclusion of a prototype
does not imply its adoption in the reference pipeline. Research scripts and
notebooks can require additional software and separately supplied resources.
Their presence is not a claim that every experiment can be reproduced from this
source-only edition.

## Data and models

This edition contains source code, algorithm configurations, schemas, and
synthetic test fixtures. It does not redistribute corpora, real reference
transcriptions, experimental outputs, model weights, or private provenance logs.

The data-dependent admission checks retain their original behavior: missing
anchored registries do not admit pages. They are not replaced with permissive
defaults for the public edition.

## License status

A project-wide code license has not yet been selected. Existing third-party
notices are retained; see [licensing and schema notices](docs/LICENSING.md).
