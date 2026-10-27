# Source-only edition

This edition is prepared from the research implementation with a fresh Git
history. It is intended to separate reusable source and synthetic examples from
corpus content and internal experiment records.

Included:

- Python source, experiment drivers, and unexecuted source notebooks;
- algorithm configurations and the G0 input schema;
- XML schemas with their existing notices and provenance;
- test source, a default synthetic test selection, and simple generated fixtures.

Excluded:

- real page images and reference transcriptions;
- per-line recognition outputs and experimental result reports;
- real usage, exposure, selection, and consultation registries;
- internal review notes, journal attestations, and local storage inventories;
- model checkpoints, corpus archives, and run directories;
- the previous Git history, branches, and tags.

`SOURCE-MANIFEST.json` records the retained original source files by path and
SHA-256. The public test selection and packaging configuration are edition-specific.
The scientific implementations and their existing fail-closed behavior have not
been relaxed to compensate for omitted inputs.

Some research code retains local-layout assumptions or references to separately
maintained dependencies. These are integration prerequisites, not bundled data.
Only the documented default suite is claimed to run without the private resources.

The image fixtures are small uniform white or red patches generated for tests.
The JSON fixtures contain short synthetic strings and dummy identifiers/hashes.
They are not scans or reference transcriptions from the research corpora.

The repository guard rejects corpus/result directories, real G0 tables, stored
notebook outputs, oversized files, and known credential formats. It is a useful
regression check, not a guarantee against every possible disclosure.
