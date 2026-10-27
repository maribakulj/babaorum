#!/bin/sh
# Négatifs perturbés du protocole de porte : 200 lignes photo_1983 × 4 modes, kraken puis A1.
cd ~/babaorum
for m in blank scramble blur_strong noise_strong; do
  ~/outils-seg/kraken-env/bin/python scripts/reader_kraken_read.py --items reports/reader_items_photo1983_pert200.jsonl --perturb $m --out reports/reader_kraken_photo1983_$m.jsonl.gz
done
for m in blank scramble blur_strong noise_strong; do
  ~/axel/.venv/bin/python scripts/reader_a1_read.py --items reports/reader_items_photo1983_pert200.jsonl --perturb $m --out reports/reader_a1_photo1983_$m.jsonl.gz
done
