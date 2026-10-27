#!/bin/sh
# T2 : treillis perturbés (100 lignes press/dev × 4 modes), kraken puis A1.
# À lancer UNIQUEMENT par : ~/outils-seg/lourd.sh sh scripts/reader_t2_pert.sh
cd ~/babaorum
for m in blank scramble blur_strong noise_strong; do
  ~/outils-seg/kraken-env/bin/python scripts/reader_kraken_read.py --items reports/reader_items_dev_pert100.jsonl --perturb $m --out reports/reader_pert_kraken_$m.jsonl.gz || exit 1
done
for m in blank scramble blur_strong noise_strong; do
  ~/axel/.venv/bin/python scripts/reader_a1_read.py --items reports/reader_items_dev_pert100.jsonl --perturb $m --out reports/reader_pert_a1_$m.jsonl.gz || exit 1
done
echo FINI
