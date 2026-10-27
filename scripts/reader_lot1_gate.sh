#!/bin/sh
# Lot 1b (15/10) : treillis du jeu de réglage photo_1983 + négatifs perturbés, puis calibration de θ_D.
# À lancer UNIQUEMENT par : ~/outils-seg/lourd.sh sh scripts/reader_lot1_gate.sh
set -e
cd ~/babaorum
~/axel/.venv/bin/python - <<'PY'
import json, random
rows = [json.loads(l) for l in open('/Users/marcel/axel/data/prepared/press/train.jsonl')]
ph = [r for r in rows if r.get('group') == 'photo_1983']
open('reports/reader_items_photo1983.jsonl', 'w').write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in ph))
lines = open('reports/reader_items_photo1983.jsonl').readlines()
open('reports/reader_items_photo1983_pert200.jsonl', 'w').write(''.join(random.Random(21).sample(lines, 200)))
print('items', len(ph))
PY
~/outils-seg/kraken-env/bin/python scripts/reader_kraken_read.py --items reports/reader_items_photo1983.jsonl --out reports/reader_kraken_photo1983.jsonl.gz
~/axel/.venv/bin/python scripts/reader_a1_read.py --items reports/reader_items_photo1983.jsonl --out reports/reader_a1_photo1983.jsonl.gz
sh scripts/reader_gate_pert.sh
~/axel/.venv/bin/python scripts/reader_gate_calib.py
echo FINI
