#!/bin/sh
# LOT-P07: chain C1+M+A on the 6 drawn pages, every output under AXEL_RUNS (never ~/axel/runs). Heavy steps go through lourd.sh.
# After every step: if any file of ~/axel/runs is newer than the reference, stop.
set -e
R="$HOME/Documents/reviews/babaorum-pipeline/rappel_pages"
export AXEL_RUNS="$R/runs"
# RESUME_AT_C1=1 skips detect and blocks (done 15-16 Oct); axel 39b169f adds the 6 years to route._YEARS
# no "python -I": e12_detect.py imports its sibling runs_root, which -I (no script dir on sys.path) cannot find (axel 52959ef, found 17:29Z)
export PYTHONDONTWRITEBYTECODE=1
cd "$HOME/axel"
L="$HOME/outils-seg/lourd.sh"; K="$HOME/outils-seg/kraken-env/bin/python"; Y="$HOME/outils-seg/layout-env/bin/python"; V=.venv/bin/python
guard() {
  n=$(find "$HOME/axel/runs" -newer "$R/.ref_avant" -type f 2>/dev/null | head -3 | wc -l)
  if [ "$n" -ne 0 ]; then echo "ARRÊT: ~/axel/runs modifié après « $1 »"; find "$HOME/axel/runs" -newer "$R/.ref_avant" -type f | head -5; exit 9; fi
  echo "$(date -u +%FT%TZ) fin $1 : ~/axel/runs intact"
}
libre() { k=$(df -k "$HOME" | awk 'NR == 2 { print $4 }'); [ "$k" -ge 1048576 ] || { echo "moins de 1 Go libre avant $1"; exit 1; }; }
# produced <dir under AXEL_RUNS>: every one of the 6 pages must have a file there, else the step did nothing (e28/e32 skip silently)
produced() { for p in $(python3 -c "import json;print(' '.join(x['page'] for x in json.load(open('$AXEL_RUNS/e12/pages.json'))))"); do [ -s "$AXEL_RUNS/$1/$p.json" ] || { echo "ARRÊT: $1/$p.json absent : l'étape n'a rien produit"; exit 8; }; done; }
step() { name=$1; shift; libre "$name"; echo "$(date -u +%FT%TZ) début $name"; "$@"; guard "$name"; }
[ -n "$RESUME_AT_C1" ] || step detect  $L $Y scripts/e12_detect.py --det ppdl3 --split dev
[ -n "$RESUME_AT_C1" ] || step blocks  $L $K scripts/e12_kraken.py blocks --det ppdl3 --zones --split dev
step c1_read $L $K scripts/e22_read.py --bench e12 --src ppdl3_z --split dev; produced e22/read/ppdl3_z
step e12_read_z $L $K scripts/e12_kraken.py read --src ppdl3_z --split dev; produced e12/read/ppdl3_z
step e28_seg_rest $L $K scripts/e28_fill.py seg --what rest; produced e28/lines/rest_z
step e28_select $V scripts/e28_fill.py select; produced e28/add/a_z
step e28_read $L $K scripts/e28_fill.py read; produced e28/read/mccatmus_press/a_z
step e32_select $V scripts/e32.py select --bench dev
step e32_read $L $K scripts/e32.py read --bench dev
echo "$(date -u +%FT%TZ) FIN chaîne"
