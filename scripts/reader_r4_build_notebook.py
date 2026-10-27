"""Génère scripts/reader_notebooks/kaggle_r4_redo.ipynb (Kaggle, T4).

R4 refait (docs/READER_LOT2A.txt) : poids R3 (r3_best.pt, sortie Kaggle du run
358707154), décodeur CORRIGÉ (témoin B, lot 2a) recopié de
src/babaorum/reader/ancre.py au moment de la génération, porte d'abstention
figée (règle 0 + porte stricte m > τ, τ = maximum des marges de négatifs tune).
(α, Δ) = (0,7 ; 10) transférés de la phase E de R3, sans nouvelle grille.
Phase I : random.Random(4), 200 lignes eval, modes blanc / brouillé / flou fort /
bruit fort, comme R3 ; H-R4 sur les bras SANS porte (blanc et brouillé) ;
comptes avec porte descriptifs. Un seul passage. Aucune ligne de test.

À générer SEULEMENT après le diff du lot 2a relu par pM :
    ~/axel/.venv/bin/python scripts/reader_r4_build_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import reader_r3_build_notebook as r3  # noqa: E402

OUT = HERE / "scripts" / "reader_notebooks" / "kaggle_r4_redo.ipynb"

HEAD = r3.md("""# Babaorum lecteur — R4 refait : invention, décodeur corrigé, porte d'abstention

Protocole : `docs/READER_LOT2A.txt` (témoin B, porte figée). Poids : `r3_best.pt` du run R3 358707154.
(α, Δ) = (0,7 ; 10) transférés. H-R4 sur les bras sans porte. Aucune ligne de test.""")

SETUP = r3.code("""import os, sys, json, gzip, glob, hashlib, shutil, subprocess, time, zipfile, math, random
D, OUT, TMP = '/kaggle/input', '/kaggle/working/reader_r4', '/kaggle/tmp'
os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
T_START = time.time()
L = 32
REPO, REV = 'dots-studio/dots.mocr', 'e539fbb52280393adc081b289ec597430a0f9031'
ALPHA, CORRIDOR = 0.7, 10.0          # transférés de la phase E de R3 (reports/reader_r3_grid_tune300.json)
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)""")

R4_CELL = r3.code(r"""# R4 — porte figée puis invention (un seul passage)
import re
from PIL import ImageFilter
_h = sorted(glob.glob(f'{D}/**/r3_best.pt', recursive=True))
assert len(_h) == 1, f'il faut exactement un r3_best.pt : {_h}'
model = Ancre(len(chars) + 1).to(DEV_T)
print('poids R3 :', _h[0], model.load_state_dict(torch.load(_h[0], map_location=DEV_T), strict=True))
model.eval()
STEM2IT = {it['stem']: it for it in TRAIN + DEV}

def page_of(stem):
    return stem.split('__')[1] if '__' in stem else stem
ALL_DEV = sorted(it['stem'] for it in DEV)
pages = sorted({page_of(s) for s in ALL_DEV}); random.Random(1).shuffle(pages)
TUNE_P = set(pages[: len(pages) // 2])
tune = sorted(s for s in ALL_DEV if page_of(s) in TUNE_P)
evl = sorted(s for s in ALL_DEV if page_of(s) not in TUNE_P)
_sha = hashlib.sha256('\n'.join(sorted({page_of(s) for s in evl})).encode()).hexdigest()
print('tune', len(tune), 'eval', len(evl), 'sha eval pages', _sha[:8])
assert _sha.startswith('996ab0cd'), 'partition eval différente de R1/R3'

def gseed(key):
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)

def perturb(img, mode, key):
    seed = gseed(key)
    if mode == 'blank':
        return Image.new('L', img.size, 255)
    if mode == 'blur_strong':
        return img.filter(ImageFilter.GaussianBlur(max(2.0, img.height / 12)))
    arr = np.asarray(img).copy()
    if mode == 'noise_strong':
        rng = np.random.default_rng(seed)
        return Image.fromarray(np.clip(arr.astype(np.float32) + rng.normal(0, 70, arr.shape), 0, 255).astype(np.uint8))
    if mode == 'bouillie':
        rng = np.random.default_rng(seed)
        return Image.fromarray(rng.integers(0, 256, arr.shape, dtype=np.uint8))
    h, w = arr.shape[0] // 28, arr.shape[1] // 28
    tiles = [arr[r*28:(r+1)*28, c*28:(c+1)*28].copy() for r in range(h) for c in range(w)]
    random.Random(seed).shuffle(tiles)
    for k, t in enumerate(tiles):
        r, c = divmod(k, w); arr[r*28:(r+1)*28, c*28:(c+1)*28] = t
    return Image.fromarray(arr)

def ids2txt(ids):
    return ''.join(chars[c - 1] for c in ids)

enc = load_encoder(L)

@torch.no_grad()
def read_imgs(imgs, joint=True):
    # imgs : liste de PIL ; renvoie (ctc, att, joint, marge) par ligne
    out = []
    for k in range(0, len(imgs), 16):
        ims = [prep(im, HEIGHT, MAXW) for im in imgs[k:k+16]]
        fs = encode_lines(enc, ims, L, WHITE, DEV_T, amp)
        for f in fs:
            x = store(f).float()[None].to(DEV_T)
            n = torch.tensor([x.shape[1]], device=DEV_T)
            logp, mem = model.ctc(x, n)
            tl = int(n[0]) * model.ctc.up
            lp = logp[0, :tl].float().cpu().numpy()
            ctc = greedy(torch.from_numpy(lp), chars)
            prev = torch.zeros(1, 1, dtype=torch.long, device=DEV_T)
            pad = torch.zeros(1, logp.shape[1], dtype=torch.bool, device=DEV_T)
            att = []
            for _ in range(tl // 2 + 2):
                nxt = int(model.dec(prev, mem, pad)[0, -1].argmax())
                if nxt == 0:
                    break
                att.append(chars[nxt - 1]); prev = torch.cat([prev, torch.tensor([[nxt]], device=DEV_T)], 1)
            jt = None
            if joint:
                hs = joint_decode(model, x, x.shape[1], alpha=ALPHA, corridor=CORRIDOR)
                jt = ids2txt(hs[0].ids) if hs else ''
            out.append({'ctc': ctc, 'att': ''.join(att), 'joint': jt, 'margin': readability_margin(lp)})
    return out

# 1. Négatifs de calibration (READER_LOT2A, protocole figé) : une instance Random(5)
rng = random.Random(5)
tpages = sorted({page_of(s) for s in tune}); rng.shuffle(tpages)
negs = []
for p in tpages:
    stems = sorted(s for s in tune if page_of(s) == p)
    s = stems[rng.randrange(len(stems))]
    mode = 'scramble' if rng.random() < 0.5 else 'bouillie'
    negs.append((s, mode))
neg_rows = read_imgs([perturb(CROPS[s], m, s + '#gate') for s, m in negs], joint=False)
for (s, m), r in zip(negs, neg_rows):
    r.update({'stem': s, 'mode': m})
rule0 = [r for r in neg_rows if r['margin'] == -math.inf]
kept = [r for r in neg_rows if r['margin'] != -math.inf]
assert not any(r['margin'] == math.inf for r in neg_rows), '+inf parmi les négatifs : rejet total'
n = len(kept)
tau = max(r['margin'] for r in kept) if kept else -math.inf
GATE = {'negatives': len(neg_rows), 'rule0_excluded': len(rule0), 'n': n, 'tau': tau,
        'modes': {m: sum(1 for r in kept if r['mode'] == m) for m in ('scramble', 'bouillie')},
        'expected_fa_1_over_n_plus_1': 1 / (n + 1) if n else None,
        'alpha95': 1 - 0.05 ** (1 / n) if n and tau > -math.inf else None}
print('porte', GATE, flush=True)

def gated(r):
    return r['margin'] != -math.inf and r['margin'] > tau

# 2. Coût : lignes tune vraies refusées (aucun réglage après coup)
true_rows = read_imgs([CROPS[s] for s in tune], joint=False)
GATE['tune_true_lines'] = len(true_rows)
GATE['tune_true_rejected'] = sum(1 for r in true_rows if not gated(r))
GATE['tune_true_rule0'] = sum(1 for r in true_rows if r['margin'] == -math.inf)
print('coût', {k: GATE[k] for k in ('tune_true_lines', 'tune_true_rejected', 'tune_true_rule0')}, flush=True)

# 3. Invention : mêmes 200 lignes eval que R3 (Random(4)), un passage
def words3(text):
    return len(re.findall(r"[^\W\d_]{2,}", text or '')) >= 3

picks = random.Random(4).sample(evl, 200)
INV = {}
for mode in ('blank', 'scramble', 'blur_strong', 'noise_strong'):
    rows = read_imgs([perturb(CROPS[s], mode, s + '#r4') for s in picks])
    for s, r in zip(picks, rows):
        r['stem'] = s; r['gate_pass'] = gated(r)
    INV[mode] = {'lines': len(rows),
                 'no_gate': {a: sum(words3(r[a]) for r in rows) for a in ('ctc', 'att', 'joint')},
                 'with_gate': {a: sum(words3(r[a]) for r in rows if r['gate_pass']) for a in ('ctc', 'att', 'joint')},
                 'gate_pass': sum(r['gate_pass'] for r in rows), 'rows': rows}
    print(mode, {k: v for k, v in INV[mode].items() if k != 'rows'}, flush=True)
RES = {'alpha': ALPHA, 'corridor': CORRIDOR, 'gate': GATE,
       'invention': {m: {k: v for k, v in d.items() if k != 'rows'} for m, d in INV.items()},
       'h_r4_scope': 'bras SANS porte, blanc et brouillé ; avec porte : descriptif',
       'hours': (time.time() - T_START) / 3600}
json.dump({'neg_rows': neg_rows, 'true_margins': [r['margin'] for r in true_rows], 'inv': INV},
          open(f'{OUT}/r4_rows.json', 'w'), ensure_ascii=False)
json.dump(RES, open(f'{OUT}/r4_results.json', 'w'), indent=1)
print(json.dumps(RES, indent=1, ensure_ascii=False))""")


def main() -> None:
    src = json.loads(r3.SRC.read_text())
    cells = src["cells"]
    keep = [cells[i] for i in (2, 3, 4, 5, 6, 7, 8, 9, 10)]
    for c in keep:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    ancre = (HERE / "src" / "babaorum" / "reader" / "ancre.py").read_text()
    ancre = ancre.replace("from __future__ import annotations\n", "")
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [HEAD, SETUP] + keep + [r3.code(r3.ANCRE_CELL_TMPL.format(src=ancre)), R4_CELL],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(OUT, len(nb["cells"]), "cellules")


if __name__ == "__main__":
    main()
