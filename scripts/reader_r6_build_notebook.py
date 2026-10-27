"""Génère scripts/reader_notebooks/kaggle_r6_<phase>.ipynb (docs/READER_R6_PREREG.txt).

phase = SMOKE (B0 puis B3, 300 pas chacun : débit et perte seulement, aucun CER)
ou un bras B0..B5 (S pas, choix du point sur la moitié TUNE, mesure finale ligne
par ligne sur tune et eval, treillis du dev écrits pour H-R6c).
Cellules de recadrage reprises du carnet axel kaggle_e17_vitctc (MD5 contrôlés),
architectures = src/babaorum/reader/r6_arch.py, métrique = axel/reader/metric.py.

    ~/axel/.venv/bin/python scripts/reader_r6_build_notebook.py SMOKE
    ~/axel/.venv/bin/python scripts/reader_r6_build_notebook.py B0 --steps S
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
E17 = Path.home() / "axel/notebooks/kaggle_e17_vitctc.ipynb"
METRIC = Path.home() / "axel/src/axel/reader/metric.py"
ARCH = HERE / "src/babaorum/reader/r6_arch.py"


def code(s: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s}


SETUP = """import os, sys, json, gzip, glob, hashlib, shutil, subprocess, time, zipfile, math, random, re, unicodedata
D, TMP, OUT = '/kaggle/input', '/kaggle/tmp', '/kaggle/working/r6'
os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
PHASE = {phase!r}          # 'SMOKE' ou un bras
STEPS = {steps}            # S (fixé après la fumée, identique pour tous les bras)
SMOKE_STEPS = 300
EVAL_EVERY = 1000          # pas entre deux mesures sur TUNE (choix du point)
BATCH, LR, WD, WARMUP, CLIP, SEED = 32, 1e-3, 0.01, 1000, 1.0, 0
T_START = time.time()
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)"""

TRAIN_CELL = r'''import numpy as np, torch, torch.nn as nn
from PIL import ImageFilter
DEV_T = 'cuda'
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
# codec : celui d'A1 (timing.json de la sortie axel-e17-vitctc) s'il est attaché, sinon caractères du train
_t = [p for p in sorted(glob.glob(f'{D}/**/timing.json', recursive=True)) if 'chars' in json.load(open(p))]
clean = lambda t: unicodedata.normalize('NFC', ' '.join(t.split()))
assert len(_t) <= 1, f'plusieurs timing.json avec chars : {_t}'
CHARS = json.load(open(_t[0]))['chars'] if _t else sorted({c for it in TRAIN for c in clean(it['truth'])})
print('codec :', _t[0] if _t else 'caractères du train', len(CHARS))
IDX = {c: i + 1 for i, c in enumerate(CHARS)}

# partition tune/eval de press/dev (graine 1, sha256 eval 996ab0cd…, READER_R3_PREREG)
def page_of(stem):
    return stem.split('__')[1] if '__' in stem else stem
pages = sorted({page_of(it['stem']) for it in DEV}); random.Random(1).shuffle(pages)
TUNE_P = set(pages[: len(pages) // 2])
TUNE = [it for it in DEV if page_of(it['stem']) in TUNE_P]
EVAL = [it for it in DEV if page_of(it['stem']) not in TUNE_P]
_h = lambda xs: hashlib.sha256('\n'.join(sorted(xs)).encode()).hexdigest()
print('tune', len(TUNE), 'eval', len(EVAL), 'sha eval pages', _h({page_of(it['stem']) for it in EVAL})[:8])

def augment(im, rng):
    # chacune avec probabilité 0,3 (pré-enregistrement)
    if rng.random() < 0.3:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    if rng.random() < 0.3:
        a = np.asarray(im, np.float32) + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(3, 12), (im.height, im.width))
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.3:
        im = im.filter(ImageFilter.MinFilter(3) if rng.random() < 0.5 else ImageFilter.MaxFilter(3))
    if rng.random() < 0.3:
        im = im.rotate(rng.uniform(-2, 2), resample=Image.BILINEAR, expand=False, fillcolor=255)
    if rng.random() < 0.3:
        f = rng.uniform(-0.1, 0.1); d = int(round(abs(f) * im.height))
        if d > 0:
            if f > 0:
                c = Image.new('L', (im.width, im.height + 2 * d), 255); c.paste(im, (0, d)); im = c
            else:
                im = im.crop((0, d // 2, im.width, im.height - d // 2)) if im.height - d > 4 else im
    return im

def prep(im, h_in, stride):
    w = max(stride, round(im.width * h_in / max(1, im.height)))
    w = min(w, 4000)
    w = int(math.ceil(w / stride) * stride)
    g = im.convert('L').resize((w, h_in), Image.BICUBIC)
    return torch.from_numpy(np.asarray(g, np.float32) / 255.0)[None]

def batchify(ims):
    W = max(x.shape[-1] for x in ims)
    xb = torch.ones(len(ims), 1, ims[0].shape[1], W)
    for i, x in enumerate(ims):
        xb[i, :, :, : x.shape[-1]] = x
    return xb, torch.tensor([x.shape[-1] for x in ims])

def targets(its):
    ys = [[IDX[c] for c in clean(it['truth']) if c in IDX] for it in its]
    return ys

def greedy(lp):
    a = lp.argmax(-1).tolist(); out, prev = [], -1
    for c in a:
        if c != prev and c != 0:
            out.append(CHARS[c - 1])
        prev = c
    return ''.join(out)

@torch.no_grad()
def read(model, its, keep_logp=False):
    # évaluation LIGNE PAR LIGNE (règle R6 : indépendance au lot exacte)
    model.eval(); rows = []
    for it in its:
        x, w = batchify([prep(CROPS[it['stem']], model.h_in, model.stride)])
        with torch.autocast('cuda', dtype=torch.float16):
            lp, L = model(x.to(DEV_T), w.to(DEV_T))[:2]
        lp = lp[0, : int(L[0])].float().cpu()
        r = {'stem': it['stem'], 'truth': it['truth'], 'hyp': greedy(lp)}
        if keep_logp:
            r['logp'] = lp.half().numpy().tolist()
        rows.append(r)
    model.train()
    return rows

def cer_rows(rows):
    return score([(r['hyp'], r['truth']) for r in rows])['cer_d150']

def run(arm, steps, smoke=False):
    torch.manual_seed(SEED); rng = random.Random(SEED)
    model = ARMS[arm](len(CHARS) + 1).to(DEV_T)
    desc = model.describe()
    print(desc.split('\n')[0])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / WARMUP if s < WARMUP else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - WARMUP) / max(1, steps - WARMUP)))))  # cosinus après l'échauffement
    scaler = torch.amp.GradScaler()
    ctc = nn.CTCLoss(zero_infinity=True)
    order = sorted(TRAIN, key=lambda it: CROPS[it['stem']].width / max(1, CROPS[it['stem']].height))
    chunks = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
    log = open(f'{OUT}/{arm}{"_smoke" if smoke else ""}_log.jsonl', 'w')
    best, step, t0, losses, inf_lines = 1e9, 0, time.time(), [], 0
    while step < steps:
        rng.shuffle(chunks)
        for b in chunks:
            if step >= steps:
                break
            ys = targets(b)
            keep = [i for i, y in enumerate(ys) if y]
            b = [b[i] for i in keep]; ys = [ys[i] for i in keep]
            x, w = batchify([prep(augment(CROPS[it['stem']], rng), model.h_in, model.stride) for it in b])
            with torch.autocast('cuda', dtype=torch.float16):
                out = model(x.to(DEV_T), w.to(DEV_T), with_aux=True) if arm == 'B5' else model(x.to(DEV_T), w.to(DEV_T))
            lp, L = out[0].float(), out[1]
            tg = torch.tensor([c for y in ys for c in y]); tl = torch.tensor([len(y) for y in ys])
            inf_lines += int((L.cpu() < tl).sum())
            loss = ctc(lp.transpose(0, 1), tg, L.cpu(), tl)
            if arm == 'B5':
                loss = (1 - model.aux_weight) * loss + model.aux_weight * ctc(out[2].float().transpose(0, 1), tg, L.cpu(), tl)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), CLIP); scaler.step(opt); scaler.update(); sched.step()
            step += 1; losses.append(loss.item())
            if step % 100 == 0:
                row = {'step': step, 'loss': float(np.mean(losses[-100:])), 'steps_per_s': step / (time.time() - t0),
                       'lines_T_lt_U': inf_lines}
                log.write(json.dumps(row) + '\n'); log.flush(); print(row, flush=True)
            if not smoke and step % EVAL_EVERY == 0:
                c = cer_rows(read(model, TUNE))
                log.write(json.dumps({'step': step, 'tune_cer_d150': c}) + '\n'); log.flush(); print('tune', step, c, flush=True)
                if c < best - 1e-9:
                    best = c; torch.save(model.state_dict(), f'{OUT}/{arm}_best.pt')
    res = {'arm': arm, 'steps': step, 'seconds': time.time() - t0, 'steps_per_s': step / (time.time() - t0),
           'params': sum(p.numel() for p in model.parameters()), 'describe': desc, 'lines_T_lt_U': inf_lines}
    if not smoke:
        model.load_state_dict(torch.load(f'{OUT}/{arm}_best.pt'))
        tune_rows = read(model, TUNE, keep_logp=True); eval_rows = read(model, EVAL, keep_logp=True)
        res.update({'best_tune_cer_d150': best, 'tune_cer_d150': cer_rows(tune_rows), 'eval_cer_d150': cer_rows(eval_rows)})
        with gzip.open(f'{OUT}/{arm}_dev.jsonl.gz', 'wt') as g:
            for r in tune_rows + eval_rows:
                g.write(json.dumps(r, ensure_ascii=False) + '\n')
    json.dump(res, open(f'{OUT}/{arm}{"_smoke" if smoke else ""}_result.json', 'w'), indent=1)
    return res

if PHASE == 'SMOKE':
    out = [run(a, SMOKE_STEPS, smoke=True) for a in ('B0', 'B3')]
    print(json.dumps([{k: v for k, v in r.items() if k != 'describe'} for r in out], indent=1))
else:
    r = run(PHASE, STEPS)
    print(json.dumps({k: v for k, v in r.items() if k != 'describe'}, indent=1))'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--steps", type=int, default=0)
    a = ap.parse_args()
    if a.phase != "SMOKE" and a.steps <= 0:
        raise SystemExit("un bras exige --steps S (fixé après la fumée)")
    e17 = json.loads(E17.read_text())["cells"]
    crops = [e17[i] for i in (5, 6, 7)]
    for c in crops:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    metric = METRIC.read_text().replace("from __future__ import annotations\n", "")
    arch = ARCH.read_text().replace("from __future__ import annotations\n", "")
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [md(f"# Babaorum lecteur — R6 banc d'architectures, phase {a.phase}\n\nPré-enregistrement : "
                     "docs/READER_R6_PREREG.txt (p9). Aucune ligne de test. La fumée ne mesure aucun CER."),
                  code(SETUP.format(phase=a.phase, steps=a.steps)),
                  code("%%bash\npip install -q rapidfuzz 2>&1 | tail -1"),
                  *crops,
                  code("# --- axel/reader/metric.py (D-150), recopié par le générateur ---\n" + metric),
                  code("import torch\n# --- babaorum/reader/r6_arch.py, recopié par le générateur ---\n" + arch),
                  code(TRAIN_CELL)],
    }
    out = HERE / "scripts/reader_notebooks" / f"kaggle_r6_{a.phase}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(out)


if __name__ == "__main__":
    main()
