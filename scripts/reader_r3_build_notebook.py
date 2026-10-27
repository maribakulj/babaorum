"""Génère reader/notebooks/colab_r3_ancre.ipynb (Colab, T4 ou mieux).

R3 : encodeur dots.mocr gelé coupé à L = 32 (comme E17 A1) ; tête CTC
initialisée depuis head_a1.pt + décodeur d'attention léger (lecteur/ancre.py),
entraînement multitâche sur caractéristiques en cache ; décodage joint ancré
réglé sur la moitié « tune » du dev (pages tirées, graine 1, comme R1),
mesuré sur la moitié « eval » ; R4 : invention sur images perturbées.
Les cellules d'encodeur et de recadrage sont reprises telles quelles du
carnet axel kaggle_e17_vitctc.ipynb (lecture seule). Aucun test lu.

    ~/axel/.venv/bin/python reader/scripts/r3_build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SRC = Path.home() / "axel/notebooks/kaggle_e17_vitctc.ipynb"
OUT = HERE / "scripts" / "reader_notebooks" / "colab_r3_ancre.ipynb"


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


HEAD = md("""# Babaorum lecteur — R3 « Ancre » : encodeur dots gelé + CTC + décodeur, décodage joint ancré

Plan : `docs/READER_PLAN.txt` (onglet lecteur, Claude w2:p9). **Lancer seulement avec l'accord de Marcel.**
Phases : F (caractéristiques L = 32 en cache local), T (entraînement multitâche, tête CTC partie de A1),
E (dev : réglage sur la moitié tune, mesure sur la moitié eval), I (invention sur images perturbées).
Durée visée ≈ 3 h sur T4. Sorties : `MyDrive/babaorum/reader_r3/`. Aucune ligne de test n'est lue.""")

SETUP = code("""import os, sys, json, gzip, glob, hashlib, shutil, subprocess, time, zipfile, math, random
KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
if KAGGLE:   # entrées = jeux Kaggle (axel-e8-presse, sortie de axel-e17-vitctc), aucun consentement Drive
    D, OUT, TMP = '/kaggle/input', '/kaggle/working/reader_r3', '/kaggle/tmp'
else:
    from google.colab import drive
    drive.mount('/content/drive')
    D = '/content/drive/MyDrive/babaorum/in'
    OUT = '/content/drive/MyDrive/babaorum/reader_r3'
    TMP = '/content/tmp'
os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
T_START = time.time()
PHASES = ('F', 'T', 'E', 'I')
L = 32
REPO, REV = 'dots-studio/dots.mocr', 'e539fbb52280393adc081b289ec597430a0f9031'   # révision d'E3/E17
BUDGET_H = 3.5            # plus de nouvelle époque au-delà ; E et I se lancent quand même
EPOCHS, LAMBDA_CTC = 12, 0.3
SMOKE = False             # True : essai court (compte gratuit, < 1 h) : 2 000 lignes train, 300 dev, 1 époque
if SMOKE:
    EPOCHS, BUDGET_H = 1, 0.8
    OUT = OUT + '_smoke'; os.makedirs(OUT, exist_ok=True)
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)""")

ANCRE_CELL_TMPL = "# --- reader/lecteur/ancre.py, recopié par le générateur : ne pas modifier ici ---\n{src}"

FEATS = code("""# F — caractéristiques L = 32, une fois, sur le disque local (fp16, ~18 Go)
FEAT = f'{TMP}/feats'; os.makedirs(FEAT, exist_ok=True)
INDEX = {}
ITEMS_DEV = list(DEV)
if SMOKE:
    TRAIN = random.Random(0).sample(TRAIN, 2000); DEV = random.Random(0).sample(DEV, 300)
if 'F' in PHASES and not os.path.exists(f'{FEAT}/index.json'):
    enc = load_encoder(L)
    t0 = time.time(); k = 0
    for split, its in (('train', TRAIN), ('dev', DEV)):
        for b in bucketed(its, 32, shuffle=False):
            with torch.no_grad():
                fs = feats_at(enc, b, L)
            torch.save([store(f) for f in fs], f'{FEAT}/{k}.pt')
            for j, it in enumerate(b):
                INDEX[it['stem']] = (k, j)
            k += 1
            if k % 200 == 0:
                print(split, k, round(time.time() - t0), 's', flush=True)
    json.dump(INDEX, open(f'{FEAT}/index.json', 'w'))
    del enc; torch.cuda.empty_cache()
    print('caractéristiques', len(INDEX), round(time.time() - t0), 's')
INDEX = json.load(open(f'{FEAT}/index.json'))
SHARDS = {}
for stem, (k, j) in INDEX.items():
    SHARDS.setdefault(k, []).append(stem)
STEM2IT = {it['stem']: it for it in TRAIN + DEV}
DEV_STEMS = {it['stem'] for it in DEV}
# Partition tune/eval par page (graine 1, sha256 eval 996ab0cd… dans READER_R3_PREREG) :
# fixée AVANT l'apprentissage ; le choix d'époque ne voit que tune (revue pM, 15/10).
def page_of(stem):
    return stem.split('__')[1] if '__' in stem else stem
ALL_DEV = sorted(it['stem'] for it in ITEMS_DEV)
pages = sorted({page_of(s) for s in ALL_DEV}); random.Random(1).shuffle(pages)
TUNE_P = set(pages[: len(pages) // 2])
TUNE_STEMS = {s for s in DEV_STEMS if page_of(s) in TUNE_P}
print('dev', len(DEV_STEMS), 'dont tune', len(TUNE_STEMS))""")

TRAIN_CELL = code("""# T — Ancre : tête CTC partie de head_a1.pt, décodeur neuf ; perte λ·CTC + (1−λ)·attention
_c = sorted(glob.glob(f'{D}/**/chars.json', recursive=True))
_tj = [p for p in sorted(glob.glob(f'{D}/**/timing.json', recursive=True)) if 'chars' in json.load(open(p))]
if _c:
    assert len(_c) == 1, f'plusieurs chars.json : {_c}'
    A1_CHARS = json.load(open(_c[0]))
else:
    assert len(_tj) == 1, f'il faut exactement un timing.json avec chars : {_tj}'
    A1_CHARS = json.load(open(_tj[0]))['chars']
assert len(A1_CHARS) == 141, len(A1_CHARS)
_h = sorted(glob.glob(f'{D}/**/head_a1.pt', recursive=True))
assert len(_h) == 1 and _h[0].endswith('head_a1.pt'), f'il faut exactement un head_a1.pt : {_h}'
sd = torch.load(_h[0], map_location='cpu', weights_only=False)
assert isinstance(sd, dict) and 'out.weight' in sd, 'head_a1.pt doit être le state_dict de la tête A1'
print('tête A1 :', _h[0], '| codec :', _c[0] if _c else _tj[0])
assert A1_CHARS == chars, "codec différent de celui d'A1 (ordre compris)"
model = Ancre(len(chars) + 1).to(DEV_T)
print(model.ctc.load_state_dict(sd, strict=True))
opt = torch.optim.AdamW([{'params': model.ctc.parameters(), 'lr': 2e-4},
                         {'params': model.dec.parameters(), 'lr': 1e-3}], weight_decay=0.01)

def batches(stems_by_shard, train):
    ks = list(stems_by_shard)
    if train:
        random.shuffle(ks)
    for k in ks:
        fs = torch.load(f'{FEAT}/{k}.pt')
        stems = SHARDS[k]
        keep = [(f, s) for f, s in zip(fs, stems) if s in stems_by_shard[k]]
        if keep:
            yield keep

def collate(keep):
    fs = [f.float() for f, _ in keep]
    n = torch.tensor([f.shape[0] for f in fs])
    x = torch.zeros(len(fs), int(n.max()), fs[0].shape[1])
    for i, f in enumerate(fs):
        x[i, : f.shape[0]] = f
    ys = [[IDX[c] for c in clean(STEM2IT[s]['truth']) if c in IDX] for _, s in keep]
    return x.to(DEV_T), n.to(DEV_T), ys, [s for _, s in keep]

TR_SH = {k: set(v) - DEV_STEMS for k, v in SHARDS.items()}
TR_SH = {k: v for k, v in TR_SH.items() if v}
DV_SH = {k: set(v) & DEV_STEMS for k, v in SHARDS.items()}
DV_SH = {k: v for k, v in DV_SH.items() if v}
TUNE_SH = {k: v & TUNE_STEMS for k, v in DV_SH.items() if v & TUNE_STEMS}

@torch.no_grad()
def greedy_both(x, n):
    # CTC glouton et attention gloutonne (en lot), même modèle
    model.eval()
    logp, mem = model.ctc(x, n)
    tl = n * model.ctc.up
    pad = torch.arange(logp.shape[1], device=x.device)[None] >= tl[:, None]
    B = x.shape[0]; prev = torch.zeros(B, 1, dtype=torch.long, device=x.device)
    done = torch.zeros(B, dtype=torch.bool, device=x.device)
    for _ in range(int(tl.max()) // 2 + 2):
        nxt = model.dec(prev, mem, pad)[:, -1].argmax(-1)
        nxt = torch.where(done, torch.zeros_like(nxt), nxt)
        prev = torch.cat([prev, nxt[:, None]], 1); done |= nxt == 0
        if done.all():
            break
    model.train()
    att = []
    for row in prev[:, 1:].tolist():
        s = []
        for c in row:
            if c == 0:
                break
            s.append(chars[c - 1])
        att.append(''.join(s))
    ctc = [greedy(lp[: int(t)].float().cpu(), chars) for lp, t in zip(logp, tl)]
    return ctc, att, logp, tl

def eval_dev():
    # choix d'époque : moitié TUNE seulement (aucune ligne eval)
    pc, pa = [], []
    for keep in batches(TUNE_SH, False):
        x, n, ys, stems = collate(keep)
        c, a, _, _ = greedy_both(x, n)
        for s, hc, ha in zip(stems, c, a):
            t = STEM2IT[s]['truth']; pc.append((hc, t)); pa.append((ha, t))
    return score(pc), score(pa)

LOG = open(f'{OUT}/r3_train_log.jsonl', 'a')
if 'T' in PHASES:
    sc0 = eval_dev()
    print('départ (A1 + décodeur neuf) : CTC', sc0[0]['cer_d150'], 'attention', sc0[1]['cer_d150'], flush=True)
    best = 1e9
    for ep in range(1, EPOCHS + 1):
        if time.time() - T_START > BUDGET_H * 3600:
            break
        t0 = time.time(); ls = []
        for keep in batches(TR_SH, True):
            x, n, ys, _ = collate(keep)
            ok = [i for i, y in enumerate(ys) if y]
            if not ok:
                continue
            x, n, ys = x[ok], n[ok], [ys[i] for i in ok]
            with torch.autocast('cuda', dtype=DT):
                lo = model.losses(x, n.cpu(), ys)
            loss = LAMBDA_CTC * lo['ctc'] + (1 - LAMBDA_CTC) * lo['att']
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            ls.append((lo['ctc'].item(), lo['att'].item()))
        sc = eval_dev()
        row = {'epoch': ep, 'selection': 'tune', 'ctc_loss': float(np.mean([a for a, _ in ls])), 'att_loss': float(np.mean([b for _, b in ls])),
               'dev_ctc_d150': sc[0]['cer_d150'], 'dev_att_d150': sc[1]['cer_d150'], 'epoch_s': time.time() - t0}
        LOG.write(json.dumps(row) + '\\n'); LOG.flush(); print(row, flush=True)
        crit = min(sc[0]['cer_d150'], sc[1]['cer_d150'])
        if crit < best - 0.01:
            best = crit; torch.save(model.state_dict(), f'{OUT}/r3_best.pt.tmp'); os.replace(f'{OUT}/r3_best.pt.tmp', f'{OUT}/r3_best.pt')
model.load_state_dict(torch.load(f'{OUT}/r3_best.pt', map_location=DEV_T))""")

EVAL_CELL = code("""# E — décodage joint ancré : réglage (α, couloir) sur 300 lignes de la moitié tune, mesure sur la moitié eval
tune = sorted(TUNE_STEMS)
evl = sorted(s for s in DEV_STEMS if page_of(s) not in TUNE_P)
if SMOKE:
    # fumée : aucune ligne de la moitié eval n'est décodée ni imprimée (revue pM, 15/10)
    evl = tune[: min(50, len(tune))]
FEATS_OF = {}
for keep in batches(DV_SH, False):
    for f, s in keep:
        FEATS_OF[s] = f

def ids2txt(ids):
    return ''.join(chars[c - 1] for c in ids)

def run_joint(stems, alpha, corridor, beam=8, topk=8):
    out = {}
    for s in stems:
        f = FEATS_OF[s].float()[None].to(DEV_T)
        hs = joint_decode(model, f, f.shape[1], alpha=alpha, beam=beam, topk=topk, corridor=corridor)
        out[s] = ids2txt(hs[0].ids) if hs else ''
    return out

RES = {'tune_lines': len(tune), 'eval_lines': len(evl)}
if 'E' in PHASES:
    sub = random.Random(3).sample(tune, min(30 if SMOKE else 300, len(tune)))
    grid = {}
    for alpha in (0.3, 0.5, 0.7, 1.0):
        for corridor in (10.0, 30.0):
            t0 = time.time()
            o = run_joint(sub, alpha, corridor)
            grid[f'{alpha}/{corridor}'] = score([(o[s], STEM2IT[s]['truth']) for s in sub])['cer_d150']
            print('tune', alpha, corridor, grid[f'{alpha}/{corridor}'], round(time.time() - t0), 's', flush=True)
    RES['grid_tune300'] = grid
    # égalité : plus grand α (plus de poids au CTC), puis plus petit couloir (règle fixée avant mesure)
    a, c = map(float, min(grid, key=lambda k: (grid[k], -float(k.split('/')[0]), float(k.split('/')[1]))).split('/'))
    RES['chosen'] = {'alpha': a, 'corridor': c}
    oj = run_joint(evl, a, c)
    pc, pa, rows = [], [], []
    for keep in batches({k: v & set(evl) for k, v in DV_SH.items() if v & set(evl)}, False):
        x, n, ys, stems = collate(keep)
        cg, ag, logp, tl = greedy_both(x, n)
        for i, s in enumerate(stems):
            t = STEM2IT[s]['truth']; pc.append((cg[i], t)); pa.append((ag[i], t))
            lp = logp[i, : int(tl[i])].float().cpu().numpy()
            rows.append({'stem': s, 'truth': t, 'ctc': cg[i], 'att': ag[i], 'joint': oj[s],
                         'margin': readability_margin(lp), 'logp': lp.astype(np.float16).tolist()})
    RES['eval_ctc_greedy'] = score(pc); RES['eval_att_greedy'] = score(pa)
    RES['eval_joint'] = score([(oj[s], STEM2IT[s]['truth']) for s in evl])
    with gzip.open(f'{OUT}/r3_dev_eval.jsonl.gz', 'wt') as g:
        for r in rows:
            g.write(json.dumps(r, ensure_ascii=False) + '\\n')
    json.dump(RES, open(f'{OUT}/r3_results.json', 'w'), indent=1)
    print({k: (v['cer_d150'] if isinstance(v, dict) and 'cer_d150' in v else v) for k, v in RES.items()})""")

INVENT_CELL = code("""# I — invention : 200 lignes de la moitié eval × {blanc, brouillé 28 px, flou fort, bruit fort}
from PIL import ImageFilter
def perturb(img, mode, key):
    seed = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)
    if mode == 'blank':
        return Image.new('L', img.size, 255)
    if mode == 'blur_strong':
        return img.filter(ImageFilter.GaussianBlur(max(2.0, img.height / 12)))
    arr = np.asarray(img).copy()
    if mode == 'noise_strong':
        rng = np.random.default_rng(seed)
        return Image.fromarray(np.clip(arr.astype(np.float32) + rng.normal(0, 70, arr.shape), 0, 255).astype(np.uint8))
    h, w = arr.shape[0] // 28, arr.shape[1] // 28
    tiles = [arr[r*28:(r+1)*28, c*28:(c+1)*28].copy() for r in range(h) for c in range(w)]
    random.Random(seed).shuffle(tiles)
    for k, t in enumerate(tiles):
        r, c = divmod(k, w); arr[r*28:(r+1)*28, c*28:(c+1)*28] = t
    return Image.fromarray(arr)

def words3(text):
    ws = [w for w in re.findall(r"[^\\W\\d_]{2,}", text)]
    return len(ws) >= 3

import re
if 'I' in PHASES:
    enc = load_encoder(L)
    picks = random.Random(4).sample(evl, min(10 if SMOKE else 200, len(evl)))
    a, c = RES.get('chosen', {'alpha': 0.5, 'corridor': 10.0}).values()
    inv = {}
    for mode in ('blank', 'scramble', 'blur_strong', 'noise_strong'):
        rows = []
        for k in range(0, len(picks), 16):
            b = [STEM2IT[s] for s in picks[k:k+16]]
            ims = [prep(perturb(CROPS[it['stem']], mode, it['stem'] + '#r4'), HEIGHT, MAXW) for it in b]
            with torch.no_grad():
                fs = encode_lines(enc, ims, L, WHITE, DEV_T, amp)
            x, n, _, stems = collate([(store(f), it['stem']) for f, it in zip(fs, b)])
            cg, ag, logp, tl = greedy_both(x, n)
            for i, it in enumerate(b):
                f = store(fs[i]).float()[None].to(DEV_T)
                hs = joint_decode(model, f, f.shape[1], alpha=a, corridor=c)
                jt = ids2txt(hs[0].ids) if hs else ''
                lp = logp[i, : int(tl[i])].float().cpu().numpy()
                rows.append({'stem': it['stem'], 'ctc': cg[i], 'att': ag[i], 'joint': jt, 'margin': readability_margin(lp)})
        inv[mode] = {arm: sum(words3(r[arm]) for r in rows) for arm in ('ctc', 'att', 'joint')}
        inv[mode]['lines'] = len(rows)
        inv[mode]['rows'] = rows
        print(mode, {k: v for k, v in inv[mode].items() if k != 'rows'}, flush=True)
    RES['invention'] = {m: {k: v for k, v in d.items() if k != 'rows'} for m, d in inv.items()}
    json.dump(inv, open(f'{OUT}/r3_invention.json', 'w'), ensure_ascii=False)
    json.dump(RES, open(f'{OUT}/r3_results.json', 'w'), indent=1)
RES['hours'] = (time.time() - T_START) / 3600
print(json.dumps({k: v for k, v in RES.items() if k != 'grid_tune300'}, indent=1, ensure_ascii=False))""")


def main() -> None:
    src = json.loads(SRC.read_text())
    cells = src["cells"]
    keep = [cells[i] for i in (2, 3, 4, 5, 6, 7, 8, 9, 10)]
    for c in keep:
        c["outputs"] = [] if c["cell_type"] == "code" else c.get("outputs", [])
        if c["cell_type"] == "code":
            c["execution_count"] = None
    ancre = (HERE / "src" / "babaorum" / "reader" / "ancre.py").read_text()
    ancre = ancre.replace("from __future__ import annotations\n", "")
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [HEAD, SETUP] + keep + [code(ANCRE_CELL_TMPL.format(src=ancre)), FEATS, TRAIN_CELL,
                                         EVAL_CELL, INVENT_CELL],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(OUT, len(nb["cells"]), "cellules")


if __name__ == "__main__":
    main()
