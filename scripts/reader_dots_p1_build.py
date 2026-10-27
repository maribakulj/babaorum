"""P1 de docs/READER_DOTS_CTC_PLAN.txt — construit les entrées et les carnets Kaggle (CLASSE B ; rien n'est lancé).

    ~/axel/.venv/bin/python scripts/reader_dots_p1_build.py --inputs <dossier vide du scratchpad>

Entrées (fichiers à plat, sans archive, sha256 dans MANIFEST.json, contrôlés par le carnet) :
  items.jsonl         400 lignes TUNE (axel e8_items, split dev, tiges de reports/reader_r1e_tune.json) et
                      3 lignes TRAIN de photo_1983 pour la fumée (reports/reader_items_photo1983.jsonl, 3
                      premières ; crop_md5 calculé ici sur les mêmes pixels) ; champs : stem, part (tune/smoke),
                      image (« newseye/<nom> » ou « bnl/<nom> », archive publique), box, polygon, truth, crop_md5 ;
  a1.jsonl            treillis A1 de ces 403 lignes (logp float16) : TUNE depuis axel
                      runs/e17/kaggle/a1/dev_pred_a1.jsonl.gz (T4 fp16), fumée depuis reports/reader_a1_photo1983 ;
  timing.json         codec A1 (axel runs/e17/kaggle/a1) ;
  ctc_prefix.py, dots_guided.py (babaorum), crops.py, vlm.py, lattice.py (axel, recopiés tels quels).
Carnets : scripts/reader_notebooks/kaggle_dots_p1_smoke.ipynb (Kaggle CPU, VRAIS poids, 3 lignes train, libre
+ un guidage λ 0,5 PEN −10 B 4 : la chaîne et le temps, aucune qualité) et kaggle_dots_p1_tune.ipynb (GPU T4 ×2 :
400 lignes TUNE : libre glouton, faisceau 4 sans guide, guide λ {0,3 ; 0,5 ; 0,7} au faisceau 4 ; PEN = FLOOR fixe).
Après la fumée v2 (sorties guidées dégénérées) : faisceau corrigé (src/babaorum/reader/dots_beam.py :
l'expansion générique de transformers brouillait l'image de chaque faisceau) ; masque de VOCABULAIRE
(allowed_vocab) au lieu du masque de scripts ; la fumée ajoute le témoin « faisceau sans guide » et le TEST
SENTINELLE (score joint de la vérité contre les sorties dégénérées). Chargement de dots.mocr
recopié d'axel notebooks/kaggle_e3_e7.ipynb (D-182 : flash_attn factice, sdpa, bf16 seulement si capacité ≥ 8,
sinon float32 réparti sur 2 GPU, sdpa_forward 4D, rustine du processeur, max_pixels 11 289 600). Toile E3
(crop_line polygone pad 4 ; bnl/ : image entière ; line_canvas_size(0, 0)). Prompt E3. Longueur max de
génération = min(512, T_A1 + 16) jetons (la longueur de la vérité n'est PAS utilisée, contrairement à E3).
Sortie par ligne : stem, config, raw (sortie brute), s, jetons, ctc_prefixe, ctc_complet (texte nettoyé par
vlm.join_paragraphs puis clean_text ; null si −∞ ; après l'audit pT : ctc_prefixe_clean, ctc_complet_clean,
ctc_complet_raw, fin = eos / plafond, configuration guide_l0.0, PEN = FLOOR partout), ids (jetons générés), n_plancher (pas dont le Δ CTC
rejoué est ≤ FLOOR = −50 ou non fini). Réserves écrites : BnL = image entière sans marge (MD5 du lecteur ; E3
recadrait avec pad 4 : le témoin libre BnL n'est pas strictement celui d'E3) ; treillis des 3 lignes de fumée
lus sur le Mac (la fumée ne mesure aucune qualité).
Le choix du réglage (P2), le CER D-150 et la métrique en couches se font sur le Mac depuis ces sorties.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
AXEL = Path.home() / "axel"
R = HERE / "reports"
NB = HERE / "scripts/reader_notebooks"
DATASET = "babaorum-dots-p1-inputs"

SETUP = r'''import os, sys, json, gzip, glob, hashlib, shutil, subprocess, time, zipfile, math, types, importlib.machinery
PHASE = {phase!r}
D, TMP, OUT = '/kaggle/input', '/kaggle/tmp', '/kaggle/working'
os.makedirs(TMP, exist_ok=True)
def find(name):
    hits = sorted(glob.glob(f'{{D}}/**/{{name}}', recursive=True))
    assert len(hits) == 1, (name, hits)
    return hits[0]
IN = os.path.dirname(find('MANIFEST.json'))
MANIFEST_SHA = {manifest_sha!r}  # audit pT B3 : empreinte COMPLÈTE attendue, épinglée à la construction
assert hashlib.sha256(open(f'{{IN}}/MANIFEST.json', 'rb').read()).hexdigest() == MANIFEST_SHA, 'MANIFEST inattendu'
man = json.load(open(f'{{IN}}/MANIFEST.json'))
for f, h in man.items():
    assert hashlib.sha256(open(f'{{IN}}/{{f}}', 'rb').read()).hexdigest() == h, f
CODE = f'{{TMP}}/code'; os.makedirs(CODE, exist_ok=True)
for f in ('ctc_prefix.py', 'dots_guided.py', 'dots_beam.py', 'crops.py', 'vlm.py', 'lattice.py'):
    shutil.copy(f'{{IN}}/{{f}}', CODE)
# les modules babaorum importent « babaorum.reader.ctc_prefix » : paquet minimal
pkg = f'{{CODE}}/babaorum/reader'; os.makedirs(pkg, exist_ok=True)
open(f'{{CODE}}/babaorum/__init__.py', 'w').close(); open(f'{{pkg}}/__init__.py', 'w').close()
shutil.copy(f'{{IN}}/ctc_prefix.py', pkg); shutil.copy(f'{{IN}}/dots_guided.py', pkg)
sys.path.insert(0, CODE)
print('entrées conformes :', len(man), 'fichiers ; phase', PHASE)'''

IMAGES = r'''from PIL import Image
Image.MAX_IMAGE_PIXELS = None
items = [json.loads(l) for l in open(f'{IN}/items.jsonl')]
items = [it for it in items if it['part'] == ('tune' if PHASE == 'TUNE' else 'smoke')]
need = {}
for it in items:
    kind, name = it['image'].split('/', 1)
    need.setdefault(kind, set()).add(name)
NEWSEYE = 'https://zenodo.org/records/4293602/files/ATR_TrainingSet_BnF_Newseye_M2+.zip?download=1'
BNL = 'https://data.bnl.lu/wp-content/uploads/2021/07/bnl-ground-truth-newspapers-before-1878.zip'
def fetch(url, path):
    if not os.path.exists(path):
        subprocess.run(['wget', '-q', '-O', path, url], check=True)
def extract(zf, wanted, dest):
    os.makedirs(dest, exist_ok=True)
    for info in zf.infolist():
        base = os.path.basename(info.filename)
        if base in wanted and not os.path.exists(f'{dest}/{base}'):
            with zf.open(info) as s, open(f'{dest}/{base}', 'wb') as d:
                shutil.copyfileobj(s, d)
IMG = f'{TMP}/img'
if need.get('newseye'):
    fetch(NEWSEYE, f'{TMP}/newseye.zip')
    with zipfile.ZipFile(f'{TMP}/newseye.zip') as zf:
        extract(zf, need['newseye'], f'{IMG}/newseye')
    os.remove(f'{TMP}/newseye.zip')
if need.get('bnl'):
    fetch(BNL, f'{TMP}/bnl.zip')
    with zipfile.ZipFile(f'{TMP}/bnl.zip') as zf:
        inner = next(i for i in zf.infolist() if i.filename.endswith('/antiqua.zip'))
        with zf.open(inner) as s, open(f'{TMP}/antiqua.zip', 'wb') as d:
            shutil.copyfileobj(s, d)
    os.remove(f'{TMP}/bnl.zip')
    with zipfile.ZipFile(f'{TMP}/antiqua.zip') as zf:
        extract(zf, need['bnl'], f'{IMG}/bnl')
    os.remove(f'{TMP}/antiqua.zip')
from crops import crop_line
from vlm import line_canvas_size
CANVAS, mism = {}, []
by_img = {}
for it in items:
    by_img.setdefault(it['image'], []).append(it)
for img, group in by_img.items():
    with Image.open(f'{IMG}/{img}') as raw_img:
        page = raw_img.convert('L')
    for it in group:
        crop = page if img.startswith('bnl/') else crop_line(page, tuple(it['box']), it['polygon'], pad=4).image
        if hashlib.md5(crop.tobytes()).hexdigest() != it['crop_md5']:
            mism.append(it['stem'])
        cv = line_canvas_size(crop.width, crop.height, 0, 0)
        assert cv.scale == 1.0, (it['stem'], cv)
        canvas = Image.new('RGB', (cv.width, cv.height), (255, 255, 255))
        canvas.paste(crop.convert('RGB'), (cv.left, cv.top))
        CANVAS[it['stem']] = canvas
assert not mism, ('md5 différents', len(mism), mism[:5])
print(len(CANVAS), 'toiles, md5 conformes')'''

MODEL = r'''import torch, numpy as np
import transformers
assert transformers.__version__ == '4.51.3', transformers.__version__  # le pip doit précéder tout import
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
fa = types.ModuleType('flash_attn'); fa.__spec__ = importlib.machinery.ModuleSpec('flash_attn', None)
fa.flash_attn_varlen_func = None; sys.modules['flash_attn'] = fa
REPO, REV = 'dots-studio/dots.mocr', 'e539fbb52280393adc081b289ec597430a0f9031'
path = snapshot_download(REPO, revision=REV)
if PHASE == 'SMOKE':
    dtype, device_map = torch.float32, 'cpu'
else:
    CAP, NGPU = torch.cuda.get_device_capability(0), torch.cuda.device_count()
    if CAP[0] >= 8:
        dtype, device_map = torch.bfloat16, 'cuda:0'
    else:
        assert NGPU >= 2, 'float32 sur T4 : accélérateur « GPU T4 x2 »'
        dtype, device_map = torch.float32, {'model': 0, 'lm_head': 0, 'vision_tower': 1}
cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
cfg.vision_config.attn_implementation = 'sdpa'
model = AutoModelForCausalLM.from_pretrained(path, config=cfg, torch_dtype=dtype, attn_implementation='sdpa',
                                             trust_remote_code=True, device_map=device_map).eval()
from transformers import Qwen2_5_VLProcessor
_q_init = Qwen2_5_VLProcessor.__init__
def q_init(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kw):
    if chat_template is None and isinstance(video_processor, str):
        chat_template = video_processor
    _q_init(self, image_processor, tokenizer, chat_template=chat_template, **kw)
Qwen2_5_VLProcessor.__init__ = q_init
processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
processor.image_processor.max_pixels = 11_289_600
VT = type(model.vision_tower)
vis = sys.modules[VT.__module__]
_vt_forward = VT.forward
def vt_forward(self, hidden_states, grid_thw, bf16=True):
    # D-182 (3) : le code distant force l'entrée en bf16 ; on la ramène au type des poids
    return _vt_forward(self, hidden_states.to(self.dtype), grid_thw, bf16=False)
VT.forward = vt_forward
import torch.nn.functional as F
def sdpa_forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None):
    # D-182 (4), recopié tel quel d'axel notebooks/kaggle_e3_e7.ipynb
    n = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(n, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    q = vis.apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = vis.apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
    mask = None
    if len(cu_seqlens) > 2:
        mask = torch.zeros([1, 1, n, n], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            mask[..., cu_seqlens[i - 1]:cu_seqlens[i], cu_seqlens[i - 1]:cu_seqlens[i]] = True
    q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, mask, dropout_p=0.0)
    return self.proj(out[0].transpose(0, 1).reshape(n, -1))
vis.VisionSdpaAttention.forward = sdpa_forward
def bind(m, fn):  # accelerate garde l'ancien forward dans _old_forward (E3, version 3)
    if hasattr(m, '_old_forward'):
        m._old_forward = types.MethodType(fn, m)
    else:
        m.forward = types.MethodType(fn, m)
n_patched = 0
for m in model.modules():
    if isinstance(m, vis.VisionSdpaAttention):
        bind(m, sdpa_forward); n_patched += 1
    elif isinstance(m, VT):
        bind(m, vt_forward)
assert n_patched == cfg.vision_config.num_hidden_layers, n_patched
from dots_beam import patch_beam_expansion
patch_beam_expansion(model)  # fumée v2 : l'expansion générique brouillait l'image de chaque faisceau
tok = processor.tokenizer
print('modèle', dtype, device_map, 'torch', torch.__version__, 'transformers', transformers.__version__, sys.version.split()[0])'''

RUN = r'''from ctc_prefix import CTCPrefixScorer
from dots_guided import CTCGuide, FLOOR, allowed_vocab
from lattice import CODEC_FOLD
from vlm import clean_text, join_paragraphs
import unicodedata
CHARS = json.load(open(f'{IN}/timing.json'))['chars']
LABELS = ('',) + tuple(CHARS)
index = {c: i for i, c in enumerate(LABELS) if i}
_cache = {}
def enc(ch):  # recopié de reader_r1d_joint.encoder_for(labels, nfd=False)
    if ch in _cache:
        return _cache[ch]
    out = None
    if ch in index:
        out = [index[ch]]
    else:
        units = unicodedata.normalize('NFD', ch)
        if all(u in index for u in units):
            out = [index[u] for u in units]
        else:
            fold = CODEC_FOLD.get(ch)
            if fold and all(u in index for u in fold):
                out = [index[u] for u in fold]
    _cache[ch] = out
    return out
LAT = {}
for l in open(f'{IN}/a1.jsonl'):
    r = json.loads(l)
    LAT[r['stem']] = np.asarray(r['logp'], dtype=np.float64)
stems = [it['stem'] for it in items]
assert len(stems) == len(set(stems)) == (400 if PHASE == 'TUNE' else 3), (PHASE, len(stems), len(set(stems)))
assert set(stems) <= set(LAT) and len(LAT) == 403, 'couverture des treillis'
eos_ids = {tok.eos_token_id} | set(model.generation_config.eos_token_id or [])
dec = lambda ids: tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
# masque de VOCABULAIRE (pA) : permis = caractères complets tous dans le codec A1 (repli, espace compris) ou
# fragments d'octets qui composent un tel caractère ; tout le reste interdit (jetons '' et spéciaux compris)
ok_char = lambda c: c.isspace() or enc(c) is not None
ALLOWED = allowed_vocab(lambda i: dec([i]), len(tok), ok_char,
                        encode=lambda t: tok(t, add_special_tokens=False)['input_ids'])
forbidden = set(range(model.config.vocab_size)) - ALLOWED - eos_ids
print('jetons permis', len(ALLOWED), 'interdits', len(forbidden))
PROMPT_OCR = 'Extract the text content from this image.'
# (nom, λ, faisceau) ; λ None = sans guide. PEN = FLOOR fixe : avec le masque de vocabulaire, seul un caractère
# hors codec RECOMPOSÉ de fragments permis peut encore la payer (compté dans hors_codec).
# audit pT B1 : guide_l0.0 = MÊME dispositif (masque, top-K 10, PEN, fin au plafond) avec λ = 0 : le contraste
# guide λ* − guide_l0.0 isole le score CTC ; faisceau_libre reste le témoin du faisceau nu.
CONFIGS = [('libre', None, 1), ('faisceau_libre', None, 4)] + (
    [('guide_l0.5', 0.5, 4)] if PHASE == 'SMOKE' else [(f'guide_l{l}', l, 4) for l in (0.0, 0.3, 0.5, 0.7)])
dest = f'{OUT}/dots_p1_{PHASE.lower()}.jsonl'
done = set()
if os.path.exists(dest):  # reprise : lignes complètes seulement, fichier tronqué à la dernière ligne valide
    good = []
    for l in open(dest, encoding='utf-8'):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            break
        good.append(l if l.endswith('\n') else l + '\n')
        done.add((r['stem'], r['config']))
    open(dest, 'w', encoding='utf-8').writelines(good)
clean = ''
t_all = time.time()
with open(dest, 'a', encoding='utf-8') as fh:
    for k, it in enumerate(items):
        img = CANVAS[it['stem']]
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': PROMPT_OCR}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], padding=True, return_tensors='pt').to(model.device)
        p0 = inputs['input_ids'].shape[1]
        lp = LAT[it['stem']]
        max_new = min(512, lp.shape[0] + 16)  # borne tirée du treillis, pas de la vérité
        for name, lam, nb in CONFIGS:
            if (it['stem'], name) in done:
                continue
            scorer = CTCPrefixScorer(lp, enc)
            t0 = time.time()
            with torch.inference_mode():
                if lam is None:
                    kw = {'num_beams': nb, 'length_penalty': 0.0} if nb > 1 else {}  # glouton : comme E3
                    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, **kw)
                else:
                    guide = CTCGuide(scorer, dec, p0, eos_ids, lam=lam, top_k=10, pen=FLOOR, forbidden=forbidden,
                                     max_new=max_new)
                    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, num_beams=nb,
                                         length_penalty=0.0, logits_processor=[guide])
            gen = out[0, p0:].tolist()
            raw = dec(gen)
            clean = clean_text(join_paragraphs(raw))
            ref = CTCGuide(scorer, dec, p0, eos_ids, lam=1.0, top_k=1, pen=FLOOR)
            kept, bad, _ = ref.ctc_text(clean)
            pre, full = scorer.prefix(kept), scorer.full(kept)
            kept_r, bad_r, cut_r = ref.ctc_text(raw)
            full_r = scorer.full(kept_r)
            fin = 'eos' if gen and gen[-1] in eos_ids else ('plafond' if len(gen) >= max_new else 'autre')
            # pas au plancher (audit pT M1) : Δ CTC rejoué jeton par jeton sur le texte BRUT décodé, comme le guide ;
            # le dernier pas compte la fin (complet − préfixe), qu'elle vienne d'un eos ou du plafond
            n_pl, prev = 0, ref._score('')[0]
            for j in range(1, len(gen) + 1):
                cur = ref._score(dec(gen[:j]))[0]
                d = cur - prev
                if j == len(gen) and fin in ('eos', 'plafond'):
                    d = (full_r - prev) if gen[-1] in eos_ids else (d + full_r - cur)
                n_pl += (not math.isfinite(d)) or d <= FLOOR
                prev = cur
            fh.write(json.dumps({'stem': it['stem'], 'config': name, 'raw': raw, 'clean': clean,
                                 's': round(time.time() - t0, 3), 'tokens': len(gen), 'hors_codec': bad,
                                 'ids': gen, 'n_plancher': n_pl, 'fin': fin, 'max_new': max_new,
                                 'ctc_prefixe_clean': pre if math.isfinite(pre) else None,
                                 'ctc_complet_clean': full if math.isfinite(full) else None,
                                 'ctc_complet_raw': full_r if math.isfinite(full_r) else None,
                                 'hors_codec_raw': bad_r, 'utf8_coupe_raw': cut_r,
                                 'dtype': str(dtype)}, ensure_ascii=False) + '\n')
            fh.flush()
        if k % 20 == 0:
            print(k, round(time.time() - t_all), 's', repr(clean[:60]), flush=True)
print('fini', round(time.time() - t_all), 's')
json.dump({'phase': PHASE, 'manifest_sha256': MANIFEST_SHA, 'modele': REPO, 'revision': REV, 'dtype': str(dtype),
           'device_map': str(device_map), 'torch': torch.__version__, 'transformers': transformers.__version__,
           'python': sys.version.split()[0], 'lignes': len(items), 'configs': CONFIGS, 'jetons_permis': len(ALLOWED),
           'secondes': round(time.time() - t_all)}, open(f'{OUT}/dots_p1_{PHASE.lower()}_run.json', 'w'), indent=1)
if PHASE == 'SMOKE':
    # TEST SENTINELLE (pA, 4) : score joint, avec λ 0,5, de la vérité, de la sortie libre et des sorties dégénérées
    # de la fumée v2 ; log P_LLM par forçage (invite + texte + fin), log P_CTC complet. La vérité doit gagner.
    DEGEN = {'r1l5': '- 环 3\n\n  ①\n\n  ②\n\n  ③\n\n  ④\n\n  ⑤\n\n  ⑥\n\n  ⑦\n\n  ⑧\n\n  ⑨\n\n  ⑩\n\n  ⑪\n\n  ⑫\n\n  ⑬\n\n  ⑭\n\n  �', 'r1l7': 'P\n\nE\n\nT\n\nE\n\nE\n\nE\n\nE\n\nE\n\nE\n\nE\n\nE\n\nE\n\nN\n\nF\n\nO\n\nR\n\nN\n\nA\n\nT\n\nI\n\nO\n\nN\n\nE\n\n', 'r1l10': '# 智 趣 集 智\n\nA 集 智 趣 趣 集 智 集 智 趣 趣 集 智 趣 趣 集 智 趣 趣 集 智 趣 趣 集 智 �'}  # sorties brutes EXACTES de la fumée v2 (guide_l0.5_p-10, faisceau brouillé)
    libre = {json.loads(l)['stem']: json.loads(l)['raw'] for l in open(dest) if json.loads(l)['config'] == 'libre'}
    sent = []
    for it in items:
        img = CANVAS[it['stem']]
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': PROMPT_OCR}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], padding=True, return_tensors='pt').to(model.device)
        p0 = inputs['input_ids'].shape[1]
        scorer = CTCPrefixScorer(LAT[it['stem']], enc)
        ref = CTCGuide(scorer, dec, p0, eos_ids, lam=1.0, top_k=1, pen=FLOOR)
        for kind, cand in (('verite', it['truth']), ('libre', libre[it['stem']]),
                           ('degenere_v2', DEGEN[it['stem'].split('__')[-1]])):
            ids = tok(cand, add_special_tokens=False)['input_ids']
            full_ids = torch.cat([inputs['input_ids'], torch.tensor([ids + [tok.eos_token_id]], device=model.device)], dim=1)
            with torch.inference_mode():
                logits = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids),
                               pixel_values=inputs['pixel_values'], image_grid_thw=inputs['image_grid_thw']).logits
            lpm = torch.log_softmax(logits[0, p0 - 1:-1].float(), -1)  # len(ids) + 1 positions
            llm = float(lpm[torch.arange(len(ids)), torch.tensor(ids)].sum())
            llm += float(torch.logsumexp(lpm[len(ids), sorted(eos_ids)], 0))  # fin : l'un des jetons de fin
            kept, bad, _ = ref.ctc_text(cand)
            ctc = scorer.full(kept) + FLOOR * bad
            sent.append({'stem': it['stem'], 'cas': kind, 'texte': cand, 'llm': llm, 'ctc': ctc,
                         'joint_l0.5': 0.5 * llm + 0.5 * ctc})
            print(it['stem'].split('__')[-1], kind, round(llm, 2), round(ctc, 2), round(0.5 * llm + 0.5 * ctc, 2), flush=True)
    json.dump(sent, open(f'{OUT}/dots_p1_sentinelle.json', 'w'), ensure_ascii=False, indent=1)
    for st in {s_['stem'] for s_ in sent}:
        sc = {s_['cas']: s_['joint_l0.5'] for s_ in sent if s_['stem'] == st}
        print('SENTINELLE', st.split('__')[-1], 'vérité bat dégénéré :', sc['verite'] > sc['degenere_v2'],
              '| vérité bat libre :', sc['verite'] > sc['libre'])'''


def code(s: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def notebook(phase: str, manifest_sha: str) -> dict:
    head = (f"# babaorum — dots.mocr guidé par le CTC, P1 {phase}\n\n"
            + "docs/READER_DOTS_CTC_PLAN.txt. " + ("Kaggle CPU, vrais poids, 3 lignes TRAIN : la chaîne et le temps, "
                                                 "aucune qualité." if phase == "SMOKE" else
                                                 "GPU T4 ×2 (accélérateur « GPU T4 x2 » et Internet ON obligatoires), 400 lignes TUNE : "
                                                 "libre, faisceau sans guide, guide λ {0,3 ; 0,5 ; 0,7}, faisceau 4."))
    return {"cells": [md(head), code(SETUP.format(phase=phase, manifest_sha=manifest_sha)), code("%%bash\nset -e\npip install -q 'transformers==4.51.3' 'accelerate>=0.34' 2>&1 | tail -1"),
                      code(IMAGES), code(MODEL),
                      code(RUN)],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    a = ap.parse_args()
    out = Path(a.inputs)
    assert not out.exists() or not any(out.iterdir()), "dossier d'entrées non vide"
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(AXEL / "src"))
    from axel.image.crops import crop_line
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    tune = {r["stem"] for r in json.loads((R / "reader_r1e_tune.json").read_text())["rows"]}
    assert len(tune) == 400
    items = []
    for x in gzip.open(AXEL / "data/e8/kaggle/e8_items.jsonl.gz", "rt"):
        r = json.loads(x)
        if r["stem"] in tune:
            assert r["split"] == "dev"
            items.append({"stem": r["stem"], "part": "tune", "image": r["image"], "box": r["box"],
                          "polygon": r["polygon"], "truth": r["truth"], "crop_md5": r["crop_md5"]})
    assert len(items) == 400
    for x, _ in zip(open(R / "reader_items_photo1983.jsonl"), range(3)):
        r = json.loads(x)
        stem = f"{r['corpus']}__{r['page']}__{r['line_id']}"
        with Image.open(r["image"]) as im:
            crop = crop_line(im.convert("L"), tuple(r["box"]), r.get("polygon"), pad=4).image
        items.append({"stem": stem, "part": "smoke", "image": "newseye/" + Path(r["image"]).name, "box": r["box"],
                      "polygon": r.get("polygon"), "truth": r["truth"],
                      "crop_md5": hashlib.md5(crop.tobytes()).hexdigest()})
    (out / "items.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    want = {i["stem"] for i in items}
    got = 0
    with open(out / "a1.jsonl", "w") as g:  # NON compressé : Kaggle décompresse les .gz au dépôt (fumée v1)
        for src in (AXEL / "runs/e17/kaggle/a1/dev_pred_a1.jsonl.gz", R / "reader_a1_photo1983.jsonl.gz"):
            for x in gzip.open(src, "rt"):
                r = json.loads(x)
                if r.get("stem") in want:
                    g.write(json.dumps({"stem": r["stem"], "logp": r["logp"]}) + "\n")
                    got += 1
    assert got == 403, got
    shutil.copy(AXEL / "runs/e17/kaggle/a1/timing.json", out)
    for f in (HERE / "src/babaorum/reader/ctc_prefix.py", HERE / "src/babaorum/reader/dots_guided.py",
              HERE / "src/babaorum/reader/dots_beam.py",
              AXEL / "src/axel/image/crops.py", AXEL / "src/axel/image/vlm.py", AXEL / "src/axel/image/lattice.py"):
        shutil.copy(f, out)
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    manifest_sha = hashlib.sha256((out / "MANIFEST.json").read_bytes()).hexdigest()
    (out / "dataset-metadata.json").write_text(json.dumps({"title": DATASET, "id": f"mabaku/{DATASET}",
                                                           "licenses": [{"name": "other"}]}))
    for phase in ("SMOKE", "TUNE"):
        (NB / f"kaggle_dots_p1_{phase.lower()}.ipynb").write_text(json.dumps(notebook(phase, manifest_sha), indent=1,
                                                                             ensure_ascii=False))
    print(json.dumps({"items": len(items), "a1": got, "manifest": manifest}, indent=1))


if __name__ == "__main__":
    main()
