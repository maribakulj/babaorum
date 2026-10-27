"""P0 — fumée CPU du guidage CTC de dots.mocr (CLASSE A ; docs/READER_DOTS_CTC_PLAN.txt). AUCUNE mesure de
qualité : le décodeur Qwen est TRONQUÉ à 2 blocs (config.num_hidden_layers = 2), il écrit n'importe quoi ;
on vérifie seulement que la chaîne tourne (chargement, toile E3, generate libre et guidé avec CTCGuide,
formes, temps) et que le score CTC du texte guidé est cohérent (préfixe ≥ complet).
Lignes : 3 lignes de photo_1983 (TRAIN), les 3 premières de reports/reader_items_photo1983.jsonl ; treillis
A1 de ces lignes : reports/reader_a1_photo1983.jsonl.gz (A1 sur Mac). Toile : celle d'E3 (axel
scripts/e3_dots.py, crop_canvas : crop_line polygone pad 4, line_canvas_size(min_height 0, margin 0),
toile blanche RGB). Modèle : ~/outils-seg/modeles/DotsMOCR, float32, CPU, attention sdpa, flash_attn factice
(D-182). Prompt E3. Faisceau 4, λ 0,5, K 10, longueur max 24 jetons, length_penalty 0.
Deux modes (règle pA du 18/10 : dots.mocr ne se charge JAMAIS en entier sur le Mac, le swap remplit le disque) :
  --mini : SUR LE MAC. Modèle ALÉATOIRE de même architecture et même vocabulaire, dimensions réduites (largeur 64,
           1 bloc de vision, 2 blocs de décodeur ; ≈ 20 M paramètres, < 100 Mo) : vérifie processeur, code distant,
           generate en faisceau et CTCGuide, sans aucun poids réel.
  défaut : VRAIS poids, décodeur tronqué à 2 blocs ; REFUSÉ sur macOS (Kaggle CPU seulement).
Masque : non_latin_ids (scripts non latins), PAS le masque de vocabulaire allowed_vocab des carnets Kaggle.
Environnement : ~/outils-seg/lourd.sh env PYTHONPATH=~/axel/src:src ~/outils-seg/dots-hf-env/bin/python \\
    scripts/reader_dots_guided_fumee.py --out <scratchpad>/dots_fumee.json
"""

from __future__ import annotations

import argparse
import gzip
import importlib.machinery
import json
import sys
import time
import types
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.home() / "axel/src"))
sys.path.insert(0, str(HERE / "src"))
from axel.image.crops import crop_line  # noqa: E402
from axel.image.lattice import CODEC_FOLD  # noqa: E402
from axel.image.vlm import line_canvas_size  # noqa: E402
from babaorum.reader.ctc_prefix import CTCPrefixScorer  # noqa: E402
from babaorum.reader.dots_guided import CTCGuide, non_latin_ids  # noqa: E402

DOTS = Path.home() / "outils-seg/modeles/DotsMOCR"
TIMING = Path.home() / "axel/runs/e17/kaggle/a1/timing.json"
R = HERE / "reports"
PROMPT_OCR = "Extract the text content from this image."


def encoder_for(labels: tuple[str, ...], nfd: bool):
    """Recopié tel quel de scripts/reader_r1d_joint.py."""
    index = {c: i for i, c in enumerate(labels) if i}
    cache: dict[str, list[int] | None] = {}

    def enc(ch: str) -> list[int] | None:
        if ch in cache:
            return cache[ch]
        out: list[int] | None = None
        if not nfd and ch in index:
            out = [index[ch]]
        else:
            units = unicodedata.normalize("NFD", ch)
            if all(u in index for u in units):
                out = [index[u] for u in units]
            else:
                fold = CODEC_FOLD.get(ch)
                if fold and all(u in index for u in fold):
                    out = [index[u] for u in fold]
        cache[ch] = out
        return out

    return enc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mini", action="store_true")
    a = ap.parse_args()
    assert a.mini or sys.platform != "darwin", "vrais poids de dots.mocr interdits sur le Mac (règle pA) : --mini"
    assert not Path(a.out).resolve().is_relative_to(R), "fumée : sortie hors de reports/"
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    import torch
    import transformers  # noqa: F401  (avant le module factice, comme D-182)

    fa = types.ModuleType("flash_attn")
    fa.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    fa.flash_attn_varlen_func = None
    sys.modules["flash_attn"] = fa
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, Qwen2_5_VLProcessor

    _q_init = Qwen2_5_VLProcessor.__init__

    def q_init(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kw):
        if chat_template is None and isinstance(video_processor, str):
            chat_template = video_processor
        _q_init(self, image_processor, tokenizer, chat_template=chat_template, **kw)

    Qwen2_5_VLProcessor.__init__ = q_init
    cfg = AutoConfig.from_pretrained(DOTS, trust_remote_code=True)
    cfg.vision_config.attn_implementation = "sdpa"
    cfg.num_hidden_layers = 2  # FUMÉE : décodeur tronqué
    t0 = time.time()
    if a.mini:  # poids aléatoires, dimensions réduites (même vocabulaire, même code)
        cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.num_key_value_heads = 64, 128, 2, 1
        v = cfg.vision_config
        v.embed_dim, v.hidden_size, v.intermediate_size, v.num_hidden_layers, v.num_attention_heads = 64, 64, 128, 1, 2
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32, attn_implementation="sdpa",
                                                 trust_remote_code=True).eval()
        print("paramètres", sum(p.numel() for p in model.parameters()), flush=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(DOTS, config=cfg, torch_dtype=torch.float32,
                                                     attn_implementation="sdpa", trust_remote_code=True).eval()
    vt = type(model.vision_tower)
    _vt_forward = vt.forward
    vt.forward = lambda self, hs, grid_thw, bf16=True: _vt_forward(self, hs.to(self.dtype), grid_thw, bf16=False)
    processor = AutoProcessor.from_pretrained(DOTS, trust_remote_code=True)
    processor.image_processor.max_pixels = 11_289_600  # comme E3
    tok = processor.tokenizer
    load_s = time.time() - t0

    items = [json.loads(x) for x, _ in zip(open(R / "reader_items_photo1983.jsonl"), range(3))]
    stem = lambda it: f"{it['corpus']}__{it['page']}__{it['line_id']}"  # noqa: E731
    lats = {}
    for x in gzip.open(R / "reader_a1_photo1983.jsonl.gz", "rt"):
        r = json.loads(x)
        if r.get("stem") in {stem(it) for it in items}:
            lats[r["stem"]] = r["logp"]
    enc = encoder_for(("",) + tuple(json.load(open(TIMING))["chars"]), nfd=False)
    eos_ids = {tok.eos_token_id} | set(getattr(model.generation_config, "eos_token_id", None) or [])
    # jetons spéciaux non-fin et ids hors du tokenizer : se décodent en '' ; interdits (relecture P0, point 2)
    forbidden = (set(tok.all_special_ids) | set(range(len(tok), model.config.vocab_size))) - eos_ids
    t_mask = time.time()
    forbidden |= non_latin_ids(lambda i: tok.decode([i], skip_special_tokens=False), len(tok)) - eos_ids  # masque pA
    print("masque", len(forbidden), "ids interdits,", round(time.time() - t_mask, 1), "s", flush=True)
    rows = []
    for it in items:
        with Image.open(it["image"]) as im:
            page = im.convert("L")
        crop = crop_line(page, tuple(it["box"]), it.get("polygon"), pad=4).image
        cv = line_canvas_size(crop.width, crop.height, 0, 0)
        canvas = Image.new("RGB", (cv.width, cv.height), (255, 255, 255))
        canvas.paste(crop.convert("RGB"), (cv.left, cv.top))
        msgs = [{"role": "user", "content": [{"type": "image", "image": canvas}, {"type": "text", "text": PROMPT_OCR}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[canvas], padding=True, return_tensors="pt")
        p0 = inputs["input_ids"].shape[1]
        dec = lambda ids: tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)  # noqa: E731
        scorer = CTCPrefixScorer(__import__("numpy").asarray(lats[stem(it)], dtype="float64"), enc)
        guide = CTCGuide(scorer, dec, p0, eos_ids, lam=0.5, top_k=10, forbidden=forbidden)
        row = {"stem": stem(it), "truth": it["truth"], "canvas": [cv.width, cv.height], "prompt_tokens": p0,
               "T_A1": len(lats[stem(it)])}
        with torch.inference_mode():
            t1 = time.time()
            free = model.generate(**inputs, max_new_tokens=24, do_sample=False)
            row["libre_s"] = round(time.time() - t1, 1)
            t1 = time.time()
            guided = model.generate(**inputs, max_new_tokens=24, do_sample=False, num_beams=4, length_penalty=0.0,
                                    logits_processor=[guide])
            row["guide_s"] = round(time.time() - t1, 1)
        row["libre"] = dec(free[0, p0:].tolist())
        row["guide"] = dec(guided[0, p0:].tolist())
        kept, bad, _ = guide.ctc_text(row["guide"])
        row["guide_ctc"] = {"prefixe": scorer.prefix(kept), "complet": scorer.full(kept), "hors_codec": bad}
        assert row["guide_ctc"]["prefixe"] >= row["guide_ctc"]["complet"] - 1e-9
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps({"fumee": ("mini aléatoire" if a.mini else "vrais poids, décodeur tronqué à 2 blocs")
                                       + ", aucune qualité", "chargement_s": round(load_s, 1),
                                       "lignes": rows}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
