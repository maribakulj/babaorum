"""R7(a) — exclusions des zones vides (fin de S2), AVANT toute lecture (docs/READER_R7A_PREREG.txt, « EXCLUSIONS » ;
CLASSE B ; écrit SANS DONNÉES, accord pA du 22/10 ; rien ne passe sur des pages avant la livraison de G0 « suivi »).

    ~/outils-seg/dots-hf-env/bin/python scripts/reader_r7a_exclure.py --build <sortie de reader_r7a_build.py> \\
        --build-manifest-sha256 <SHA de son MANIFEST.json> --decisions <decisions.json> --sortie <dossier vide>

Entrées :
  --build : sortie FIGÉE du constructeur (items.jsonl, contact.json, zones/<n>.png, contact.png, MANIFEST.json…) ; le
    MANIFEST est épinglé par --build-manifest-sha256 et CHAQUE fichier de sortie qu'il liste est rehaché ; un fichier
    en plus ou en moins arrête.
  --decisions : decisions.json, écrit par p9 après avoir regardé la planche contact et les zones pleine résolution :
    {"build_manifest_sha256": <le même SHA>,
     "regardees": [<toutes les tiges de zone de contact.json>],
     "exclusions": [{"stem": <tige de zone>, "motif": "T" | "O" | "I" | "B", "note": <texte>}, …],
     "difficiles": [{"stem": <tige de zone>, "nature": "tache" | "transparence" | "trou", "note": <texte>}, …]}
Liste FERMÉE des motifs (PREREG) : T texte imprimé visible (oubli de l'ALTO, texte pâle compris) ; O ornement ou filet ;
  I illustration ou image ; B hors du papier imprimé (bord du scan, règle, mire, fond de numérisation). Une tache, une
  transparence (texte du verso) ou un trou ne sont PAS des motifs : la zone RESTE ; elle peut être notée « difficile »
  (rapportée, jamais exclue). Tout autre motif, toute tige inconnue ou qui n'est pas une zone, tout doublon, toute zone
  à la fois exclue et difficile, et toute zone non regardée arrêtent.
Sorties, dans cet ordre : items_retenus.jsonl (lignes de texte inchangées + zones gardées) ; exclusions.json (liste,
  comptes par motif × famille × type de page, difficiles, zones retenues par page, pages avec au moins une zone,
  plancher 200 : effectif seulement, sans arrêt) ; planche_exclusions.png (planche de CONTRÔLE : chaque zone exclue,
  groupée par motif, avec sa tige et son motif, pour une seconde lecture des exclusions par le relecteur) ;
  MANIFEST.json EN DERNIER (sorties + SHA des entrées : MANIFEST du constructeur, decisions.json, ce script, PREREG).
Contrôles ajoutés après relecture Sonnet (22/10) : JSON sans clé dupliquée ; chaînes exigées ; note non vide pour une
  exclusion ; PREREG et constructeur courants identiques à ceux du MANIFEST du constructeur ; items.jsonl et contact.json
  lus une fois, hachés puis analysés ; types de page et familles fermés.
Ordre avec la lecture : S3/S4 ne liront QUE items_retenus.jsonl, et exigeront le SHA de ce MANIFEST, épinglé avant
  toute lecture (aucune exclusion après la lecture).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
MOTIFS = {"T": "texte imprimé visible", "O": "ornement ou filet", "I": "illustration ou image", "B": "hors du papier imprimé"}
DIFFICILES = ("tache", "transparence", "trou")
TYPES, FAMILLES = ("presse", "livre"), ("F1", "F2")
PLANCHER = 200


class ExclusionError(Exception):
    pass


def need(cond, msg) -> None:
    """Contrôle qui survit à python -O."""
    if not cond:
        raise ExclusionError(msg)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def strict_json(raw: bytes):
    """JSON UTF-8 sans clé dupliquée (mineur 9 Sonnet)."""
    def pairs(kv):
        need(len({k for k, _ in kv}) == len(kv), f"clé JSON dupliquée : {[k for k, _ in kv]}")
        return dict(kv)
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ExclusionError(f"JSON illisible : {exc}") from exc


def check_build(build: Path, manifest_sha: str) -> dict:
    """MANIFEST épinglé ; chaque sortie listée rehachée ; aucun fichier en plus ni en moins."""
    mp = build / "MANIFEST.json"
    need(mp.is_file() and sha(mp) == manifest_sha, "MANIFEST du constructeur différent du SHA épinglé")
    man = json.loads(mp.read_text())
    need(isinstance(man, dict) and isinstance(man.get("sorties"), dict) and isinstance(man.get("entrees"), dict),
         "MANIFEST du constructeur : format")
    sur_disque = {str(p.relative_to(build)) for p in build.rglob("*") if p.is_file()} - {"MANIFEST.json"}
    need(sur_disque == set(man["sorties"]), f"fichiers du constructeur en plus ou en moins : "
                                            f"{sorted(sur_disque ^ set(man['sorties']))[:5]}")
    for rel, h in man["sorties"].items():
        need(sha(build / rel) == h, f"sortie du constructeur modifiée : {rel}")
    return man


def check_decisions(dec, manifest_sha: str, zones: dict) -> tuple[dict, dict]:
    """Schéma strict de decisions.json ; renvoie ({tige: (motif, note)}, {tige: (nature, note)})."""
    need(isinstance(dec, dict) and set(dec) == {"build_manifest_sha256", "regardees", "exclusions", "difficiles"},
         "decisions.json : clés attendues build_manifest_sha256, regardees, exclusions, difficiles")
    need(dec["build_manifest_sha256"] == manifest_sha, "decisions.json écrit pour une autre sortie du constructeur")
    reg = dec["regardees"]
    need(isinstance(reg, list) and all(isinstance(t, str) for t in reg), "regardees : liste de tiges (chaînes) attendue")
    need(len(reg) == len(set(reg)) and set(reg) == set(zones),
         "toutes les zones doivent être regardées, une fois chacune (regardees)")
    exc, dif = {}, {}
    for key, table, champ, ok in (("exclusions", exc, "motif", set(MOTIFS)), ("difficiles", dif, "nature", set(DIFFICILES))):
        need(isinstance(dec[key], list), f"{key} : liste attendue")
        for x in dec[key]:
            need(isinstance(x, dict) and set(x) == {"stem", champ, "note"} and all(isinstance(x[k], str) for k in x),
                 f"{key} : entrée mal formée (stem, {champ}, note : chaînes) : {str(x)[:80]}")
            need(x["stem"] in zones, f"{key} : tige inconnue ou qui n'est pas une zone : {x['stem']!r}")
            need(x[champ] in ok, f"{key} : {champ} hors de la liste fermée : {x[champ]!r} ({x['stem']})")
            need(x["note"].strip() or key == "difficiles", f"exclusion sans note : {x['stem']}")  # mineur 5
            need(x["stem"] not in table, f"{key} : zone en double : {x['stem']}")
            table[x["stem"]] = (x[champ], x["note"])
    need(not set(exc) & set(dif), f"zone à la fois exclue et difficile : {sorted(set(exc) & set(dif))[:3]}")
    return exc, dif


def main() -> None:
    from PIL import Image, ImageDraw

    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--build-manifest-sha256", required=True)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--sortie", required=True)
    a = ap.parse_args()
    build, out = Path(a.build), Path(a.sortie)
    need(not out.exists() or not any(out.iterdir()), "dossier de sortie non vide")
    need(out.resolve() != build.resolve(), "la sortie ne peut pas être le dossier du constructeur")

    man = check_build(build, a.build_manifest_sha256)
    for rel in ("docs/READER_R7A_PREREG.txt", "scripts/reader_r7a_build.py"):  # mineur 2 : mêmes pièces qu'au constructeur
        need(isinstance(man["entrees"].get(rel), str) and man["entrees"][rel] == sha(HERE / rel),
             f"{rel} différent de celui du constructeur (ou absent de son MANIFEST)")
    raw = {n: (build / n).read_bytes() for n in ("items.jsonl", "contact.json")}  # mineur 3 : une lecture, hachée puis lue
    for n, b in raw.items():
        need(hashlib.sha256(b).hexdigest() == man["sorties"][n], f"{n} modifié pendant la lecture")
    items = [json.loads(x) for x in raw["items.jsonl"].decode("utf-8").splitlines()]
    contact = json.loads(raw["contact.json"])
    zones = {i["stem"]: i for i in items if i["part"] == "vide"}
    need(all(z.get("type_page") in TYPES and z.get("famille") in FAMILLES for z in zones.values()),
         "zone de type de page ou de famille inconnus")  # mineur 4
    need(len(zones) == sum(i["part"] == "vide" for i in items), "tiges de zone en double dans items.jsonl")
    need([c["stem"] for c in contact] == list(zones) and [c["n"] for c in contact] == list(range(len(contact))),
         "contact.json ne suit pas les zones d'items.jsonl, dans l'ordre")
    fichier = {c["stem"]: c["fichier"] for c in contact}
    dec_raw = Path(a.decisions).read_bytes()
    exc, dif = check_decisions(strict_json(dec_raw), a.build_manifest_sha256, zones)

    retenus = [i for i in items if i["stem"] not in exc]
    gardees = [z for s, z in zones.items() if s not in exc]
    comptes = {m: {t: {f: sum(1 for s, (mm, _) in exc.items() if mm == m and zones[s]["type_page"] == t
                              and zones[s]["famille"] == f) for f in FAMILLES} for t in TYPES} for m in MOTIFS}
    pages = sorted({z["page_key"] for z in zones.values()})
    par_page = {p: sum(z["page_key"] == p for z in gardees) for p in pages}
    rapport = {"zones": len(zones), "exclues": len(exc), "retenues": len(gardees),
               "exclues_par_motif": {m: sum(mm == m for mm, _ in exc.values()) for m in MOTIFS},
               "exclues_par_motif_type_famille": comptes,
               "retenues_par_type_famille": {t: {f: sum(z["type_page"] == t and z["famille"] == f for z in gardees)
                                                 for f in FAMILLES} for t in TYPES},
               "retenues_par_page": par_page, "pages_avec_zone_retenue": sum(v > 0 for v in par_page.values()),
               "plancher": {"seuil": PLANCHER, "atteint": len(gardees) >= PLANCHER,
                            "note": "sans arrêt (pA) : sous le plancher, le rapport donne effectif et IC sans verdict"},
               "difficiles_par_nature": {n: sum(nn == n for nn, _ in dif.values()) for n in DIFFICILES},
               "exclusions": [{"stem": s, "motif": m, "libelle": MOTIFS[m], "note": nt, "page_key": zones[s]["page_key"],
                               "famille": zones[s]["famille"], "type_page": zones[s]["type_page"], "fichier": fichier[s]}
                              for s, (m, nt) in exc.items()],
               "difficiles": [{"stem": s, "nature": n, "note": nt, "fichier": fichier[s]} for s, (n, nt) in dif.items()]}

    out.mkdir(parents=True, exist_ok=True)
    (out / "items_retenus.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in retenus))
    (out / "exclusions.json").write_text(json.dumps(rapport, indent=1, ensure_ascii=False))
    rangs = [(m, s) for m in MOTIFS for s, (mm, _) in exc.items() if mm == m]  # planche de contrôle, groupée par motif
    sheet = Image.new("L", (1240, 70 * len(rangs) + 30), 255)
    d = ImageDraw.Draw(sheet)
    d.text((5, 5), f"exclusions : {len(exc)} sur {len(zones)} zones ; MANIFEST constructeur {a.build_manifest_sha256[:12]}", fill=0)
    for k, (m, s) in enumerate(rangs):
        y = 30 + 70 * k
        with Image.open(build / fichier[s]) as im:
            t = im.convert("L")
        t.thumbnail((900, 60))
        d.text((5, y), f"{m} {MOTIFS[m]}"[:40], fill=0)
        d.text((5, y + 14), s[-40:], fill=0)
        sheet.paste(t, (330, y))
    sheet.save(out / "planche_exclusions.png")
    entrees = {"build_MANIFEST.json": a.build_manifest_sha256, "decisions.json": hashlib.sha256(dec_raw).hexdigest(),
               "scripts/reader_r7a_exclure.py": sha(HERE / "scripts/reader_r7a_exclure.py"),
               "docs/READER_R7A_PREREG.txt": sha(HERE / "docs/READER_R7A_PREREG.txt"),
               "scripts/reader_r7a_build.py": man["entrees"]["scripts/reader_r7a_build.py"]}
    manifest = {"entrees": entrees, "sorties": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: rapport[k] for k in ("zones", "exclues", "retenues", "exclues_par_motif", "pages_avec_zone_retenue",
                                              "plancher", "difficiles_par_nature")}, indent=1, ensure_ascii=False))
    if not rapport["plancher"]["atteint"]:
        print(f"SOUS LE PLANCHER : {len(gardees)} zones retenues ; rapport sans verdict", flush=True)


if __name__ == "__main__":
    main()
