"""Tests du convertisseur REF-GALLICA → VT R7(a) (scripts/ref_gallica_vt_convert.py) : aller-retour sur pages
synthétiques et sur un vrai training.jsonl de BBVLM, contrôle des lignes contre la livraison BBVLM, statut de retrait,
acceptation par check_vt du constructeur R7(a) (y compris sur le fichier écrit par le CLI), refus codés."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, HERE / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


conv = _load("ref_gallica_vt_convert", "scripts/ref_gallica_vt_convert.py")
PK = "gallica:9000000/f3"
# textes « difficiles » : accents, double espace, insécable, ligature, s long, guillemets, NFD, U+2028/U+2029/U+0085, \r
TEXTES = ["LA FRONDE — 16 DÉCEMBRE", "l’œuvre  de ſaint « Jean »", "fin de ligne ", "été (NFD)", "ﬁn",
          "sep ligne", "para graphe", "suite\u0085NEL", "retour\rchariot"]
BBVLM_REEL = Path.home() / "BBVLM/runs/industrialisation-20261007/repair-fronde-v3/training.jsonl"


def jsonl(objets, fin=b"\n"):
    return b"".join(json.dumps(o, ensure_ascii=False).encode("utf-8") + fin for o in objets)


def page(textes=TEXTES, page_id="P_abc", ids=None, extra=None):
    ids = ids or [f"L{k:04d}" for k in range(1, len(textes) + 1)]
    return [{"page_id": page_id, "line_id": i, "block_id": "B1", "bbox": [0, 0, 1, 1], "text": t, **(extra or {})}
            for i, t in zip(ids, textes)]


def test_aller_retour_exact_textes_difficiles():
    src = page()
    vt, retirees = conv.convertir(jsonl(src), jsonl(src), PK, "presse")
    assert retirees == [] and vt["page_key"] == PK
    assert [(l["id"], l["text"]) for l in vt["lines"]] == [(r["line_id"], r["text"]) for r in src]  # ordre, ids, octets
    assert [l["text"].encode("utf-8") for l in vt["lines"]] == [t.encode("utf-8") for t in TEXTES]


def test_crlf_accepte_sans_toucher_au_texte():
    src = page()
    vt, _ = conv.convertir(jsonl(src, b"\r\n"), jsonl(src), PK, "presse")
    assert [l["text"] for l in vt["lines"]] == TEXTES


@pytest.mark.skipif(not BBVLM_REEL.exists(), reason="training.jsonl réel de BBVLM absent")
def test_aller_retour_sur_vrai_fichier_bbvlm():
    raw = BBVLM_REEL.read_bytes()
    vt, retirees = conv.convertir(raw, raw, "gallica:6703127/f3", "presse")
    src = [json.loads(x) for x in raw.split(b"\n") if x.strip()]
    assert retirees == []
    assert len(src) > 0 and [(l["id"], l["text"]) for l in vt["lines"]] == [(r["line_id"], r["text"]) for r in src]


def test_cli_de_bout_en_bout_puis_check_vt(tmp_path):
    b = _load("reader_r7a_build", "scripts/reader_r7a_build.py")
    e, s, o = tmp_path / "relu.jsonl", tmp_path / "source.jsonl", tmp_path / "vt.json"
    e.write_bytes(jsonl(page()))
    s.write_bytes(jsonl(page()))
    cmd = [sys.executable, str(HERE / "scripts/ref_gallica_vt_convert.py"), "--entree", str(e), "--source-bbvlm", str(s),
           "--page-key", PK, "--type", "presse", "--sortie", str(o)]
    env = {"PYTHONPATH": str(HERE / "src")}
    assert subprocess.run(cmd, env=env, capture_output=True).returncode == 0
    vt = json.loads(o.read_text(encoding="utf-8"))
    lignes = [{"stem": f"{PK}#{l['id']}", "truth_id": l["id"], "truth": l["text"]} for l in vt["lines"]]
    b.check_vt(PK, o, lignes)  # le fichier ÉCRIT PAR LE CLI passe la jointure exacte du constructeur
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "SORTIE_EXISTANTE" in r.stderr
    r = subprocess.run(cmd[:-1] + [str(tmp_path / "absent/vt.json")], env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "SORTIE_INVALIDE" in r.stderr
    c = list(cmd)
    c[3], c[-1] = str(tmp_path / "absent.jsonl"), str(tmp_path / "vt2.json")  # c[2] = « --entree »
    r = subprocess.run(c, env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "ENTREE_ILLISIBLE" in r.stderr and not (tmp_path / "vt2.json").exists()
    assert not list(tmp_path.glob(".vt-*.tmp"))  # aucun fichier temporaire laissé


def test_ligne_supprimee_par_relecteur_retiree_et_comptee():
    src = [{"page_id": "P", "line_id": f"L{k}", "text": t} for k, t in enumerate(["titre", "", "~~~", "corps"], 1)]
    relu = [dict(x) for x in src]
    relu[1]["statut"] = relu[2]["statut"] = "supprimee_par_relecteur"  # filet (vide) et ornement lu
    vt, retirees = conv.convertir(jsonl(relu), jsonl(src), PK, "presse")
    assert [l["id"] for l in vt["lines"]] == ["L1", "L4"] and retirees == ["L2", "L3"]
    assert [l["text"] for l in vt["lines"]] == ["titre", "corps"]


def test_ligne_vide_sans_statut_refusee():
    src = [{"page_id": "P", "line_id": "L1", "text": "titre"}, {"page_id": "P", "line_id": "L2", "text": ""}]
    with pytest.raises(conv.VTRefus) as e:
        conv.convertir(jsonl(src), jsonl(src), PK, "presse")
    assert e.value.code == "TEXTE_VIDE"


SRC = page()


@pytest.mark.parametrize("relu, source, pk, typ, code", [
    (jsonl(page(ids=["L1", "L2", "L1", "L4", "L5", "L6", "L7", "L8", "L9"])), jsonl(SRC), PK, "presse", "ID_EN_DOUBLE"),
    (b'{"page_id": "P", "text": "a"}\n', jsonl(SRC), PK, "presse", "LIGNE_SANS_ID"),
    (b'{"page_id": "P", "line_id": "", "text": "a"}\n', jsonl(SRC), PK, "presse", "LIGNE_SANS_ID"),
    (b'{"page_id": "P", "line_id": 3, "text": "a"}\n', jsonl(SRC), PK, "presse", "ID_INVALIDE"),
    (b'{"page_id": "P", "line_id": " L1", "text": "a"}\n', jsonl(SRC), PK, "presse", "ID_INVALIDE"),
    (b'{"line_id": "L1", "text": "a"}\n', jsonl(SRC), PK, "presse", "CHAMP_MANQUANT"),
    (b'{"page_id": ["P"], "line_id": "L1", "text": "a"}\n', jsonl(SRC), PK, "presse", "PAGE_ID_INVALIDE"),
    (jsonl(page(page_id="Q")), jsonl(SRC), PK, "presse", "PAGE_ID_DIFFERENT"),
    (jsonl(page(textes=["a"], page_id="P1")) + jsonl(page(textes=["b"], page_id="P2", ids=["L9"])), jsonl(SRC), PK,
     "presse", "PAGE_ID_MULTIPLES"),
    (jsonl(SRC[:-1]), jsonl(SRC), PK, "presse", "NB_LIGNES"),
    (jsonl(SRC[1:] + SRC[:1]), jsonl(SRC), PK, "presse", "IDS_DIFFERENTS"),
    (jsonl([{**SRC[0], "text": None}] + SRC[1:]), jsonl(SRC), PK, "presse", "TEXTE_INVALIDE"),
    (jsonl([{**SRC[0], "text": "  "}] + SRC[1:]), jsonl(SRC), PK, "presse", "TEXTE_VIDE"),
    (jsonl([{**SRC[0], "statut": "douteuse"}] + SRC[1:]), jsonl(SRC), PK, "presse", "STATUT_INCONNU"),
    (jsonl([{**x, "statut": "supprimee_par_relecteur"} for x in SRC]), jsonl(SRC), PK, "presse", "PAGE_VIDE"),
    (jsonl(SRC)[:-3], jsonl(SRC), PK, "presse", "JSON_INVALIDE"),
    (b"[1, 2]\n", jsonl(SRC), PK, "presse", "JSON_INVALIDE"),
    (b'{"page_id": "P", "line_id": "L1", "text": "a", "text": "b"}\n', jsonl(SRC), PK, "presse", "CLE_JSON_DOUBLE"),
    (b"\xef\xbb\xbf" + jsonl(SRC), jsonl(SRC), PK, "presse", "BOM_INTERDIT"),
    (b'{"page_id": "P", "line_id": "L1", "text": "\xff"}\n', jsonl(SRC), PK, "presse", "ENCODAGE_INVALIDE"),
    (json.dumps({**SRC[0], "text": "a\ud800"}).encode("ascii") + b"\n" + jsonl(SRC[1:]), jsonl(SRC), PK, "presse",
     "ENCODAGE_INVALIDE"),  # substitut isolé, échappé en JSON ASCII
    (json.dumps({**SRC[0], "line_id": "L\ud800"}).encode("ascii") + b"\n" + jsonl(SRC[1:]), jsonl(SRC), PK, "presse",
     "ENCODAGE_INVALIDE"),  # substitut isolé dans l'id (mineur pT)
    (b"\n\n", jsonl(SRC), PK, "presse", "PAGE_VIDE"),
    (jsonl(SRC), jsonl(SRC), "gallica:bpt6k9000000t/f3", "presse", "CLE_NON_CANONIQUE"),
    (jsonl(SRC), jsonl(SRC), "pas-une-cle", "presse", "CLE_REFUSEE"),
    (jsonl(SRC), jsonl(SRC), "gallica_book:bpt6k76354152/f2", "livre", "CLE_REFUSEE"),  # espace réservé, fermé
    (jsonl(SRC), jsonl(SRC), PK, "livre", "CLE_REFUSEE"),  # clé de presse donnée pour un livre
    (jsonl(SRC), jsonl(SRC), "gallica_book:bpt6k76354152/f2", "presse", "CLE_REFUSEE"),  # clé de livre pour la presse
    (jsonl(SRC), jsonl(SRC), PK, "revue", "TYPE_INCONNU"),
])
def test_refus_codes(relu, source, pk, typ, code):
    with pytest.raises(conv.VTRefus) as e:
        conv.convertir(relu, source, pk, typ)
    assert e.value.code == code
