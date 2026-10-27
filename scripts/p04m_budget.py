"""AV-P04, décision budgétaire de pA (pQ B1) : registre CUMULÉ des tentatives Kaggle de la mesure (porte et deux passes, relances comprises),
borne 24 h. Un carnet ne voit pas les sessions précédentes : le registre est tenu sur le Mac, à côté de l'avenant.
- check  : AVANT tout `kaggle kernels push` de la mesure. Refuse (code 1) si cumul + 12 h (une session au plus) > 24 h.
- record : APRÈS chaque tentative, quelle qu'en soit l'issue. Les heures sont celles du rapport de la passe (`elapsed_hours`) ; sans rapport
           (plantage, coupure, dépassement), on compte 12 h, la borne de la session, par prudence.
- FUMÉE (pA, 25/10) : hors des 24 h, mais bornée à 1 h et à UNE tentative ; consignée à part (category "fumee"), jamais comptée dans les 24 h.
  check --category fumee refuse s'il existe déjà une fumée ; record --category fumee signale tout dépassement d'1 h (arrêt, on en reparle).
    python p04m_budget.py check  --ledger <ledger.json> [--category fumee]
    python p04m_budget.py record --ledger <ledger.json> --kernel mabaku/babaorum-p04m-passe1 --version 1 --status COMPLETE [--report r.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

BOUND_H = 24.0
SESSION_H = 12.0
SMOKE_H = 1.0


def load(p: Path) -> list:
    return json.loads(p.read_text()) if p.exists() else []


def total(ledger: list) -> float:
    return sum(e["hours"] for e in ledger if e.get("category", "mesure") == "mesure")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--ledger", required=True)
    c.add_argument("--category", choices=("mesure", "fumee"), default="mesure")
    r = sub.add_parser("record")
    r.add_argument("--ledger", required=True)
    r.add_argument("--kernel", required=True)
    r.add_argument("--version", type=int, required=True)
    r.add_argument("--status", required=True)
    r.add_argument("--report")
    r.add_argument("--category", choices=("mesure", "fumee"), default="mesure")
    a = ap.parse_args()
    ledger_p = Path(a.ledger)
    ledger = load(ledger_p)
    if a.cmd == "check" and a.category == "fumee":
        n = sum(1 for e in ledger if e.get("category") == "fumee")
        print(json.dumps({"smoke_attempts": n, "push_allowed": n == 0, "bound_hours": SMOKE_H}))
        return 0 if n == 0 else 1
    if a.cmd == "check":
        t = total(ledger)
        ok = t + SESSION_H <= BOUND_H
        print(json.dumps({"attempts": len(ledger), "hours_so_far": t, "next_session_bound": SESSION_H, "push_allowed": ok}))
        return 0 if ok else 1
    if any(e["kernel"] == a.kernel and e["version"] == a.version for e in ledger):
        print("this attempt is already recorded", file=sys.stderr)
        return 1
    bound = SMOKE_H if a.category == "fumee" else SESSION_H
    hours, source = bound if a.category == "mesure" else None, "borne de session (pas de rapport)"
    if a.report and Path(a.report).is_file():
        h = json.loads(Path(a.report).read_text()).get("elapsed_hours")
        if isinstance(h, (int, float)) and not isinstance(h, bool) and math.isfinite(h) and h > 0:
            hours, source = float(h), "rapport de la passe"
    if a.category == "mesure" and hours is not None and hours > SESSION_H:
        hours, source = SESSION_H, "rapport au-delà de la session : borne de session"
    if a.category == "fumee" and hours is None:
        source = "fumée sans rapport : durée inconnue, dépassement présumé"
    ledger.append({"kernel": a.kernel, "version": a.version, "category": a.category, "status": a.status, "hours": hours, "hours_source": source,
                   "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    ledger_p.write_text(json.dumps(ledger, ensure_ascii=False, indent=1) + "\n")
    if a.category == "fumee":
        over = hours is None or hours > SMOKE_H
        print(json.dumps({"smoke_hours": hours, "over_1h": over, "action": "ARRÊT, on en reparle (pA)" if over else "ok"}))
        return 1 if over else 0
    print(json.dumps({"attempts": len(ledger), "hours_so_far": total(ledger), "over_bound": total(ledger) > BOUND_H}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
