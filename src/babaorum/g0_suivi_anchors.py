"""G0-04 role `suivi`: anchors and paths of the real input files (data only, no logic).

B1 in place (25/10/2026, pA agreement, GO technique pQ on the five files): the anchors below are the SHA-256 of the exact files.
Registry v3 in place (25/10/2026, pA agreement, GO technique pQ): usage f8537a3a7597, sources c8a4eb87f514 (p04_train 44 -> 12 pages,
ngram8 index reference in lm_texts); the other three anchors are unchanged.
Plugging the registry of B1, or a later version g0_page_usage_v<N>, means changing the values below in a reviewed change of THIS file only;
g0_suivi.py stays as audited. Paths are relative to the repository root.
"""
SUIVI_PATHS: dict[str, str] = {
    "usage": "config/g0_page_usage_v3.json",
    "sources": "config/g0_usage_sources_v3.json",
    "journals": "config/g0_journal_review_v1.json",
    "select_evidence": "config/g0_select_evidence_v1.json",
    "consultations": "config/g0_suivi_consultations_v1.json",
}
SUIVI_ANCHORS: dict[str, str | None] = {
    "usage": "f8537a3a7597739cd830af874c834a9dc29fd8cf1b130b4d8837f5e23306bb78",
    "sources": "c8a4eb87f51478d4152c1e7177acb19729a785f9aee09d7ecaf02c6b6062e0e6",
    "journals": "9ab1637382cd2b32d3418277b97f9ac08c55756cf7534cb38a84353b395e0971",
    "select_evidence": "2a84092a7865a11e64820ff28e52b746a52adbebe3e385ef6ac3618a0967a7c3",
    "consultations": "28c92439736aa36f4584cc4edccf93554552c82259c727297c1a1d7d4402cfbe",
}
