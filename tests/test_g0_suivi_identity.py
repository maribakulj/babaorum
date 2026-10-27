"""G0-04 v4.1 point 1 (identity) and point 2 (scope), one test per contract line; synthetic keys only."""
import pytest

from babaorum.g0_suivi_identity import (NAMESPACES, NO_PAGE_KEY, Alias, PageIdentity, Usage, applicable_usages,
                                        page_identity, page_key_or_refusal)
from babaorum.identity import IdentityError
from babaorum.legacy.data.doc_key import noid_check

SHA1 = "0585bc2c3229f045a868998f0cc8c2223ccdbfa4"
VALID_ARK = "bpt6k6330162" + noid_check("bpt6k6330162")


# point 1 — Gallica and KB delegate to babaorum.identity; the three Paris-Soir aliases give one page
def test_three_paris_soir_aliases_give_the_same_page():
    pages = {page_identity(v, "gallica").page for v in
             ("Paris-Soir_7636525_D0000001", "read_7636525-001", "bnf_7636525-001", "gallica:7636525/f1")}
    assert pages == {"gallica:7636525/f1"}
    assert page_identity("read_7636525-001", "gallica").document == "gallica:7636525"


def test_document_without_folio_is_refused_for_a_page():
    for value in ("7636525", "gallica:7636525", "bpt6k" + "6330162" + noid_check("bpt6k6330162")):
        assert page_key_or_refusal(value, "gallica") == (None, [NO_PAGE_KEY])


def test_valid_ark_with_folio_and_invalid_check_character():
    assert page_identity(f"{VALID_ARK}/f3", "gallica").page == "gallica:6330162/f3"
    with pytest.raises(IdentityError):
        page_identity("bpt6k6330162x/f3", "gallica")


def test_opaque_arks_are_refused_and_never_merged_with_bpt6k():
    for value in ("btv1b8452305m/f3", "bd6t5379436f/f2"):
        assert page_key_or_refusal(value, "gallica") == (None, [NO_PAGE_KEY])


def test_reserved_and_unknown_namespaces_are_refused():
    assert page_key_or_refusal("x", "gallica_book") == (None, [NO_PAGE_KEY])
    assert page_key_or_refusal("x", "impresso") == (None, [NO_PAGE_KEY])


def test_kb_page_and_kb_value_in_the_gallica_namespace():
    assert page_identity("kb_00531074", "kb") == PageIdentity("kb", "kb:00531074", None, None)
    assert page_key_or_refusal("kb:00531074", "gallica") == (None, [NO_PAGE_KEY])


# point 1 / v4.1 §1 — Wikimedia keeps the sealed prefix and an explicit view
def test_two_views_of_one_wikimedia_file_stay_distinct():
    a = page_identity(f"wikimedia-file-sha1:{SHA1}/f3", "wikimedia")
    b = page_identity(f"wikimedia-file-sha1:{SHA1}/f13", "wikimedia")
    assert a.page != b.page and a.document == b.document == f"wikimedia-file-sha1:{SHA1}"
    assert a.document_scope == "fichier"
    assert page_identity(f"wikimedia-file-sha1:{SHA1}/single-image", "wikimedia").page.endswith("/single-image")


def test_wikimedia_sha1_alone_or_short_form_is_refused():
    for value in (f"wikimedia-file-sha1:{SHA1}", f"wikimedia:{SHA1}/f3", f"wikimedia-file-sha1:{SHA1}/f0"):
        assert page_key_or_refusal(value, "wikimedia") == (None, [NO_PAGE_KEY])


def test_as_name_and_canonical_form():
    a = page_identity("18840315_1-0003", "as")
    assert a == PageIdentity("as", "as:18840315_1/p3", "as:18840315_1", "document")
    assert page_identity(a.page, "as") == a
    assert page_key_or_refusal("18840315_1", "as") == (None, [NO_PAGE_KEY])


def test_nzz_key_from_the_ground_truth_file_name():
    path = "NZZ-black-letter-ground-truth-master/xml/NZZ_groundtruth/nzz_17800719_0_0_a1_p1_1.xml"
    a = page_identity(path, "nzz")
    assert a == PageIdentity("nzz", "nzz:17800719_0_0_a1_p1_1", None, None)
    assert page_identity(a.page, "nzz") == a
    assert page_key_or_refusal("nzz_1780.xml", "nzz") == (None, [NO_PAGE_KEY])


# extension of p9 (19/10): NewsEye/READ photos and the BnL pack, disjoint from gallica:
def test_newseye_photo_and_bnl_have_their_own_disjoint_namespaces():
    photo = page_identity("P1020460", "newseye_photo")
    bnl = page_identity("000000", "bnl")
    assert photo == PageIdentity("newseye_photo", "newseye_photo:P1020460", None, None)
    assert bnl == PageIdentity("bnl", "bnl:000000", None, None)
    for ident in (photo, bnl):
        assert not ident.page.startswith("gallica:") and ident.document is None
    assert page_key_or_refusal("P1020460", "gallica") == (None, [NO_PAGE_KEY])
    assert page_key_or_refusal("000000", "gallica") == (None, [NO_PAGE_KEY])
    assert page_key_or_refusal("newseye_photo:P1020460", "bnl") == (None, [NO_PAGE_KEY])


def test_every_canonical_form_reads_back_identically():
    samples = {"gallica": "Paris-Soir_7636525_D0000001", "kb": "kb_00531074", "wikimedia": f"wikimedia-file-sha1:{SHA1}/f3",
               "as": "18840315_1-0003", "nzz": "nzz_17800719_0_0_a1_p1_1.xml", "newseye_photo": "P1020460", "bnl": "000000"}
    assert set(samples) == set(NAMESPACES)
    for ns, value in samples.items():
        ident = page_identity(value, ns)
        assert page_identity(ident.page, ns) == ident


# point 1 — aliases between namespaces; point 2 — scope
def _ids():
    return (page_identity("Paris-Soir_7636525_D0000001", "gallica"),
            page_identity(f"wikimedia-file-sha1:{SHA1}/f1", "wikimedia"))


def test_proven_page_alias_cumulates_page_usages():
    gal, wik = _ids()
    usages = [Usage("page", gal.page, "train", "press/train"), Usage("page", wik.page, "eval", "presse_e7")]
    got, _ = applicable_usages(wik, usages, [Alias(gal, wik, proven_page=True, source="s")])
    assert {u.usage for u in got} == {"train", "eval"}


def test_unproven_alias_inherits_document_usages_only():
    gal, wik = _ids()
    usages = [Usage("page", gal.page, "train", "press/train"), Usage("document", gal.document, "train", "R2_LM")]
    got, _ = applicable_usages(wik, usages, [Alias(gal, wik, proven_page=False, source="s")])
    assert [(u.scope, u.source) for u in got] == [("document", "R2_LM")]


def test_document_training_refuses_a_page_with_no_individual_entry():
    """Decisive test of point 2: issue learnt by R2, candidate f2 with no page entry."""
    f2 = page_identity("Paris-Soir_7636525_D0000002", "gallica")
    got, labels = applicable_usages(f2, [Usage("document", "gallica:7636525", "train", "R2_LM")])
    assert [(u.usage, u.source) for u in got] == [("train", "R2_LM")] and labels == []


def test_page_usage_on_a_sibling_page_is_only_a_label():
    f1 = page_identity("Paris-Soir_7636525_D0000001", "gallica")
    f2 = page_identity("Paris-Soir_7636525_D0000002", "gallica")
    got, labels = applicable_usages(f2, [Usage("page", f1.page, "select", "e12_dev")])
    assert got == [] and labels == ["ISSUE_SHARED:select:e12_dev"]


def test_file_scope_reaches_every_view_of_a_wikimedia_file():
    a = page_identity(f"wikimedia-file-sha1:{SHA1}/f13", "wikimedia")
    got, _ = applicable_usages(a, [Usage("fichier", f"wikimedia-file-sha1:{SHA1}", "train", "presse_e7_train")])
    assert [u.source for u in got] == ["presse_e7_train"]


def test_unknown_scope_is_refused():
    with pytest.raises(ValueError):
        Usage("volume", "x", "train", "s")


# relecture Sonnet (21/10), points non bloquants intégrés
def test_proven_alias_also_inherits_the_other_side_document_usages():
    gal, wik = _ids()
    got, _ = applicable_usages(wik, [Usage("document", gal.document, "train", "R2_LM")], [Alias(gal, wik, proven_page=True, source="s")])
    assert [u.source for u in got] == ["R2_LM"]


def test_issue_shared_is_never_emitted_for_document_or_file_scope_nor_between_views_of_a_file():
    f2 = page_identity("Paris-Soir_7636525_D0000002", "gallica")
    _, labels = applicable_usages(f2, [Usage("document", "gallica:7636525", "eval", "e22")])
    assert labels == []
    a = page_identity(f"wikimedia-file-sha1:{SHA1}/f2", "wikimedia")
    _, labels = applicable_usages(a, [Usage("page", f"wikimedia-file-sha1:{SHA1}/f1", "eval", "presse_e7")])
    assert labels == []


def test_usage_keys_must_be_canonical_and_aliases_distinct():
    for key in ("read_7636525-001", None, ""):
        with pytest.raises(ValueError):
            Usage("page", key, "train", "s")
    gal, _ = _ids()
    with pytest.raises(ValueError):
        Alias(gal, gal, proven_page=True, source="s")
    gal2 = page_identity("Paris-Soir_7636525_D0000002", "gallica")
    for source in ("", "  ", None):
        with pytest.raises(ValueError):
            Alias(gal, gal2, proven_page=False, source=source)


# contre-vérification pT (21/10), trois bloquants et un mineur
def test_unproven_alias_reaches_every_page_of_the_linked_document_and_file():
    gal, wik = _ids()
    usages = [Usage("document", gal.document, "train", "R2_LM"), Usage("page", gal.page, "eval", "e22")]
    aliases = [Alias(gal, wik, proven_page=False, source="s")]
    other_view = page_identity(f"wikimedia-file-sha1:{SHA1}/f2", "wikimedia")
    got, _ = applicable_usages(other_view, usages, aliases)
    assert [(u.scope, u.source) for u in got] == [("document", "R2_LM")]
    as_p3 = page_identity("18840315_1-0003", "as")
    as_p7 = page_identity("18840315_1-0007", "as")
    got, labels = applicable_usages(as_p7, [Usage("document", gal.document, "train", "R2_LM")],
                                    [Alias(gal, as_p3, proven_page=False, source="s")])
    assert [u.source for u in got] == ["R2_LM"] and labels == []


def test_proven_page_usages_follow_a_chain_of_aliases_but_never_through_a_document_link():
    gal, wik = _ids()
    as_p1 = page_identity("18840315_1-0001", "as")
    train = [Usage("page", gal.page, "train", "press/train")]
    got, _ = applicable_usages(as_p1, train, [Alias(gal, wik, True, "s"), Alias(wik, as_p1, True, "s")])
    assert [u.source for u in got] == ["press/train"]
    got, _ = applicable_usages(as_p1, train, [Alias(gal, wik, True, "s"), Alias(wik, as_p1, False, "s")])
    assert got == []


def test_usage_rejects_keys_that_are_not_canonical_or_do_not_fit_the_scope():
    for scope, key in (("page", "gallica:7636525"), ("document", "gallica:7636525/f1"),
                       ("document", "gallica:07636525"), ("page", "gallica:7636525/f01"),
                       ("document", "kb:00531074"), ("fichier", "gallica:7636525"),
                       ("document", f"wikimedia-file-sha1:{SHA1}"), ("fichier", f"wikimedia-file-sha1:{SHA1}/f1"),
                       ("document", "as:18840315_01")):
        with pytest.raises(ValueError):
            Usage(scope, key, "train", "s")
    Usage("page", "gallica:7636525/f1", "train", "s")
    Usage("document", "gallica:7636525", "train", "s")
    Usage("document", "as:18840315_1", "train", "s")
    Usage("fichier", f"wikimedia-file-sha1:{SHA1}", "train", "s")


def test_as_and_nzz_dates_are_real_dates_and_as_rank_is_positive():
    for value, ns in (("18841315_1-0003", "as"), ("18840315_0-0003", "as"), ("nzz_17801319_0_0_a1_p1_1.xml", "nzz")):
        assert page_key_or_refusal(value, ns) == (None, [NO_PAGE_KEY])


# relecture Sonnet (22/10) : un espace sans document ne fait jamais passer un usage de page par un lien non prouvé
def test_unproven_alias_with_a_page_less_namespace_never_carries_page_usages():
    gal, _ = _ids()
    n1 = page_identity("nzz_17800719_0_0_a1_p1_1.xml", "nzz")
    n2 = page_identity("nzz_17800719_0_0_a1_p1_2.xml", "nzz")
    usages = [Usage("page", n1.page, "train", "nzz_train"), Usage("page", gal.page, "eval", "e22")]
    got, _ = applicable_usages(gal, usages, [Alias(gal, n1, proven_page=False, source="s")])
    assert [u.source for u in got] == ["e22"]  # its own usage only, never the nzz page's
    assert applicable_usages(n2, usages, [Alias(n1, n2, proven_page=False, source="s")]) == ([], [])
    got, _ = applicable_usages(n2, usages, [Alias(n1, n2, proven_page=True, source="s")])
    assert [u.source for u in got] == ["nzz_train"]


# contre-vérification pT (22/10), M2 : ISSUE_SHARED seulement pour une autre page établie du même numéro
def test_issue_shared_is_not_emitted_through_a_document_link_without_page_proof():
    gal, wik = _ids()
    wik_f2 = page_identity(f"wikimedia-file-sha1:{SHA1}/f2", "wikimedia")
    train = [Usage("page", gal.page, "train", "press/train")]
    assert applicable_usages(wik_f2, train, [Alias(gal, wik, proven_page=False, source="s")]) == ([], [])
    gal_f2 = page_identity("Paris-Soir_7636525_D0000002", "gallica")
    _, labels = applicable_usages(wik_f2, train, [Alias(gal_f2, wik_f2, proven_page=True, source="s")])
    assert labels == ["ISSUE_SHARED:train:press/train"]  # f2 is proven to be Gallica f2, an established other folio
