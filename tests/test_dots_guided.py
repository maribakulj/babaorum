import math

import numpy as np
import torch

from babaorum.reader.ctc_prefix import CTCPrefixScorer
from babaorum.reader.dots_guided import CTCGuide

LABELS = ("", "a", "b", " ")
VOCAB = ["<p>", "a", "b", "ab", " ", "<eos>", "x"]  # jetons factices ; 0 = invite, 5 = fin, 6 hors codec
EOS = 5


def enc(ch):
    return [LABELS.index(ch)] if ch in LABELS[1:] else None


def decode(ids):
    return "".join(VOCAB[i] for i in ids if i not in (0, EOS))


def scorer():
    x = np.random.default_rng(0).normal(size=(10, len(LABELS)))
    return CTCPrefixScorer(x - np.logaddexp.reduce(x, axis=1, keepdims=True), enc)


def test_deltas_and_eos():
    s = scorer()
    g = CTCGuide(s, decode, prompt_len=1, eos_ids={EOS}, lam=1.0, top_k=len(VOCAB), pen=-7.0)
    ids = torch.tensor([[0, 1]])  # invite puis « a »
    sc = torch.log_softmax(torch.zeros(1, len(VOCAB)), -1)
    out = g(ids, sc)[0]
    assert math.isclose(out[2].item(), s.prefix("ab") - s.prefix("a"), rel_tol=1e-6)
    assert math.isclose(out[3].item(), s.prefix("aab") - s.prefix("a"), rel_tol=1e-6)
    assert math.isclose(out[EOS].item(), s.full("a") - s.prefix("a"), rel_tol=1e-6)
    assert math.isclose(out[6].item(), -7.0, rel_tol=1e-6)  # « x » hors codec : pénalité seule


def test_top_k_and_mix():
    s = scorer()
    g = CTCGuide(s, decode, prompt_len=1, eos_ids={EOS}, lam=0.3, top_k=2)
    sc = torch.tensor([[-9.0, -0.1, -3.0, -0.5, -9.0, -9.0, -9.0]])
    out = g(torch.tensor([[0]]), sc)[0]
    assert torch.isinf(out[[0, 2, 4, 6]]).all()  # hors des 2 meilleurs
    assert torch.isfinite(out[EOS])  # la fin reste toujours candidate
    assert math.isclose(out[1].item(), 0.7 * -0.1 + 0.3 * s.prefix("a"), rel_tol=1e-6)


def test_cumulative_equals_prefix():
    # somme des Δ le long d'une hypothèse = préfixe CTC du texte final (λ = 1)
    s = scorer()
    g = CTCGuide(s, decode, prompt_len=1, eos_ids={EOS}, lam=1.0, top_k=len(VOCAB))
    seq, tot = [0], 0.0
    for v in (3, 4, 1, EOS):
        out = g(torch.tensor([seq]), torch.log_softmax(torch.zeros(1, len(VOCAB)), -1))[0]
        tot += out[v].item()
        seq.append(v)
    assert math.isclose(tot, s.full("ab a"), rel_tol=1e-6)


def test_eos_always_candidate_and_floor():
    # treillis trop court : aucun jeton ne tient ; la fin reste candidate et rien n'est −∞ parmi les candidats
    x = np.random.default_rng(1).normal(size=(2, len(LABELS)))
    s = CTCPrefixScorer(x - np.logaddexp.reduce(x, axis=1, keepdims=True), enc)
    g = CTCGuide(s, decode, prompt_len=1, eos_ids={EOS}, lam=0.5, top_k=2)
    sc = torch.tensor([[-9.0, -0.1, -0.2, -9.0, -9.0, -20.0, -9.0]])  # eos hors du top-2
    out = g(torch.tensor([[0, 3, 3]]), sc)[0]  # « abab » ne tient pas dans T = 2
    assert torch.isfinite(out[[1, 2, EOS]]).all()
    assert math.isclose(out[1].item(), 0.5 * -0.1 + 0.5 * -50.0, rel_tol=1e-6)


def test_forbidden_special_tokens():
    s = scorer()
    g = CTCGuide(s, decode, prompt_len=1, eos_ids={EOS}, lam=0.5, top_k=len(VOCAB), forbidden={0, EOS})
    out = g(torch.tensor([[0]]), torch.log_softmax(torch.zeros(1, len(VOCAB)), -1))[0]
    assert torch.isinf(out[0]) and torch.isfinite(out[EOS])  # spécial interdit ; la fin reste permise


def test_out_of_codec_removed_before_compaction():
    s = scorer()
    g = CTCGuide(s, decode, prompt_len=1, eos_ids={EOS}, lam=1.0, top_k=len(VOCAB), pen=-7.0)
    assert g.ctc_text("a x b") == ("a b", 1, False)
    out = g(torch.tensor([[0, 1, 4]]), torch.log_softmax(torch.zeros(1, len(VOCAB)), -1))[0]  # « a » puis « x »
    assert math.isclose(out[6].item(), (s.prefix("a") - s.prefix("a")) - 7.0, abs_tol=1e-9)


def test_non_latin_mask():
    from babaorum.reader.dots_guided import non_latin_ids

    toks = ["é", "ſt", "ﬁ", "ت", "惟", "Ω", "12", "—", "é", "�", " ab", "Ж"]
    bad = non_latin_ids(lambda i: toks[i], len(toks))
    assert bad == {3, 4, 5, 11}


def test_allowed_vocab():
    from babaorum.reader.dots_guided import allowed_vocab

    toks = ["a", "ab", " b", "x", "", "�", "�", "é", "惟"]  # 5, 6 : fragments ; seul 5 compose « é »
    ok_char = lambda c: c.isspace() or c in "abé"  # noqa: E731
    enc = lambda text: [5, 6] if text.strip() == "é" else []  # noqa: E731
    assert allowed_vocab(lambda i: toks[i], len(toks), ok_char) == {0, 1, 2, 7}
    assert allowed_vocab(lambda i: toks[i], len(toks), ok_char, encode=enc) == {0, 1, 2, 5, 6, 7}


def test_last_step_pays_full_score():
    # audit pT B2 : au dernier pas permis, un jeton non-fin paie complet − préfixe comme une fin
    s = scorer()
    g = CTCGuide(s, decode, prompt_len=1, eos_ids={EOS}, lam=1.0, top_k=len(VOCAB), pen=-7.0, max_new=2)
    sc = torch.log_softmax(torch.zeros(1, len(VOCAB)), -1)
    out = g(torch.tensor([[0, 1]]), sc)[0]  # un jeton déjà généré : le prochain est le 2e et dernier
    assert math.isclose(out[2].item(), s.full("ab") - s.prefix("a"), rel_tol=1e-6)
    assert math.isclose(out[EOS].item(), s.full("a") - s.prefix("a"), rel_tol=1e-6)
    g2 = CTCGuide(s, decode, prompt_len=1, eos_ids={EOS}, lam=1.0, top_k=len(VOCAB), pen=-7.0, max_new=5)
    assert math.isclose(g2(torch.tensor([[0, 1]]), sc)[0][2].item(), s.prefix("ab") - s.prefix("a"), rel_tol=1e-6)
