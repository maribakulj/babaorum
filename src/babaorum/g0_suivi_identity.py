"""G0-04 role `suivi`: strict page identity adapter (contract v4.1 point 1) and usage scope (point 2).

Contract: ~/Documents/reviews/babaorum-pipeline/LOT-G0-04-role-suivi.md, SHA-256 912639a1bc72fecae08c26b3d5f924a9f0d33fac1f3e40e677a39766b0dd4573
(design GO 21/10). Gallica and KB are delegated to the strict functions of babaorum.identity; axel's gallica_page and
gallica_id_of_ark are never called here (the first invents f1, the second keeps an invalid check character).
newseye_photo and bnl (request of p9, 19/10) are an extension outside that SHA: page keys only, disjoint from gallica:.
The namespace is always explicit: a bare value such as "000000" or "P1020460" is never guessed.
"""
import re
from datetime import datetime
from dataclasses import dataclass

from .identity import IdentityError, canonical_issue, canonical_page

NO_PAGE_KEY = 'SUIVI_NO_PAGE_KEY'
NAMESPACES = ('gallica', 'kb', 'wikimedia', 'as', 'nzz', 'newseye_photo', 'bnl')
RESERVED = ('gallica_book',)  # REF-GALLICA only, not used by this lot
PREFIX = {'gallica': 'gallica:', 'kb': 'kb:', 'wikimedia': 'wikimedia-file-sha1:', 'as': 'as:', 'nzz': 'nzz:',
          'newseye_photo': 'newseye_photo:', 'bnl': 'bnl:'}
SCOPES = ('page', 'document', 'fichier')

_WIKIMEDIA = re.compile(r'wikimedia-file-sha1:([0-9a-f]{40})/(?:f([0-9]+)|(single-image))')
_AS_CANON = re.compile(r'as:([0-9]{8})_([0-9]+)/p([0-9]+)')
_AS_NAME = re.compile(r'([0-9]{8})_([0-9]+)-([0-9]+)')
_NZZ_CANON = re.compile(r'nzz:([0-9]{8})_([0-9a-z_]+)')
_NZZ_NAME = re.compile(r'(?:.*/)?nzz_([0-9]{8})_([0-9a-z_]+)\.xml')
_PHOTO = re.compile(r'(?:newseye_photo:)?(P[0-9]{7})')
_BNL = re.compile(r'(?:bnl:)?([0-9]{6})')


@dataclass(frozen=True)
class PageIdentity:
    namespace: str
    page: str            # canonical page key, explicit view
    document: str | None  # canonical issue, book or file key; None when the namespace has none
    document_scope: str | None  # 'document' or 'fichier' (Wikimedia file), None with document


def _positive(text, value):
    if int(text) <= 0:
        raise IdentityError(f'explicit positive view required: {value}')
    return int(text)


def _date(text, value):
    try:
        datetime.strptime(text, '%Y%m%d')
    except ValueError:
        raise IdentityError(f'invalid date {text}: {value}') from None
    return text


def _foreign_prefix(value, namespace):
    for ns, prefix in PREFIX.items():
        if ns != namespace and value.startswith(prefix):
            raise IdentityError(f'{prefix} identity given in namespace {namespace}: {value}')


def page_identity(value, namespace):
    """Canonical page identity in an explicit namespace; raises IdentityError (refusal SUIVI_NO_PAGE_KEY) otherwise."""
    if namespace in RESERVED:
        raise IdentityError(f'namespace {namespace} is reserved for REF-GALLICA and not used here')
    if namespace not in NAMESPACES:
        raise IdentityError(f'unknown namespace: {namespace}')
    if not isinstance(value, str) or not value.strip():
        raise IdentityError('empty page identity')
    value = value.strip()
    _foreign_prefix(value, namespace)
    if namespace == 'gallica':
        page = canonical_page(value)
        if not page.startswith('gallica:'):
            raise IdentityError(f'not a Gallica page: {value}')
        return PageIdentity('gallica', page, canonical_issue(page), 'document')
    if namespace == 'kb':
        page = canonical_page(value)
        if not page.startswith('kb:'):
            raise IdentityError(f'not a KB page: {value}')
        return PageIdentity('kb', page, None, None)
    if namespace == 'wikimedia':
        m = _WIKIMEDIA.fullmatch(value)
        if not m:
            raise IdentityError(f'Wikimedia page needs wikimedia-file-sha1:<sha1>/fN or /single-image: {value}')
        view = 'single-image' if m[3] else f'f{_positive(m[2], value)}'
        return PageIdentity('wikimedia', f'wikimedia-file-sha1:{m[1]}/{view}', f'wikimedia-file-sha1:{m[1]}', 'fichier')
    if namespace == 'as':
        m = _AS_CANON.fullmatch(value) or _AS_NAME.fullmatch(value)
        if not m:
            raise IdentityError(f'AS page needs <AAAAMMJJ>_<rang>-<page> or as:<AAAAMMJJ>_<rang>/p<N>: {value}')
        issue = f'as:{_date(m[1], value)}_{_positive(m[2], value)}'
        return PageIdentity('as', f'{issue}/p{_positive(m[3], value)}', issue, 'document')
    if namespace == 'nzz':
        m = _NZZ_CANON.fullmatch(value) or _NZZ_NAME.fullmatch(value)
        if not m:
            raise IdentityError(f'NZZ page needs nzz_<AAAAMMJJ>_<suffixe>.xml or nzz:<AAAAMMJJ>_<suffixe>: {value}')
        return PageIdentity('nzz', f'nzz:{_date(m[1], value)}_{m[2]}', None, None)  # one VT = one page; no issue key in the contract
    if namespace == 'newseye_photo':
        m = _PHOTO.fullmatch(value)
        if not m:
            raise IdentityError(f'NewsEye/READ photo page needs P<7 digits>: {value}')
        return PageIdentity('newseye_photo', f'newseye_photo:{m[1]}', None, None)
    m = _BNL.fullmatch(value)
    if not m:
        raise IdentityError(f'BnL pack page needs <6 digits>: {value}')
    return PageIdentity('bnl', f'bnl:{m[1]}', None, None)


def page_key_or_refusal(value, namespace):
    """(identity, []) or (None, ['SUIVI_NO_PAGE_KEY'])."""
    try:
        return page_identity(value, namespace), []
    except IdentityError:
        return None, [NO_PAGE_KEY]


@dataclass(frozen=True)
class Usage:
    scope: str   # 'page' | 'document' | 'fichier'
    key: str     # canonical page key for 'page', canonical document or file key otherwise
    usage: str   # 'train' | 'dev' | 'select' | 'eval' ...
    source: str

    def __post_init__(self):  # keys must already be canonical AND fit the scope; nothing is accepted then lost
        if self.scope not in SCOPES:
            raise ValueError(f'unknown scope {self.scope}')
        if not isinstance(self.key, str) or _canonical_key(self.scope, self.key) != self.key:
            raise ValueError(f'usage key is not a canonical {self.scope} key: {self.key!r}')


def _namespace_of(key):
    for ns, prefix in PREFIX.items():
        if key.startswith(prefix):
            return ns
    return None


def _canonical_key(scope, key):
    """Canonical form of a key for this scope, or None when the key does not fit the scope."""
    ns = _namespace_of(key)
    if ns is None:
        return None
    try:
        if scope == 'page':
            return page_identity(key, ns).page
        if '/' in key:  # a page key is never a document or file key
            return None
        if scope == 'fichier':
            return key if ns == 'wikimedia' and re.fullmatch(r'wikimedia-file-sha1:[0-9a-f]{40}', key) else None
        if ns == 'gallica':
            return canonical_issue(key)
        if ns == 'as':
            m = re.fullmatch(r'as:([0-9]{8})_([0-9]+)', key)
            return f'as:{_date(m[1], key)}_{_positive(m[2], key)}' if m else None
    except IdentityError:
        return None
    return None  # kb, nzz, newseye_photo, bnl: no document key in the contract


@dataclass(frozen=True)
class Alias:
    """A link between two identities, with its provenance (the file or decision that establishes it).

    Every alias links the two documents/files as wholes (v4 §1): their usages propagate conservatively over the whole
    connected component, with no depth limit (pT 22/10, M1). proven_page=False means "not proven AT THE PAGE", never
    "document link without proof"; only proven_page=True merges the two pages themselves."""
    left: PageIdentity
    right: PageIdentity
    proven_page: bool
    source: str

    def __post_init__(self):
        if self.left == self.right:
            raise ValueError('an alias links two different identities')
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError('an alias needs its provenance')


def _closure(pairs):
    """Equivalence classes (transitive closure) of the given pairs, as {member: frozenset}."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        parent[find(a)] = find(b)
    classes = {}
    for x in list(parent):
        classes.setdefault(find(x), set()).add(x)
    return {x: frozenset(c) for c in classes.values() for x in c}


def _whole(ident):
    """Node of the document-level link: the document/file, or the page itself when the namespace has no document."""
    return (ident.document_scope, ident.document) if ident.document else ('page', ident.page)


def applicable_usages(ident, usages, aliases=()):
    """Usages that apply to one page (point 2), sorted; and ISSUE_SHARED labels for page usages on sibling pages.

    Page usages follow the class of the page under the transitive closure of proven aliases. Document/file usages
    follow the class of its document/file under the closure of all aliases, whatever the folio of the queried page.
    """
    same_page = _closure((a.left.page, a.right.page) for a in aliases if a.proven_page).get(ident.page, {ident.page})
    wholes = _closure((_whole(a.left), _whole(a.right)) for a in aliases).get(_whole(ident), {_whole(ident)})
    got = [u for u in usages if (u.scope == 'page' and u.key in same_page)
           or (u.scope in ('document', 'fichier') and (u.scope, u.key) in wholes)]  # a ('page', ...) node never carries usages
    applied = sorted(set(got), key=lambda u: (u.scope, u.key, u.usage, u.source))
    # ISSUE_SHARED needs an established other page of the same issue: only issues of this page or of its proven
    # page aliases, where a different key in the same namespace is a different folio (pT 22/10, M2)
    issues = {_document_of_page_key(k) for k in same_page} - {None}
    labels = sorted({f'ISSUE_SHARED:{u.usage}:{u.source}' for u in usages
                     if u.scope == 'page' and u.key not in same_page and _document_of_page_key(u.key) in issues
                     and u not in applied})
    return applied, labels


def _document_of_page_key(key):
    ns = _namespace_of(key)
    if ns in ('gallica', 'as'):
        return page_identity(key, ns).document  # keys are canonical page keys (checked by Usage)
    return None
