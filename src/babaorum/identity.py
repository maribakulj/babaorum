"""Strict C7 aliases, with distinct page and historical issue identities."""
import re
from .legacy.data.doc_key import gallica_page, noid_check


class IdentityError(ValueError):
    pass


_ARK = re.compile(
    r'(?:(?:https?://gallica\.bnf\.fr/)?ark:/12148/|gallica:)?'
    r'(bpt6k[0-9]+[0-9bcdfghjkmnpqrstvwxz])'
    r'(?:(?:/|\.)f([0-9]+))?(?:\.alto\.xml|\.xml)?')


def _value(value):
    if not isinstance(value, str) or not value.strip():
        raise IdentityError('unrecognized document/page identity')
    return value.strip()


def _ark_parts(value):
    match = _ARK.fullmatch(value)
    if not match:
        return None
    ark = match[1]
    if noid_check(ark[:-1]) != ark[-1]:
        raise IdentityError(f'invalid ARK NOID check digit: {value}')
    gid = ark[5:-1]
    if not gid or int(gid) <= 0:
        raise IdentityError(f'invalid ARK issue identity: {value}')
    return int(gid), int(match[2]) if match[2] is not None else None


def canonical_page(value):
    """A page requires an explicit folio; KB's stable key identifies only a page."""
    value = _value(value)
    match = re.fullmatch(r'gallica:(\d+)/f(\d+)', value)
    if match and int(match[1]) > 0 and int(match[2]) > 0:
        return f'gallica:{int(match[1])}/f{int(match[2])}'
    match = re.fullmatch(r'kb[:_]([0-9]{8})', value)
    if match:
        return 'kb:' + match[1]
    if value.startswith(('kb:', 'kb_')):
        raise IdentityError(f'unrecognized KB page identity: {value}')
    ark = _ark_parts(value)
    if ark:
        if ark[1] is None or ark[1] <= 0:
            raise IdentityError(f'explicit folio required for page identity: {value}')
        return f'gallica:{ark[0]}/f{ark[1]}'
    parsed = gallica_page(value)
    explicit_folio = re.search(r'(?:_[DT]|-)\d+$', value)
    if parsed and explicit_folio and int(parsed[0]) > 0 and parsed[1] > 0:
        return f'gallica:{int(parsed[0])}/f{parsed[1]}'
    raise IdentityError(f'unrecognized page identity or missing explicit folio: {value}')


def canonical_issue(value):
    """Canonical historical issue; a bare Gallica/ARK key never denotes page f1."""
    value = _value(value)
    match = re.fullmatch(r'gallica:(\d+)', value)
    if match and int(match[1]) > 0:
        return f'gallica:{int(match[1])}'
    ark = _ark_parts(value)
    if ark:
        if ark[1] is not None and ark[1] <= 0:
            raise IdentityError(f'invalid ARK folio: {value}')
        return f'gallica:{ark[0]}'
    parsed = gallica_page(value)
    if not value.startswith(('kb:', 'kb_')) and parsed and int(parsed[0]) > 0:
        return f'gallica:{int(parsed[0])}'
    try:
        page = canonical_page(value)
    except IdentityError as exc:
        raise IdentityError(f'unrecognized historical issue identity: {value}') from exc
    if page.startswith('gallica:'):
        return page.split('/f', 1)[0]
    raise IdentityError(f'KB page identity does not identify a historical issue: {value}')


def identities(value):
    """Page plus issue, with null issue for KB; never synthesizes a folio."""
    page = canonical_page(value)
    return page, page.split('/f', 1)[0] if page.startswith('gallica:') else None
