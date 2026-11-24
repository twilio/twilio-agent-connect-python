"""Refuse TLS connections whose certificate chain contains a distrusted intermediate.

The private keys of eight eMudhra intermediate CA certificates are compromised
(CVE-2026-111943, https://emudhra.we-have-a-problem.com), so anyone can use them
to issue a certificate that a default trust store accepts. Each intermediate is
identified by the SHA-256 of its DER-encoded SubjectPublicKeyInfo.

TAC's HTTP and WebSocket clients are built with :func:`distrusting_ssl_context`,
which fails the TLS handshake when the verified chain contains one of them. A
client or SSL context you supply yourself is not covered.

The chain is read from the handshake's verified chain: the public API on Python
3.13+, and the interpreter's equivalent on 3.10-3.12. If neither is available
the check is skipped and a warning is logged once.

Remove this once the Python runtime and system trust stores distrust these
intermediates.
"""

import hashlib
import ssl
from collections.abc import Iterable
from typing import Any

from tac.core.logging import get_logger

DISTRUSTED_SPKI_SHA256: frozenset[str] = frozenset(
    {
        "ebed6475c2c2d80fcb199f3fe8dd6e6b3ce60b77eb15d4e42c3dec0c6f5b5105",
        "49dd353dc2b919a3770645117d0e60c16115e7d1bc259eefe734b00ea434fe5f",
        "e0254c7fcdc265e2f0f8a428dabb75ab4ae039b4872d62f5ee57f0877d5a8a41",
        "5c0660a2b164910d8ff589c18e14552e9e0ec237dcefe2cb736d41a4811f81a0",
        "43bffa19d6ef651451f6841e019931e96f6f2eaeff28ab4b6f1ce4576de2717b",
        "63ecefee431d49523bc1136c4e72d0206a477086caf606cd5cf025686c109cf8",
        "ec4ffa1ad0ac88081314b7aaffef6a91f97cd57291bb5ba686c5415f5eb4f6b6",
        "92c063d67e8f14db71d288e11266ffdaf69cbf115f7bab14f3bc2f68d25e0e30",
    }
)

_SEQUENCE_TAG = 0x30
_VERSION_TAG = 0xA0
# Fields of tbsCertificate before the SubjectPublicKeyInfo: serialNumber,
# signature, issuer, validity, subject.
_FIELDS_BEFORE_SPKI = 5

_warned_unsupported = False


def _read_element(data: bytes, offset: int) -> tuple[int, int, int]:
    """Return ``(tag, content_start, content_end)`` of the DER element at ``offset``."""
    tag = data[offset]
    length = data[offset + 1]
    start = offset + 2
    if length & 0x80:
        count = length & 0x7F
        length = int.from_bytes(data[start : start + count], "big")
        start += count
    return tag, start, start + length


def spki_sha256(der_certificate: bytes) -> str:
    """SHA-256 (hex) of the DER-encoded SubjectPublicKeyInfo of a certificate."""
    _, start, _ = _read_element(der_certificate, 0)
    _, offset, _ = _read_element(der_certificate, start)  # tbsCertificate
    tag, _, end = _read_element(der_certificate, offset)
    if tag == _VERSION_TAG:
        offset = end
    for _ in range(_FIELDS_BEFORE_SPKI):
        _, _, offset = _read_element(der_certificate, offset)
    tag, _, end = _read_element(der_certificate, offset)
    if tag != _SEQUENCE_TAG:
        raise ValueError("Unexpected certificate structure")
    return hashlib.sha256(der_certificate[offset:end]).hexdigest()


def contains_distrusted(
    chain: Iterable[bytes], distrusted: frozenset[str] = DISTRUSTED_SPKI_SHA256
) -> bool:
    """Whether any DER-encoded certificate in ``chain`` is distrusted."""
    return any(spki_sha256(certificate) in distrusted for certificate in chain)


def _verified_chain(tls: Any) -> list[bytes] | None:
    """DER-encoded verified chain of a handshaken ``SSLSocket``/``SSLObject``, or ``None``."""
    try:
        if hasattr(tls, "get_verified_chain"):
            certificates = tls.get_verified_chain()
        else:
            certificates = tls._sslobj.get_verified_chain()
    except (AttributeError, ValueError):
        return None
    return [
        c if isinstance(c, bytes) else ssl.PEM_cert_to_DER_cert(c.public_bytes())
        for c in certificates
    ]


def _check(tls: Any) -> None:
    global _warned_unsupported
    chain = _verified_chain(tls)
    if chain is None:
        if not _warned_unsupported:
            _warned_unsupported = True
            get_logger("distrusted_intermediates").warning(
                "This Python runtime does not expose the verified certificate chain; "
                "distrusted intermediates are not being checked"
            )
        return
    if contains_distrusted(chain):
        raise ssl.SSLCertVerificationError(
            "Certificate chain contains a distrusted intermediate certificate (CVE-2026-111943)"
        )


class _DistrustingSSLObject(ssl.SSLObject):
    def do_handshake(self) -> None:
        super().do_handshake()
        _check(self)


class _DistrustingSSLSocket(ssl.SSLSocket):
    def do_handshake(self, block: bool = False) -> None:
        super().do_handshake(block)
        _check(self)


def distrusting_ssl_context(context: ssl.SSLContext) -> ssl.SSLContext:
    """Make ``context`` fail any handshake whose chain contains a distrusted intermediate.

    Covers both the socket and the memory-BIO paths, so it works for the
    blocking, asyncio and anyio-based clients. Returns ``context``.
    """
    context.sslsocket_class = _DistrustingSSLSocket
    context.sslobject_class = _DistrustingSSLObject
    return context
