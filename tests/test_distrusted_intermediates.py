"""Tests for the distrusted-intermediates TLS check."""

import asyncio
import socket
import ssl
import threading
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from tac.context.memory import MemoryClient
from tac.utils import distrusted_intermediates
from tac.utils.distrusted_intermediates import (
    DISTRUSTED_SPKI_SHA256,
    contains_distrusted,
    distrusting_ssl_context,
    spki_sha256,
)

DATA = Path(__file__).parent / "data"
KNOWN_SPKI = "ebed6475c2c2d80fcb199f3fe8dd6e6b3ce60b77eb15d4e42c3dec0c6f5b5105"
LOCALHOST_CERT = DATA / "localhost-cert.pem"
LOCALHOST_KEY = DATA / "localhost-key.pem"


def _der(path: Path) -> bytes:
    return ssl.PEM_cert_to_DER_cert(path.read_text())


def _client_context() -> ssl.SSLContext:
    context = ssl.create_default_context(cafile=str(LOCALHOST_CERT))
    return distrusting_ssl_context(context)


@pytest.fixture
def distrust_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    localhost_spki = spki_sha256(_der(LOCALHOST_CERT))
    monkeypatch.setattr(
        distrusted_intermediates,
        "contains_distrusted",
        lambda chain: localhost_spki in {spki_sha256(c) for c in chain},
    )


@pytest.fixture
def tls_server() -> Iterator[int]:
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(LOCALHOST_CERT, LOCALHOST_KEY)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    stop = threading.Event()

    def serve() -> None:
        listener.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except (TimeoutError, OSError):
                continue
            try:
                with server_context.wrap_socket(conn, server_side=True) as tls:
                    tls.recv(1)
            except OSError:
                pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    yield listener.getsockname()[1]
    stop.set()
    thread.join()
    listener.close()


class TestSpkiSha256:
    def test_hashes_known_certificate(self) -> None:
        assert spki_sha256(_der(DATA / "distrusted-intermediate.pem")) == KNOWN_SPKI

    def test_lists_eight_distinct_hashes(self) -> None:
        assert len(DISTRUSTED_SPKI_SHA256) == 8
        assert KNOWN_SPKI in DISTRUSTED_SPKI_SHA256

    def test_rejects_non_certificate(self) -> None:
        with pytest.raises((ValueError, IndexError)):
            spki_sha256(b"\x30\x03\x02\x01\x00")


class TestContainsDistrusted:
    def test_detects_distrusted_certificate_anywhere_in_chain(self) -> None:
        leaf = _der(LOCALHOST_CERT)
        intermediate = _der(DATA / "distrusted-intermediate.pem")
        assert contains_distrusted([leaf, intermediate, leaf]) is True

    def test_allows_other_chains(self) -> None:
        assert contains_distrusted([_der(LOCALHOST_CERT)]) is False
        assert contains_distrusted([]) is False


class TestHandshake:
    def test_socket_connection_allowed(self, tls_server: int) -> None:
        with socket.create_connection(("127.0.0.1", tls_server)) as sock:
            with _client_context().wrap_socket(sock, server_hostname="localhost") as tls:
                assert tls.version() is not None

    def test_socket_connection_with_distrusted_chain_refused(
        self, tls_server: int, distrust_localhost: None
    ) -> None:
        with socket.create_connection(("127.0.0.1", tls_server)) as sock:
            with pytest.raises(ssl.SSLCertVerificationError, match="distrusted"):
                _client_context().wrap_socket(sock, server_hostname="localhost")

    @pytest.mark.asyncio
    async def test_asyncio_connection_allowed(self, tls_server: int) -> None:
        _, writer = await asyncio.open_connection(
            "127.0.0.1", tls_server, ssl=_client_context(), server_hostname="localhost"
        )
        writer.close()

    @pytest.mark.asyncio
    async def test_asyncio_connection_with_distrusted_chain_refused(
        self, tls_server: int, distrust_localhost: None
    ) -> None:
        with pytest.raises(ssl.SSLCertVerificationError, match="distrusted"):
            await asyncio.open_connection(
                "127.0.0.1", tls_server, ssl=_client_context(), server_hostname="localhost"
            )

    @pytest.mark.asyncio
    async def test_httpx_request_with_distrusted_chain_refused(
        self, tls_server: int, distrust_localhost: None
    ) -> None:
        async with httpx.AsyncClient(verify=_client_context()) as client:
            with pytest.raises(httpx.ConnectError, match="distrusted"):
                await client.get(f"https://localhost:{tls_server}/")

    def test_unsupported_runtime_skips_check_and_warns_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(distrusted_intermediates, "_verified_chain", lambda tls: None)
        monkeypatch.setattr(distrusted_intermediates, "_warned_unsupported", False)
        distrusted_intermediates._check(object())
        assert distrusted_intermediates._warned_unsupported is True


class TestClientsUseDistrustingContext:
    def test_base_clients_install_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[ssl.SSLContext] = []

        def spy(context: ssl.SSLContext) -> ssl.SSLContext:
            calls.append(context)
            return distrusting_ssl_context(context)

        monkeypatch.setattr("tac.context.base.distrusting_ssl_context", spy)
        client = MemoryClient(store_id="s", api_key="k", api_secret="v")

        with client._get_sync_client():
            pass
        assert len(calls) == 1
        assert calls[0].sslobject_class is distrusted_intermediates._DistrustingSSLObject

    @pytest.mark.asyncio
    async def test_async_base_client_installs_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[ssl.SSLContext] = []

        def spy(context: ssl.SSLContext) -> ssl.SSLContext:
            calls.append(context)
            return distrusting_ssl_context(context)

        monkeypatch.setattr("tac.context.base.distrusting_ssl_context", spy)
        client = MemoryClient(store_id="s", api_key="k", api_secret="v")

        async with client._get_client():
            pass
        assert len(calls) == 1
