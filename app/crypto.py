"""Encryption at rest for WalletLedger. One small module, explained in docs/SECURITY.md.

What it does:

* Turns ``SECRET_KEY`` into separate keys per purpose with HKDF-SHA256 and a context label, so
  the key that protects chat history is not the key that protects AI keys.
* Encrypts with Fernet (AES-128-CBC + HMAC-SHA256, random IV). Tampering is detected.
* Key rotation: put previous secrets in ``SECRET_KEY_OLD`` (comma-separated). New data is always
  written with ``SECRET_KEY``; old data is read with whichever secret still works.
  ``scripts/protect_data.py rotate`` re-encrypts everything with the newest key.
* Stored text values look like ``enc1:<fernet token>`` so they are self-describing. Text without
  that prefix is legacy plaintext and is returned unchanged, so old rows keep working.

What it does NOT do: hide data from whoever runs the server. The server holds ``SECRET_KEY``
because it must read your data to answer you.
"""

from __future__ import annotations

import base64
import hashlib
import logging
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken, MultiFernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from sqlalchemy.types import Text, TypeDecorator

from app.config import INSECURE_DEV_KEY, settings

logger = logging.getLogger(__name__)

PREFIX = "enc1:"
# Every Fernet token starts with version byte 0x80 and a 64-bit timestamp -> "gAAAAA" in base64.
_TOKEN_START = "gAAAAA"
FIELD_CONTEXT = b"walletledger/field/v1"
VAULT_CONTEXT = b"walletledger/api-key-vault/v1"
_HKDF_SALT = b"walletledger/hkdf-salt/v1"
UNREADABLE = "[unreadable: encrypted with a different key]"

# Values that are public (in the repo) and therefore protect nothing.
_PUBLIC_SECRETS = {INSECURE_DEV_KEY, "change-me-to-a-long-random-string"}

_warned_unreadable = False


# --------------------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------------------


def all_secrets() -> list[str]:
    """SECRET_KEY first, then every SECRET_KEY_OLD entry (duplicates and blanks removed)."""
    out: list[str] = []
    for value in [settings.secret_key, *settings.secret_key_old.split(",")]:
        value = value.strip()
        if value and value not in out:
            out.append(value)
    return out


def weak_secret_key(secret: str | None = None) -> bool:
    """True if SECRET_KEY is a public placeholder or too short to be a real random secret."""
    secret = settings.secret_key if secret is None else secret
    return secret.strip() in _PUBLIC_SECRETS or len(secret.strip()) < 24


def derive_key(secret: str, context: bytes) -> bytes:
    """A Fernet key for one purpose (``context``) derived from one secret with HKDF-SHA256."""
    raw = HKDF(algorithm=hashes.SHA256(), length=32, salt=_HKDF_SALT, info=context).derive(
        secret.encode("utf-8")
    )
    return base64.urlsafe_b64encode(raw)


@lru_cache(maxsize=16)
def _multi(context: bytes, keys: tuple[str, ...]) -> MultiFernet:
    return MultiFernet([Fernet(derive_key(s, context)) for s in keys])


@lru_cache(maxsize=16)
def _legacy_vault(keys: tuple[str, ...]) -> MultiFernet:
    return MultiFernet(
        [Fernet(base64.urlsafe_b64encode(hashlib.sha256(s.encode("utf-8")).digest())) for s in keys]
    )


def multi_fernet(context: bytes) -> MultiFernet:
    """Encrypts with the newest secret, decrypts with any of them."""
    return _multi(context, tuple(all_secrets()))


def newest_fernet(context: bytes) -> Fernet:
    """Only the newest secret. Used to tell whether a value still needs rotating."""
    return Fernet(derive_key(all_secrets()[0], context))


# --------------------------------------------------------------------------------------
# Text fields (chat history, transaction notes)
# --------------------------------------------------------------------------------------


def is_encrypted(value: str | None) -> bool:
    return isinstance(value, str) and value.startswith(PREFIX + _TOKEN_START)


def encrypt_text(value: str) -> str:
    token = multi_fernet(FIELD_CONTEXT).encrypt(value.encode("utf-8"))
    return PREFIX + token.decode("ascii")


def decrypt_text_strict(value: str) -> str:
    """Like ``decrypt_text`` but raises ``InvalidToken`` when no configured key works."""
    if not is_encrypted(value):
        return value
    return multi_fernet(FIELD_CONTEXT).decrypt(value[len(PREFIX) :].encode("ascii")).decode("utf-8")


def decrypt_text(value: str) -> str:
    """Plaintext for an ``enc1:`` value; legacy plaintext is returned unchanged.

    If no configured key can read it (SECRET_KEY changed without SECRET_KEY_OLD), a placeholder
    is returned instead of crashing the bot, and a warning is logged once.
    """
    global _warned_unreadable
    try:
        return decrypt_text_strict(value)
    except InvalidToken, ValueError:
        if not _warned_unreadable:
            _warned_unreadable = True
            logger.warning(
                "Some stored data cannot be decrypted with SECRET_KEY / SECRET_KEY_OLD. "
                "Was SECRET_KEY changed? Put the previous value in SECRET_KEY_OLD."
            )
        return UNREADABLE


def needs_rotation(value: str) -> bool:
    """True for an ``enc1:`` value that the newest key cannot read (written with an older one)."""
    if not is_encrypted(value):
        return False
    try:
        newest_fernet(FIELD_CONTEXT).decrypt(value[len(PREFIX) :].encode("ascii"))
        return False
    except InvalidToken:
        return True


def rotate_text(value: str) -> str:
    """Re-encrypt an ``enc1:`` value with the newest key. Raises ``InvalidToken`` if unreadable."""
    token = multi_fernet(FIELD_CONTEXT).rotate(value[len(PREFIX) :].encode("ascii"))
    return PREFIX + token.decode("ascii")


class EncryptedText(TypeDecorator):
    """A Text column that is encrypted in the database and plaintext in Python.

    Do not use it in WHERE / ORDER BY / GROUP BY / LIKE: the database only sees ciphertext, and
    every encryption of the same text is different (random IV). Filter in Python instead.
    """

    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return encrypt_text(str(value))

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return decrypt_text(value)


# --------------------------------------------------------------------------------------
# API-key vault (api_keys.encrypted_key). Plain Fernet tokens, no prefix (historic format).
# --------------------------------------------------------------------------------------


def vault_fernet() -> MultiFernet:
    return multi_fernet(VAULT_CONTEXT)


def legacy_vault_fernet() -> MultiFernet:
    """The original vault cipher, ``Fernet(sha256(SECRET_KEY))``. Read-only fallback."""
    return _legacy_vault(tuple(all_secrets()))


def encrypt_vault(plaintext: str) -> str:
    return vault_fernet().encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_vault(token: str) -> str:
    """New HKDF format first, then the legacy sha256 format. Raises ``InvalidToken``."""
    data = token.encode("utf-8")
    try:
        return vault_fernet().decrypt(data).decode("utf-8")
    except InvalidToken:
        return legacy_vault_fernet().decrypt(data).decode("utf-8")


def vault_needs_rotation(token: str) -> bool:
    """True if the newest vault key cannot read ``token`` (older secret or legacy format)."""
    try:
        newest_fernet(VAULT_CONTEXT).decrypt(token.encode("utf-8"))
        return False
    except InvalidToken:
        return True
