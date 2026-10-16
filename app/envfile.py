"""Small, safe .env editor used by the setup wizard.

Keeps comments and ordering, replaces a value in place (or un-comments a `# KEY=` template line),
appends new keys at the end, and copies the file to `.env.bak` before each change (kept private: 0600).
"""

from __future__ import annotations

import contextlib
import re
import shutil
from pathlib import Path

# Public placeholder secrets: they are in the repository, so they protect nothing.
INSECURE_DEV_KEY = "walletledger-insecure-dev-key-change-in-prod"
PLACEHOLDER_SECRETS = {INSECURE_DEV_KEY, "change-me-to-a-long-random-string"}


def is_weak_secret(value: str | None) -> bool:
    """True for an empty, placeholder or too-short (under 24 characters) SECRET_KEY."""
    text = (value or "").strip()
    return not text or text in PLACEHOLDER_SECRETS or len(text) < 24


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


_LINE = re.compile(r"^\s*(#\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


def _value(raw: str) -> str:
    """The value part of a line: quotes removed; for unquoted values a trailing ` # comment` is dropped."""
    text = raw.strip()
    if text[:1] in ('"', "'") and text.count(text[0]) >= 2:
        return text[1 : text.index(text[0], 1)]
    return re.split(r"\s+#", text, maxsplit=1)[0].strip()


def _private(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.chmod(0o600)  # secrets: owner only (no effect on Windows, harmless)


def read_env(path: Path) -> dict[str, str]:
    """Active (non-commented) KEY=VALUE pairs."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = _LINE.match(raw)
        if m and not m.group(1):
            values[m.group(2)] = _value(m.group(3))
    return values


def _format(value: str) -> str:
    return f'"{value}"' if re.search(r"\s|#", value) else value


def set_values(path: Path, updates: dict[str, str], *, backup: bool = True) -> list[str]:
    """Apply updates; returns the keys that actually changed."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    current = read_env(path)
    changed = [k for k, v in updates.items() if current.get(k) != v]
    if not changed:
        return []
    if backup and path.exists():
        bak = path.with_name(path.name + ".bak")
        shutil.copyfile(path, bak)
        _private(bak)

    pending = {k: updates[k] for k in changed}
    out: list[str] = []
    for raw in lines:
        m = _LINE.match(raw)
        if m and m.group(2) in pending:
            key = m.group(2)  # replaces the active line, or revives a `# KEY=` template line
            out.append(f"{key}={_format(pending.pop(key))}")
        else:
            out.append(raw)
    # Duplicate active lines for the same key would shadow our value: drop later duplicates.
    seen: set[str] = set()
    deduped = []
    for raw in out:
        m = _LINE.match(raw)
        if m and not m.group(1) and m.group(2) in updates:
            if m.group(2) in seen:
                continue
            seen.add(m.group(2))
        deduped.append(raw)
    out = deduped
    if pending:
        if out and out[-1].strip():
            out.append("")
        out.extend(f"{k}={_format(v)}" for k, v in pending.items())
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    _private(path)
    return changed
