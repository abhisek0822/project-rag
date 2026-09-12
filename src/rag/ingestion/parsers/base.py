"""Parser protocol and helpers shared by text-based formats."""

from __future__ import annotations

from typing import Protocol

from ..models import Block


class Parser(Protocol):
    def parse(self, data: bytes, filename: str) -> list[Block]:
        """Parse raw file bytes into ordered structural blocks."""


def decode_text(data: bytes) -> str:
    """Decode common text encodings deterministically.

    UTF encodings are attempted before Windows-1252, whose mapping covers all
    byte values. This intentionally avoids a mandatory charset detector.
    """

    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    # cp1252 should always succeed; this is only a defensive final fallback.
    return data.decode("utf-8", errors="replace")
