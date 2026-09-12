"""Parser for plain-text documents."""

from __future__ import annotations

import re

from ..errors import EmptyDocumentError
from ..models import Block
from .base import decode_text

_LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return False
    letters = [char for char in stripped if char.isalpha()]
    return stripped.endswith(":") or (len(letters) >= 3 and all(char.isupper() for char in letters))


class TextParser:
    def parse(self, data: bytes, filename: str) -> list[Block]:
        text = decode_text(data).replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            raise EmptyDocumentError(filename)

        lines = text.split("\n")
        blocks: list[Block] = []
        section_path: tuple[str, ...] = ()
        index = 0
        order = 0

        while index < len(lines):
            if not lines[index].strip():
                index += 1
                continue

            start = index
            line = lines[index]
            if _looks_like_heading(line):
                heading = line.strip().rstrip(":").strip()
                section_path = (heading,)
                blocks.append(
                    Block(
                        kind="heading",
                        text=heading,
                        order=order,
                        section_path=section_path,
                        source_locator={"line_start": index + 1, "line_end": index + 1},
                    )
                )
                order += 1
                index += 1
                continue

            if _LIST_RE.match(line):
                gathered: list[str] = []
                while index < len(lines) and (
                    _LIST_RE.match(lines[index]) or lines[index].startswith(("  ", "\t"))
                ):
                    if lines[index].strip():
                        gathered.append(lines[index].rstrip())
                    index += 1
                blocks.append(
                    Block(
                        kind="list",
                        text="\n".join(gathered),
                        order=order,
                        section_path=section_path,
                        source_locator={"line_start": start + 1, "line_end": index},
                    )
                )
                order += 1
                continue

            gathered = []
            while index < len(lines) and lines[index].strip():
                if index > start and (
                    _LIST_RE.match(lines[index]) or _looks_like_heading(lines[index])
                ):
                    break
                gathered.append(lines[index].rstrip())
                index += 1
            blocks.append(
                Block(
                    kind="paragraph",
                    text="\n".join(gathered),
                    order=order,
                    section_path=section_path,
                    source_locator={"line_start": start + 1, "line_end": index},
                )
            )
            order += 1

        if not blocks:
            raise EmptyDocumentError(filename)
        return blocks
