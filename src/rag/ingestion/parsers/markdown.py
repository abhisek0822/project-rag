"""A small structure-preserving Markdown parser.

It is intentionally not a renderer: ingestion needs block boundaries and
source lines, not HTML output. The scanner recognizes headings, fenced code,
lists, pipe tables, and prose while leaving inline Markdown intact.
"""

from __future__ import annotations

import re

from ..errors import EmptyDocumentError
from ..models import Block
from .base import decode_text

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```+|~~~+)(.*)$")
_LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_TABLE_DIVIDER_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")


class MarkdownParser:
    def parse(self, data: bytes, filename: str) -> list[Block]:
        text = decode_text(data).replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            raise EmptyDocumentError(filename)

        lines = text.split("\n")
        blocks: list[Block] = []
        headings: dict[int, str] = {}
        section_path: tuple[str, ...] = ()
        index = 0
        order = 0

        def add(kind: str, value: str, start: int, end: int) -> None:
            nonlocal order
            if not value.strip():
                return
            blocks.append(
                Block(
                    kind=kind,
                    text=value.strip("\n"),
                    order=order,
                    section_path=section_path,
                    source_locator={"line_start": start + 1, "line_end": end},
                )
            )
            order += 1

        while index < len(lines):
            line = lines[index]
            if not line.strip():
                index += 1
                continue

            heading_match = _HEADING_RE.match(line)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                headings[level] = title
                for stale_level in [key for key in headings if key > level]:
                    del headings[stale_level]
                section_path = tuple(headings[key] for key in sorted(headings))
                add("heading", title, index, index + 1)
                index += 1
                continue

            fence_match = _FENCE_RE.match(line)
            if fence_match:
                start = index
                marker = fence_match.group(1)[0]
                minimum = len(fence_match.group(1))
                index += 1
                code_lines: list[str] = []
                while index < len(lines):
                    closing = lines[index].lstrip()
                    if closing.startswith(marker * minimum):
                        index += 1
                        break
                    code_lines.append(lines[index])
                    index += 1
                add("code", "\n".join(code_lines), start, index)
                continue

            if _LIST_RE.match(line):
                start = index
                list_lines: list[str] = []
                while index < len(lines):
                    candidate = lines[index]
                    if _LIST_RE.match(candidate) or (
                        candidate.strip() and candidate.startswith(("  ", "\t"))
                    ):
                        list_lines.append(candidate.rstrip())
                        index += 1
                    else:
                        break
                add("list", "\n".join(list_lines), start, index)
                continue

            if "|" in line and index + 1 < len(lines) and _TABLE_DIVIDER_RE.match(lines[index + 1]):
                start = index
                table_lines = [line.rstrip(), lines[index + 1].rstrip()]
                index += 2
                while index < len(lines) and "|" in lines[index] and lines[index].strip():
                    table_lines.append(lines[index].rstrip())
                    index += 1
                add("table", "\n".join(table_lines), start, index)
                continue

            start = index
            prose: list[str] = []
            while index < len(lines) and lines[index].strip():
                candidate = lines[index]
                if index > start and (
                    _HEADING_RE.match(candidate)
                    or _FENCE_RE.match(candidate)
                    or _LIST_RE.match(candidate)
                    or (
                        "|" in candidate
                        and index + 1 < len(lines)
                        and _TABLE_DIVIDER_RE.match(lines[index + 1])
                    )
                ):
                    break
                prose.append(candidate.rstrip())
                index += 1
            add("paragraph", "\n".join(prose), start, index)

        if not blocks:
            raise EmptyDocumentError(filename)
        return blocks
