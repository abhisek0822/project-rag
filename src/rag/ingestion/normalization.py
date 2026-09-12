"""Conservative text cleanup that preserves source structure and casing."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import replace

from .models import Block

_PROSE_SPACE_RE = re.compile(r"[ \t\f\v]+")
_LINE_WRAP_HYPHEN_RE = re.compile(r"(?<=[A-Za-z])-[ \t]*\n[ \t]*(?=[a-z])")


def _strip_control_characters(text: str) -> str:
    return "".join(
        char for char in text if char in {"\n", "\t"} or unicodedata.category(char) != "Cc"
    )


def normalize_text(text: str, *, preserve_layout: bool = False) -> str:
    """Normalize one block without flattening meaningful document structure."""

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _strip_control_characters(text)

    if preserve_layout:
        return "\n".join(line.rstrip() for line in text.split("\n")).strip("\n")

    text = _LINE_WRAP_HYPHEN_RE.sub("", text)
    lines = [_PROSE_SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    # Blank lines remain paragraph separators. A single line break inside a
    # paragraph is normally a PDF/text line wrap and becomes a space.
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if line:
            current.append(line)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs).strip()


class Normalizer:
    """Normalize blocks and suppress repeated PDF-style headers/footers."""

    def __init__(self, *, remove_repeated_margins: bool = True) -> None:
        self.remove_repeated_margins = remove_repeated_margins

    def normalize(self, blocks: Sequence[Block]) -> list[Block]:
        normalized: list[Block] = []
        for block in sorted(blocks, key=lambda item: item.order):
            preserve_layout = block.kind in {"list", "table", "code"}
            text = normalize_text(block.text, preserve_layout=preserve_layout)
            if text:
                normalized.append(replace(block, text=text))

        if self.remove_repeated_margins:
            normalized = self._remove_repeated_page_margins(normalized)
        return normalized

    @staticmethod
    def _remove_repeated_page_margins(blocks: Sequence[Block]) -> list[Block]:
        by_page: defaultdict[int, list[Block]] = defaultdict(list)
        for block in blocks:
            if block.page is not None:
                by_page[block.page].append(block)

        if len(by_page) < 3:
            return list(blocks)

        edge_counts: Counter[str] = Counter()
        page_edges: dict[int, set[str]] = {}
        for page, page_blocks in by_page.items():
            ordered = sorted(page_blocks, key=lambda item: item.order)
            candidates: set[str] = set()
            for edge in (ordered[0], ordered[-1]):
                if "\n" not in edge.text and 0 < len(edge.text) <= 120:
                    candidates.add(edge.text)
            page_edges[page] = candidates
            edge_counts.update(candidates)

        threshold = max(3, (len(by_page) * 3 + 4) // 5)  # ceil(60%)
        repeated = {text for text, count in edge_counts.items() if count >= threshold}
        if not repeated:
            return list(blocks)

        return [
            block
            for block in blocks
            if not (
                block.page is not None
                and block.text in repeated
                and block.text in page_edges.get(block.page, set())
            )
        ]
