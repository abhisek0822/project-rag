"""HTML parser built on the standard library with no runtime dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser as StandardHTMLParser

from ..errors import EmptyDocumentError
from ..models import Block
from .base import decode_text


@dataclass
class _Capture:
    tag: str
    kind: str
    parts: list[str] = field(default_factory=list)


class _BlockHTMLParser(StandardHTMLParser):
    _ignored_tags = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self._capture: _Capture | None = None
        self._ignored_depth = 0
        self._headings: dict[int, str] = {}
        self._section_path: tuple[str, ...] = ()
        self._element_index = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self._ignored_tags:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag == "br" and self._capture is not None:
            self._capture.parts.append("\n")
            return
        if tag in {"td", "th"} and self._capture is not None and self._capture.tag == "tr":
            if self._capture.parts:
                self._capture.parts.append(" | ")
            return
        if self._capture is not None:
            return

        kind: str | None = None
        if tag in {"p", "blockquote"}:
            kind = "paragraph"
        elif tag == "li":
            kind = "list"
        elif tag == "tr":
            kind = "table"
        elif tag in {"pre", "code"}:
            kind = "code"
        elif len(tag) == 2 and tag.startswith("h") and tag[1].isdigit():
            kind = "heading"

        if kind is not None:
            self._capture = _Capture(tag=tag, kind=kind)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._ignored_tags:
            if self._ignored_depth:
                self._ignored_depth -= 1
            return
        if self._ignored_depth or self._capture is None or self._capture.tag != tag:
            return

        capture = self._capture
        self._capture = None
        text = "".join(capture.parts).strip()
        if not text:
            return

        if capture.kind == "heading":
            level = int(capture.tag[1])
            self._headings[level] = " ".join(text.split())
            for stale_level in [key for key in self._headings if key > level]:
                del self._headings[stale_level]
            self._section_path = tuple(self._headings[key] for key in sorted(self._headings))
            text = self._headings[level]
        elif capture.kind != "code":
            text = " ".join(text.split()) if capture.kind == "paragraph" else text

        self.blocks.append(
            Block(
                kind=capture.kind,
                text=text,
                order=len(self.blocks),
                section_path=self._section_path,
                source_locator={"tag": tag, "element_index": self._element_index},
            )
        )
        self._element_index += 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and self._capture is not None:
            self._capture.parts.append(data)


class HTMLParser:
    def parse(self, data: bytes, filename: str) -> list[Block]:
        parser = _BlockHTMLParser()
        parser.feed(decode_text(data))
        parser.close()
        if not parser.blocks:
            raise EmptyDocumentError(filename)
        return parser.blocks
