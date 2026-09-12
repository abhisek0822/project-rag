"""Extension-based parser selection."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ..errors import UnsupportedFileTypeError
from .base import Parser
from .docx import DocxParser
from .html import HTMLParser
from .markdown import MarkdownParser
from .pdf import PDFParser
from .text import TextParser

SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".markdown", ".html", ".htm", ".docx", ".pdf"})
SUPPORTED_FILE_EXTENSIONS = SUPPORTED_EXTENSIONS
SUPPORTED_FILE_TYPES = SUPPORTED_EXTENSIONS


class ParserRegistry:
    """Select parsers by the source filename's final extension."""

    def __init__(self, parsers: Mapping[str, Parser] | None = None) -> None:
        defaults: dict[str, Parser] = {
            ".txt": TextParser(),
            ".md": MarkdownParser(),
            ".markdown": MarkdownParser(),
            ".html": HTMLParser(),
            ".htm": HTMLParser(),
            ".docx": DocxParser(),
            ".pdf": PDFParser(),
        }
        if parsers:
            defaults.update(
                {
                    (extension if extension.startswith(".") else f".{extension}").lower(): parser
                    for extension, parser in parsers.items()
                }
            )
        self._parsers = defaults

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset(self._parsers)

    def for_filename(self, filename: str) -> Parser:
        extension = Path(filename).suffix.lower()
        try:
            return self._parsers[extension]
        except KeyError as exc:
            raise UnsupportedFileTypeError(filename) from exc

    def parse(self, data: bytes, filename: str):
        return self.for_filename(filename).parse(data, filename)


def parser_for(filename: str) -> Parser:
    return ParserRegistry().for_filename(filename)


def parse_document(data: bytes, filename: str):
    return ParserRegistry().parse(data, filename)
