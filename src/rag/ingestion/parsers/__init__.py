"""Document parser adapters supported by the MVP ingestion pipeline."""

from .base import Parser, decode_text
from .docx import DocxParser
from .html import HTMLParser
from .markdown import MarkdownParser
from .pdf import PDFParser
from .registry import (
    SUPPORTED_EXTENSIONS,
    SUPPORTED_FILE_EXTENSIONS,
    SUPPORTED_FILE_TYPES,
    ParserRegistry,
    parse_document,
    parser_for,
)
from .text import TextParser

__all__ = [
    "DocxParser",
    "HTMLParser",
    "MarkdownParser",
    "PDFParser",
    "Parser",
    "ParserRegistry",
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_FILE_EXTENSIONS",
    "SUPPORTED_FILE_TYPES",
    "TextParser",
    "decode_text",
    "parse_document",
    "parser_for",
]
