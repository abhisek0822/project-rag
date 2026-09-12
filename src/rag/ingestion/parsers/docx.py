"""DOCX parser preserving paragraph/table order and heading breadcrumbs."""

from __future__ import annotations

import re
from collections.abc import Iterator
from io import BytesIO

from ..errors import DocumentParseError, EmptyDocumentError, ParserDependencyError
from ..models import Block


class DocxParser:
    def parse(self, data: bytes, filename: str) -> list[Block]:
        try:
            from docx import Document  # type: ignore[import-not-found]
            from docx.oxml.table import CT_Tbl  # type: ignore[import-not-found]
            from docx.oxml.text.paragraph import CT_P  # type: ignore[import-not-found]
            from docx.table import Table  # type: ignore[import-not-found]
            from docx.text.paragraph import Paragraph  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ParserDependencyError("DOCX parsing requires the 'python-docx' package") from exc

        try:
            document = Document(BytesIO(data))
        except Exception as exc:
            raise DocumentParseError(f"Could not open DOCX file {filename}: {exc}") from exc

        def iter_items() -> Iterator[Paragraph | Table]:
            for child in document.element.body.iterchildren():
                if isinstance(child, CT_P):
                    yield Paragraph(child, document)
                elif isinstance(child, CT_Tbl):
                    yield Table(child, document)

        blocks: list[Block] = []
        headings: dict[int, str] = {}
        section_path: tuple[str, ...] = ()
        source_index = 0

        for item in iter_items():
            if isinstance(item, Paragraph):
                text = item.text.strip()
                if not text:
                    source_index += 1
                    continue
                style_name = item.style.name if item.style is not None else ""
                heading_match = re.match(r"Heading\s+(\d+)", style_name, re.IGNORECASE)
                if heading_match:
                    level = int(heading_match.group(1))
                    headings[level] = text
                    for stale_level in [key for key in headings if key > level]:
                        del headings[stale_level]
                    section_path = tuple(headings[key] for key in sorted(headings))
                    kind = "heading"
                elif style_name.lower().startswith("list"):
                    kind = "list"
                else:
                    kind = "paragraph"
                blocks.append(
                    Block(
                        kind=kind,
                        text=text,
                        order=len(blocks),
                        section_path=section_path,
                        source_locator={
                            "body_index": source_index,
                            "paragraph_style": style_name,
                        },
                    )
                )
            else:
                rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in item.rows]
                table_text = "\n".join(row for row in rows if row.strip(" |"))
                if table_text:
                    blocks.append(
                        Block(
                            kind="table",
                            text=table_text,
                            order=len(blocks),
                            section_path=section_path,
                            source_locator={"body_index": source_index, "type": "table"},
                        )
                    )
            source_index += 1

        if not blocks:
            raise EmptyDocumentError(filename)
        return blocks
