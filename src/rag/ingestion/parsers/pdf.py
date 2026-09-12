"""Text-layer PDF parser using PyMuPDF."""

from __future__ import annotations

from ..errors import DocumentParseError, NeedsOCRError, ParserDependencyError
from ..models import Block


class PDFParser:
    def parse(self, data: bytes, filename: str) -> list[Block]:
        try:
            import fitz  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ParserDependencyError("PDF parsing requires the 'PyMuPDF' package") from exc

        try:
            document = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise DocumentParseError(f"Could not open PDF file {filename}: {exc}") from exc

        blocks: list[Block] = []
        try:
            for page_index, page in enumerate(document):
                raw_blocks = page.get_text("blocks", sort=True)
                for raw in raw_blocks:
                    # PyMuPDF returns x0, y0, x1, y1, text, block_no,
                    # block_type. Images use a non-zero block_type.
                    if len(raw) >= 7 and raw[6] != 0:
                        continue
                    text = str(raw[4]).strip() if len(raw) >= 5 else ""
                    if not text:
                        continue
                    block_number = int(raw[5]) if len(raw) >= 6 else len(blocks)
                    blocks.append(
                        Block(
                            kind="paragraph",
                            text=text,
                            order=len(blocks),
                            page=page_index + 1,
                            source_locator={
                                "page": page_index + 1,
                                "block": block_number,
                                "bbox": [float(value) for value in raw[:4]],
                            },
                        )
                    )
        except Exception as exc:
            raise DocumentParseError(f"Could not extract text from {filename}: {exc}") from exc
        finally:
            document.close()

        usable_characters = sum(sum(char.isalnum() for char in block.text) for block in blocks)
        if usable_characters < 10:
            raise NeedsOCRError(filename)
        return blocks
