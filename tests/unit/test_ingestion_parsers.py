from __future__ import annotations

import pytest

from rag.ingestion import (
    Block,
    HTMLParser,
    MarkdownParser,
    Normalizer,
    ParserRegistry,
    TextParser,
    UnsupportedFileTypeError,
    normalize_text,
)


def test_markdown_parser_preserves_structure_and_breadcrumbs() -> None:
    source = b"""# Handbook

Intro text.

## Leave

- Casual leave: 12 days
- Sick leave: 10 days

```python
days = 12
```

| Type | Days |
| --- | --- |
| Casual | 12 |
"""

    blocks = MarkdownParser().parse(source, "handbook.md")

    assert [block.kind for block in blocks] == [
        "heading",
        "paragraph",
        "heading",
        "list",
        "code",
        "table",
    ]
    assert blocks[3].section_path == ("Handbook", "Leave")
    assert blocks[3].source_locator == {"line_start": 7, "line_end": 8}
    assert blocks[4].text == "days = 12"


def test_html_parser_ignores_scripts_and_tracks_heading() -> None:
    source = (
        b"<html><style>.secret{}</style><h1>Policy</h1>"
        b"<p>Refunds are available <strong>within 30 days</strong>.</p>"
        b"<script>ignore me</script></html>"
    )

    blocks = HTMLParser().parse(source, "policy.html")

    assert [(block.kind, block.text) for block in blocks] == [
        ("heading", "Policy"),
        ("paragraph", "Refunds are available within 30 days."),
    ]
    assert blocks[1].section_path == ("Policy",)


def test_plain_text_parser_decodes_windows_text_and_lists() -> None:
    source = "BENEFITS\n\nCaf\xe9 allowance.\n\n- Dental\n- Vision".encode("cp1252")

    blocks = TextParser().parse(source, "benefits.txt")

    assert [block.kind for block in blocks] == ["heading", "paragraph", "list"]
    assert blocks[1].text == "Caf\xe9 allowance."
    assert blocks[2].section_path == ("BENEFITS",)


def test_registry_selects_by_case_insensitive_extension() -> None:
    registry = ParserRegistry()

    assert isinstance(registry.for_filename("NOTES.TXT"), TextParser)
    assert isinstance(registry.for_filename("README.MD"), MarkdownParser)
    with pytest.raises(UnsupportedFileTypeError) as failure:
        registry.for_filename("archive.zip")
    assert failure.value.code == "unsupported_file_type"


def test_normalization_cleans_text_and_repeated_page_margins() -> None:
    assert normalize_text("inter-\nnational\x00   policy") == "international policy"

    blocks = []
    order = 0
    for page in range(1, 4):
        blocks.extend(
            [
                Block("paragraph", "Company confidential", order, page=page),
                Block("paragraph", f"Useful content on page {page}", order + 1, page=page),
                Block("paragraph", "Handbook 2026", order + 2, page=page),
            ]
        )
        order += 3

    normalized = Normalizer().normalize(blocks)

    assert [block.text for block in normalized] == [
        "Useful content on page 1",
        "Useful content on page 2",
        "Useful content on page 3",
    ]
