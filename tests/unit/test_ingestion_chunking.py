from __future__ import annotations

from dataclasses import replace

from rag.ingestion import Block, Chunker, deduplicate_chunks


def word_count(text: str) -> int:
    return len(text.split())


def test_chunker_obeys_hard_limit_overlap_and_section_boundaries() -> None:
    first = " ".join(f"alpha{i}" for i in range(24))
    second = " ".join(f"beta{i}" for i in range(5))
    blocks = [
        Block(
            "paragraph",
            first,
            0,
            page=2,
            section_path=("Alpha",),
            source_locator={"line_start": 1},
        ),
        Block(
            "paragraph",
            second,
            1,
            page=3,
            section_path=("Beta",),
            source_locator={"line_start": 2},
        ),
    ]
    chunker = Chunker(
        target_tokens=10,
        max_tokens=12,
        min_tokens=3,
        overlap_tokens=2,
        token_counter=word_count,
    )

    chunks = chunker.chunk(blocks, "version-1", "Example")

    assert all(chunk.token_count <= 12 for chunk in chunks)
    assert all("beta" not in chunk.text for chunk in chunks if chunk.section_path == ("Alpha",))
    assert chunks[-1].section_path == ("Beta",)
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]
    assert chunks[0].page_start == chunks[0].page_end == 2
    assert chunks[0].embedding_text.startswith("Document: Example\nSection: Alpha\n\n")
    assert chunks == chunker.chunk(blocks, "version-1", "Example")
    assert chunks[0].id != chunker.chunk(blocks, "version-2", "Example")[0].id


def test_chunker_keeps_source_and_embedding_representations_separate() -> None:
    chunks = Chunker(
        target_tokens=20,
        max_tokens=25,
        min_tokens=2,
        overlap_tokens=0,
        token_counter=word_count,
    ).chunk(
        [
            Block(
                "paragraph",
                "Employees receive twelve days.",
                0,
                section_path=("Leave",),
                source_locator={"paragraph": 8},
            )
        ],
        "v1",
        "Handbook",
    )

    assert chunks[0].text == "Employees receive twelve days."
    assert "Handbook" not in chunks[0].text
    assert "Handbook" in chunks[0].embedding_text
    assert chunks[0].source_locator["blocks"][0]["locator"] == {"paragraph": 8}


def test_exact_content_deduplication_is_opt_in() -> None:
    original = Chunker(
        target_tokens=10,
        max_tokens=12,
        min_tokens=1,
        overlap_tokens=0,
        token_counter=word_count,
    ).chunk([Block("paragraph", "same content", 0)], "v1")[0]
    duplicate = replace(original, id="another-id", ordinal=1)

    assert deduplicate_chunks([original, duplicate]) == [original]
