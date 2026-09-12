from __future__ import annotations

from rag.ingestion import Block
from rag.retrieval import (
    CountableDocument,
    ExactCountRequest,
    count_exact_occurrences,
    format_exact_count_answer,
    parse_exact_count_request,
)


def document(*blocks: Block, filename: str = "resume.pdf") -> CountableDocument:
    return CountableDocument(
        document_id="document-1",
        version_id="version-1",
        filename=filename,
        blocks=tuple(blocks),
    )


def test_recognizes_quoted_count_question_from_ui() -> None:
    request = parse_exact_count_request('how many times, the word "ISTQB" is written ?')

    assert request == ExactCountRequest(term="ISTQB", case_sensitive=False)


def test_recognizes_unquoted_word_but_not_an_ordinary_quoted_question() -> None:
    assert parse_exact_count_request("Count the word ISTQB") == ExactCountRequest(term="ISTQB")
    assert parse_exact_count_request('What does "ISTQB" stand for?') is None


def test_counts_complete_parser_blocks_once_without_chunk_overlap() -> None:
    source = document(
        Block(
            "paragraph",
            "CERTIFICATIONS: ISTQB Foundation and ISTQB Advanced.",
            order=0,
            page=2,
        )
    )

    result = count_exact_occurrences(ExactCountRequest("ISTQB"), (source,))

    assert result.total == 2
    assert result.documents[0].count == 2
    assert len(result.evidence) == 1
    assert result.evidence[0].occurrence_count == 2
    assert result.evidence[0].page == 2
    assert format_exact_count_answer(result) == (
        "The word “ISTQB” appears 2 times in resume.pdf "
        "(case-insensitive exact whole-word match). [S1]"
    )


def test_default_count_is_case_insensitive_and_respects_word_boundaries() -> None:
    source = document(Block("paragraph", "ISTQB istqb NOTISTQB ISTQB-Advanced", order=0))

    result = count_exact_occurrences(ExactCountRequest("ISTQB"), (source,))

    assert result.total == 3


def test_case_sensitive_count_and_multiword_phrase() -> None:
    request = parse_exact_count_request(
        'How many times is the phrase "Certified Tester" written, case-sensitive?'
    )
    assert request == ExactCountRequest(term="Certified Tester", case_sensitive=True)
    source = document(
        Block(
            "paragraph",
            "Certified Tester; certified tester; Certified\nTester",
            order=0,
        )
    )

    result = count_exact_occurrences(request, (source,))

    assert result.total == 2
