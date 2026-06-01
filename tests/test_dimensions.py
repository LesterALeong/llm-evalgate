from llm_evalkit.eval.dimensions import (
    BlocklistDimension,
    FactualGroundingDimension,
    ReadabilityDimension,
    SchemaComplianceDimension,
)

# --- BlocklistDimension ---

def test_blocklist_clean():
    dim = BlocklistDimension(terms=["secret", "internal"])
    result = dim.run("This is a public document.")
    assert result.passed
    assert result.score == 1.0


def test_blocklist_hit():
    dim = BlocklistDimension(terms=["secret", "internal"])
    result = dim.run("This is an internal document.")
    assert not result.passed
    assert result.score == 0.0
    assert "internal" in result.detail


def test_blocklist_case_insensitive_default():
    dim = BlocklistDimension(terms=["SECRET"])
    result = dim.run("This contains secret info.")
    assert not result.passed


def test_blocklist_case_sensitive():
    dim = BlocklistDimension(terms=["SECRET"], case_sensitive=True)
    result = dim.run("This contains secret info.")
    assert result.passed


# --- SchemaComplianceDimension ---

def test_schema_all_present():
    dim = SchemaComplianceDimension(required_fields=["title:", "summary:", "date:"])
    text = "title: Foo\nsummary: Bar\ndate: 2024-01-01"
    result = dim.run(text)
    assert result.passed
    assert result.score == 1.0


def test_schema_missing_fields():
    dim = SchemaComplianceDimension(required_fields=["title:", "summary:", "date:"])
    text = "title: Foo"
    result = dim.run(text)
    assert not result.passed
    assert result.score < 1.0
    assert "summary:" in result.detail


# --- ReadabilityDimension ---

def test_readability_simple_text_passes():
    dim = ReadabilityDimension(threshold=0.1)
    text = "The cat sat on the mat. It was a fat cat."
    result = dim.run(text)
    assert result.passed


def test_readability_empty_text_fails():
    dim = ReadabilityDimension(threshold=0.1)
    result = dim.run("   ")
    assert not result.passed


# --- FactualGroundingDimension ---

def test_factual_no_evidence_skips():
    dim = FactualGroundingDimension(evidence=None)
    result = dim.run("Revenue was $1.2 billion.")
    assert result.passed
    assert "skipped" in result.detail


def test_factual_grounded():
    dim = FactualGroundingDimension(evidence=[1200000000.0], threshold=0.85)
    result = dim.run("Revenue was 1200000000 dollars.")
    assert result.passed


def test_factual_ungrounded():
    dim = FactualGroundingDimension(evidence=[999.0], threshold=0.85)
    result = dim.run("Revenue was 1200000000 dollars.")
    assert not result.passed


def test_factual_no_numbers_passes():
    dim = FactualGroundingDimension(evidence=[1000.0], threshold=0.85)
    result = dim.run("Revenue grew significantly.")
    assert result.passed
    assert "no numeric" in result.detail
