import pytest

from conftest import make_pdf


def test_pipeline_answers_with_sources(sample_pdf, fake_pipeline_factory):
    pipeline = fake_pipeline_factory()
    stats = pipeline.initialize_from_pdf(str(sample_pdf))
    assert stats == {"pages": 2, "chunks": 2}

    result = pipeline.query_with_sources("What is the capital of Atlantis?")
    assert result["answer"] == "Poseidonia."
    assert {s["page"] for s in result["sources"]} == {1, 2}
    assert pipeline.query("Again?") == "Poseidonia."


def test_missing_pdf(fake_pipeline_factory):
    with pytest.raises(FileNotFoundError):
        fake_pipeline_factory().initialize_from_pdf("does-not-exist.pdf")


def test_pdf_without_text(tmp_path, fake_pipeline_factory):
    path = tmp_path / "blank.pdf"
    path.write_bytes(make_pdf([""]))
    with pytest.raises(ValueError, match="No text"):
        fake_pipeline_factory().initialize_from_pdf(str(path))
