import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_pdf(pages):
    """Build a minimal valid PDF with one line of text per page (no extra deps)."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None,
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        content_id = len(objects)
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                       f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>")
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"

    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    path.write_bytes(make_pdf(["The capital of Atlantis is Poseidonia.", "Atlantis sank in a single day."]))
    return path


@pytest.fixture
def fake_pipeline_factory():
    """RAGPipeline with fake embeddings and a fake chat model (no OpenAI calls)."""
    from rag_pipeline import RAGPipeline

    def factory(api_key=None, **kwargs):
        return RAGPipeline(
            api_key=api_key,
            embeddings=DeterministicFakeEmbedding(size=32),
            llm=FakeListChatModel(responses=["Poseidonia."] * 50),
        )
    return factory
