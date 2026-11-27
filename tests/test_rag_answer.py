"""RAG answers should come from gemma3 (Ollama), with graceful fallbacks."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(str(Path(__file__).parent.parent))

pytest.importorskip("faiss")
pytest.importorskip("langchain_text_splitters")
pytest.importorskip("requests")

from src.rag.legal_rag import LegalRAG
from src.utils.llm_client import LLMResponse


def _fake_llm(available=True, content="Answer citing [Document: 1953_L_1]"):
    llm = MagicMock()
    llm.ollama.is_available.return_value = available
    llm.ollama.generate.return_value = LLMResponse(content=content, model="gemma3", tokens_used=5,
                                                   response_time=0.1, success=True)
    return llm


def test_uses_ollama_when_available():
    with patch("src.utils.llm_client.get_legal_llm", return_value=_fake_llm()):
        assert "1953_L_1" in LegalRAG._generate_answer("prompt")


def test_falls_back_to_local_model_when_ollama_down():
    local = MagicMock()
    local.summarize.return_value = "local answer"
    with patch("src.utils.llm_client.get_legal_llm", return_value=_fake_llm(available=False)):
        assert LegalRAG._generate_answer("prompt", local) == "local answer"


def test_context_only_message_when_no_llm():
    with patch("src.utils.llm_client.get_legal_llm", return_value=_fake_llm(available=False)):
        assert "ollama serve" in LegalRAG._generate_answer("prompt")
