"""Tests for Ollama options and chunked summarization (Ollama is mocked)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.append(str(Path(__file__).parent.parent))

pytest.importorskip("requests")

from src.utils.llm_client import LegalLLMClient, LLMResponse, OllamaClient


def _ok(text="summary"):
    return LLMResponse(content=text, model="m", tokens_used=10, response_time=0.1, success=True)


def test_request_uses_ollama_option_names():
    client = OllamaClient(num_ctx=8192)
    client.session = MagicMock()
    client.session.post.return_value.status_code = 200
    client.session.post.return_value.json.return_value = {"response": "hi", "eval_count": 1}
    client.generate("p", max_tokens=123)
    opts = client.session.post.call_args.kwargs["json"]["options"]
    assert opts["num_predict"] == 123 and opts["num_ctx"] == 8192
    assert "max_tokens" not in opts


def test_split_covers_whole_text_with_overlap():
    text = "\n\n".join(f"Paragraph {i} " + "x" * 900 for i in range(100))  # ~90k chars
    chunks = LegalLLMClient._split_text(text, 30000, 1500)
    assert len(chunks) >= 3
    assert all(len(c) <= 30000 for c in chunks)
    assert "Paragraph 0" in chunks[0] and "Paragraph 99" in chunks[-1]


def test_short_text_is_one_call():
    llm = LegalLLMClient(ollama_client=MagicMock())
    llm.ollama.generate.return_value = _ok()
    llm.summarize_legal_text("short judgment", "public")
    assert llm.ollama.generate.call_count == 1


def test_long_text_is_chunked_then_combined():
    llm = LegalLLMClient(ollama_client=MagicMock())
    llm.ollama.generate.return_value = _ok("notes")
    long_text = "\n\n".join("The court held " + "y" * 900 for _ in range(100))
    result = llm.summarize_legal_text(long_text, "professional")
    n_chunks = len(LegalLLMClient._split_text(long_text, llm.CHUNK_CHARS, llm.CHUNK_OVERLAP))
    assert llm.ollama.generate.call_count == n_chunks + 1  # one per chunk + final combine
    final_prompt = llm.ollama.generate.call_args.kwargs["prompt"]
    assert "[Part 1]" in final_prompt and f"[Part {n_chunks}]" in final_prompt
    assert result.success and result.tokens_used == 10 * (n_chunks + 1)
