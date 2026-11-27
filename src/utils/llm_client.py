#!/usr/bin/env python3
"""
Real LLM client for Ollama integration.
Provides actual AI reasoning instead of static responses.
"""

import os
import requests
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    tokens_used: int
    response_time: float
    success: bool
    error: Optional[str] = None

class OllamaClient:
    """Client for Ollama LLM integration."""

    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "gemma3:latest",
                 num_ctx: Optional[int] = None):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server URL
            default_model: Default model to use
            num_ctx: Context window in tokens (env OLLAMA_NUM_CTX, default 16384).
                     Larger = longer inputs but more RAM.
        """
        self.base_url = base_url
        self.default_model = os.environ.get("OLLAMA_MODEL", default_model)
        self.num_ctx = num_ctx or int(os.environ.get("OLLAMA_NUM_CTX", 16384))
        self.session = requests.Session()

    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
        except Exception as e:
            logger.error(f"Error getting models: {e}")
        return []

    def generate(self, prompt: str, model: Optional[str] = None,
                system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        """
        Generate response from LLM.

        Args:
            prompt: User prompt
            model: Model to use (optional)
            system_prompt: System prompt (optional)
            **kwargs: Additional parameters

        Returns:
            LLMResponse: Generated response
        """
        start_time = time.time()
        model = model or self.default_model

        try:
            # Prepare request
            data = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9),
                    # Ollama's names: num_predict = max output tokens, num_ctx = context window.
                    # Without num_ctx Ollama uses a small default and silently truncates long inputs.
                    "num_predict": kwargs.get("max_tokens", 2000),
                    "num_ctx": kwargs.get("num_ctx", self.num_ctx),
                }
            }

            if system_prompt:
                data["system"] = system_prompt

            # Make request (large contexts on CPU can take minutes)
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=kwargs.get("timeout", 300)
            )

            response_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                return LLMResponse(
                    content=result.get("response", "").strip(),
                    model=model,
                    tokens_used=result.get("eval_count", 0),
                    response_time=response_time,
                    success=True
                )
            else:
                return LLMResponse(
                    content="",
                    model=model,
                    tokens_used=0,
                    response_time=response_time,
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}"
                )

        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"LLM generation error: {e}")
            return LLMResponse(
                content="",
                model=model or "unknown",
                tokens_used=0,
                response_time=response_time,
                success=False,
                error=str(e)
            )

class LegalLLMClient:
    """Specialized LLM client for legal reasoning."""

    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        """Initialize legal LLM client."""
        self.ollama = ollama_client or OllamaClient()
        self.legal_system_prompt = """You are an expert Indian legal AI assistant with deep knowledge of:
- Indian Penal Code (IPC)
- Criminal Procedure Code (CrPC)
- Indian legal precedents and case law
- Legal reasoning and analysis

Provide accurate, structured legal analysis. If input is not legal in nature, clearly state that and refuse to analyze."""

    def validate_legal_input(self, text: str) -> Dict[str, Any]:
        """Validate if input contains legal content."""
        if not text or len(text.strip()) < 3:
            return {"valid": False, "reason": "Input too short or empty"}

        text_lower = text.lower()

        # Check for legal patterns (more comprehensive)
        legal_patterns = [
            # Legal keywords
            r'\b(?:section|ipc|crpc|court|case|accused|victim|murder|theft|fraud|assault|evidence|judgment|legal|law|crime|criminal|civil|contract)\b',
            # Case numbers
            r'\b(?:case|cases)\s+(?:no|nos|number|numbers)\.?\s*\d+',
            r'\b\d+\s+of\s+\d{4}\b',  # "297 of 1951"
            # Legal citations
            r'\bv\.?\s+[A-Z]',  # "A v. B"
            r'\b(?:appellant|respondent|petitioner|defendant|plaintiff)\b',
            # Indian legal terms
            r'\b(?:supreme court|high court|district court|magistrate|judge|advocate|lawyer|bail|custody|warrant|summons)\b',
            # Legal actions
            r'\b(?:filed|charged|convicted|acquitted|sentenced|appealed|remanded)\b',
            # Legal documents
            r'\b(?:fir|complaint|petition|appeal|writ|order|decree|injunction)\b'
        ]

        import re
        legal_score = 0
        for pattern in legal_patterns:
            if re.search(pattern, text_lower):
                legal_score += 1

        # Additional check for numbers that might be legal references
        if re.search(r'\d+', text) and len(text.strip()) > 5:
            legal_score += 0.5  # Partial score for potential legal references

        # Be more lenient - accept if any legal pattern is found
        if legal_score == 0:
            return {"valid": False, "reason": "No legal content detected"}

        return {"valid": True, "score": legal_score}

    def analyze_legal_case(self, case_text: str, analysis_type: str = "general") -> LLMResponse:
        """Analyze legal case with structured reasoning."""
        # Validate input
        validation = self.validate_legal_input(case_text)
        if not validation["valid"]:
            return LLMResponse(
                content=f"Invalid input: {validation['reason']}",
                model="validation",
                tokens_used=0,
                response_time=0.0,
                success=False,
                error=validation["reason"]
            )

        # Create specialized prompt based on analysis type
        if analysis_type == "chain_of_thought":
            prompt = f"""Analyze this legal case using Chain-of-Thought reasoning following LAW → FACT → ARGUMENT → OUTCOME:

Case: {case_text}

Note: If this appears to be a case number or citation (like "Cases Nos. 297 and 298 of 1951"), provide analysis based on what such cases typically involve in Indian legal context.

Provide structured analysis:

1. LAW Analysis:
   - Identify applicable IPC/CrPC sections (if determinable)
   - State relevant legal principles
   - Explain legal framework
   - If case numbers only, discuss typical legal issues from that era

2. FACT Analysis:
   - Extract key facts (if available)
   - If only case numbers provided, note the limitation
   - Assess what can be determined from the information

3. ARGUMENT Analysis:
   - Typical arguments for such cases
   - Legal precedent considerations
   - Procedural aspects

4. OUTCOME Prediction:
   - Analysis based on available information
   - Confidence level (0.0-1.0)
   - Note limitations if insufficient details

Provide meaningful analysis even with limited information."""

        elif analysis_type == "section_prediction":
            prompt = f"""Predict applicable IPC/CrPC sections for this case:

Case Facts: {case_text}

Provide:
1. Primary applicable sections with confidence scores
2. Secondary/related sections
3. Reasoning for each section
4. Case severity assessment
5. Case category

Format as structured analysis."""

        else:
            prompt = f"""Analyze this legal case comprehensively:

Case: {case_text}

Provide detailed legal analysis covering applicable laws, facts, arguments, and likely outcomes."""

        return self.ollama.generate(
            prompt=prompt,
            system_prompt=self.legal_system_prompt,
            temperature=0.3,  # Lower temperature for legal analysis
            max_tokens=3000
        )

    def detect_bias(self, text: str) -> LLMResponse:
        """Detect bias in legal text using LLM."""
        prompt = f"""Analyze this legal text for bias and problematic language:

Text: {text}

Check for:
1. Gender bias (stereotypes, discriminatory language)
2. Religious bias (prejudicial references)
3. Caste/social bias (discriminatory assumptions)
4. Factual inconsistencies or hallucinations
5. Inappropriate legal citations

Provide structured analysis with:
- Bias instances found (with quotes)
- Severity levels (Low/Medium/High/Critical)
- Suggested improvements
- Overall bias score (0.0-1.0)

Format as JSON."""

        return self.ollama.generate(
            prompt=prompt,
            system_prompt="You are an expert in detecting bias and ensuring fair legal analysis. Be thorough and objective.",
            temperature=0.2
        )

    # ~4 chars per token: 30k chars ~ 7.5k tokens, leaves room in a 16k context for prompt + output
    CHUNK_CHARS = 30000
    CHUNK_OVERLAP = 1500

    @staticmethod
    def _split_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
        """Split on paragraph boundaries into chunks of at most ~chunk_chars, with overlap."""
        chunks, start = [], 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            if end < len(text):
                cut = text.rfind("\n\n", start + chunk_chars // 2, end)
                if cut == -1:
                    cut = text.rfind(". ", start + chunk_chars // 2, end)
                if cut != -1:
                    end = cut + 1
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
        return [c for c in chunks if c]

    def summarize_legal_text(self, text: str, view_type: str = "professional") -> LLMResponse:
        """Summarize legal text; long judgments are summarized chunk-by-chunk, then combined."""
        if len(text) <= self.CHUNK_CHARS:
            return self._summarize_single(text, view_type)

        chunks = self._split_text(text, self.CHUNK_CHARS, self.CHUNK_OVERLAP)
        logger.info(f"Long document ({len(text)} chars): summarizing {len(chunks)} chunks")
        notes, total_tokens, total_time = [], 0, 0.0
        for i, chunk in enumerate(chunks, 1):
            prompt = f"""This is part {i} of {len(chunks)} of a court judgment.

Text: {chunk}

Extract concise notes (bullet points) covering only what appears in this part:
- Parties and facts
- Legal issues raised
- Arguments of each side
- Statutes/sections and precedents cited (exactly as written)
- Findings, reasoning and any final order/decision

Do not invent anything not present in the text."""
            resp = self.ollama.generate(prompt=prompt, system_prompt=self.legal_system_prompt,
                                        temperature=0.2, max_tokens=800)
            total_tokens += resp.tokens_used
            total_time += resp.response_time
            if not resp.success:
                return resp
            notes.append(f"[Part {i}]\n{resp.content}")

        final = self._summarize_single("\n\n".join(notes), view_type, from_notes=True)
        final.tokens_used += total_tokens
        final.response_time += total_time
        return final

    def _summarize_single(self, text: str, view_type: str, from_notes: bool = False) -> LLMResponse:
        """Summarize text that fits in one context window."""
        if from_notes:
            text = ("(The following are ordered notes extracted from each part of a long judgment. "
                    "The final decision is usually in the last parts.)\n\n" + text)
        if view_type == "public":
            prompt = f"""Summarize this legal text for the general public in simple, clear language:

Legal Text: {text}

Requirements:
- Use plain English, avoid legal jargon
- Explain legal terms when necessary
- Make it accessible to non-lawyers
- Keep key legal points accurate
- Maximum 200 words

Provide clear, understandable summary."""

        else:  # professional
            prompt = f"""Provide a professional legal summary of this text:

Legal Text: {text}

Requirements:
- Maintain legal terminology and precision
- Include key legal principles and precedents
- Highlight important sections and citations
- Structure for legal professionals
- Maximum 300 words

Provide comprehensive legal summary."""

        return self.ollama.generate(
            prompt=prompt,
            system_prompt=self.legal_system_prompt,
            temperature=0.4
        )

# Global instance
legal_llm = LegalLLMClient()

def get_legal_llm() -> LegalLLMClient:
    """Get global legal LLM client instance."""
    return legal_llm
