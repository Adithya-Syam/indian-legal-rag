# ⚖️ LLM-RAG Legal Assistant for Indian Law

[![tests](../../actions/workflows/tests.yml/badge.svg)](../../actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

A legal AI assistant for Indian law. It combines a local LLM (via [Ollama](https://ollama.com)) with **retrieval-augmented generation (RAG)** over real court judgments. It offers step-by-step case reasoning, dual-view summaries, statutory section prediction, bias and hallucination checks, and multilingual chat.

> **Legal codes:** section predictions show current law first: **BNS/BNSS** (in force since 1 July 2024), with the repealed IPC/CrPC section in brackets, e.g. `BNS 103 (formerly IPC 302)`.

> ⚠️ This is a research/education project, **not legal advice**.

---

## ✨ Features

| Feature | What it does | Module |
|---|---|---|
| 🧠 Chain-of-Thought analysis | Reasons in 4 visible steps: **Law → Fact → Argument → Outcome** | `src/reasoning/cot_reasoning.py` |
| 📝 Dual-view summarizer | Professional + plain-language summary; long judgments are summarized in parts, then combined | `src/visualization/legal_visualizer.py` |
| ⚖️ Section predictor | Maps facts to BNS/BNSS (and IPC/CrPC) sections with confidence scores (keywords + semantic similarity) | `src/legal_sections/section_predictor.py` |
| 🔍 Bias & hallucination check | Flags gender/religion/caste bias and suspicious or invalid citations | `src/bias/bias_detector.py` |
| 🌐 Multilingual chat | Detects Hindi, Tamil and other Indian languages and answers in the same language | `src/multilingual/multilingual_chat.py` |
| 📚 RAG over judgments | Answers grounded in retrieved judgments, with cited sources | `src/rag/legal_rag.py` |
| 📊 Evaluation | ROUGE, BLEU, readability, coherence, bias precision/recall/F1 | `src/evaluation/comprehensive_evaluator.py` |

## 🏗️ Architecture

```
                    ┌──────────────── offline (once) ────────────────┐
 dataset/ ──► preprocess_data.py ──► build_rag_index.py ──► data/rag_index/ (FAISS)
 (judgments +                         split text → MiniLM embeddings
  summaries)
                    └────────────────────────────────────────────────┘

 web_app/enhanced_app.py (Streamlit)
   │
   ├─ question ──► legal_rag.search() ──► top-k judgment pieces ─┐
   │                                                              ▼
   ├─ reasoning / summaries / bias ─────────────────────► gemma3 via Ollama (localhost:11434)
   │
   └─ section predictor (rule-based + embeddings, no LLM call)
```

One LLM (`gemma3` in Ollama) serves every feature. Fine-tuning Llama 3.2 1B is optional.

## 🚀 Quick start

### 1. Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com) installed and running
- 8 GB RAM minimum (16 GB recommended). On 8 GB machines, use `gemma3:1b`.

### 2. Install
```bash
git clone https://github.com/Adithya-Syam/indian-legal-rag.git
cd indian-legal-rag
python -m venv venv
# Windows: venv\Scripts\activate    macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
ollama pull gemma3
```

### 3. Get the dataset
The dataset is **not included** in this repo. It uses the IN-Abs, IN-Ext and UK-Abs legal summarization datasets (Shukla et al., *Legal Case Document Summarization*, AACL-IJCNLP 2022). Download them and place them so that you have:
```
dataset/
├── IN-Abs/
├── IN-Ext/
└── UK-Abs/
```
On Windows you can link an existing copy instead: `New-Item -ItemType Junction -Path dataset -Target C:\path\to\dataset`

### 4. Build and run
```bash
python scripts/preprocess_data.py --dataset_path dataset --output_path data/processed --clean_text
python scripts/build_rag_index.py --documents_path dataset --index_path data/rag_index
streamlit run web_app/enhanced_app.py        # opens http://localhost:8501
```
With `make`: `make quick-start`.

## ⚙️ Configuration

| Setting | Where | Default |
|---|---|---|
| LLM model | env `OLLAMA_MODEL` (e.g. `gemma3:1b` on 8 GB RAM) | `gemma3:latest` |
| LLM context window | env `OLLAMA_NUM_CTX` | `16384` |
| Embedding model | `config/rag_config.yaml` | `all-MiniLM-L6-v2` |
| Chunk size / overlap, top-k | `config/rag_config.yaml` | 1000 / 200, 5 |
| Fine-tuning hyperparameters | `config/training_config*.yaml` | 3 epochs, LoRA |

## 🎓 Optional: fine-tuning
Fine-tunes `meta-llama/Llama-3.2-1B-Instruct` with LoRA on the judgment → summary pairs. You need a GPU and a Hugging Face account with access to the Llama model.
```bash
pip install -r requirements-train.txt
huggingface-cli login
python scripts/train_summarization.py
```
See [STEPS.md](STEPS.md) for details.

## 🧪 Tests
```bash
python -m pytest                    # unit tests (no Ollama needed)
python -m pytest tests/integration  # end-to-end scripts (need Ollama running)
```

## 📁 Project structure
```
├── config/            # model, RAG and training settings (YAML)
├── scripts/           # preprocess_data, build_rag_index, train_summarization
├── src/
│   ├── bias/          # bias & hallucination detection
│   ├── chatbot/       # chatbot combining RAG + LLM
│   ├── data/          # dataset loading & preprocessing
│   ├── evaluation/    # metrics
│   ├── legal_sections/# BNS/BNSS + IPC/CrPC section predictor
│   ├── models/        # optional fine-tuned summarizer (LoRA)
│   ├── multilingual/  # language detection + multilingual chat
│   ├── rag/           # FAISS retrieval + grounded answers
│   ├── reasoning/     # chain-of-thought analysis
│   ├── utils/         # Ollama client, device helpers
│   └── visualization/ # dual-view summaries
├── tests/             # unit tests (+ integration/ scripts)
└── web_app/           # Streamlit app
```

## ⚠️ Limitations
- The section predictor covers a small set of common sections. Always verify against the official BNS/BNSS text.
- LLM output can be wrong. The bias/hallucination checker reduces this risk but doesn't remove it.
- Translation quality depends on the LLM in use.

## 📚 More docs
- [DOCUMENTATION.md](DOCUMENTATION.md): detailed usage guide
- [STEPS.md](STEPS.md): fine-tuning and deployment steps

## 📄 License
[MIT](LICENSE)
