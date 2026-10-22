# TruthLens

### LLM Hallucination Detection, Severity Assessment & Automated Correction

> A Multi-Task Transformer Framework for detecting, assessing, and correcting LLM hallucinations.

TruthLens analyzes AI-generated responses for potential hallucinations, estimates their severity, identifies suspicious claims, and generates a possible correction — designed as a **decision-support and quality-checking system, not an absolute truth oracle**.

---

## 1. Problem Statement

Large Language Models can generate answers that sound convincing but contain incorrect, unsupported, outdated, or fabricated information — commonly known as **hallucinations**. TruthLens acts as a verification and correction layer between an LLM and the end user.

## 2. Motivation

Hallucinations can mislead users, spread misinformation, and in high-stakes domains (medical, legal, financial, safety) cause real harm. Automated detection and severity estimation help users calibrate trust and prioritize verification.

## 3. Objectives

- Detect hallucinated LLM responses (binary)
- Assess severity (Low / Medium / High)
- Identify problematic claims
- Generate corrected responses
- Provide confidence scores (with disclaimers)
- Explain model predictions vs verified evidence
- Provide a clean beginner-friendly Streamlit interface
- Support Demonstration → Pretrained → Fine-Tuned modes
- Enable easy extension and evaluation

## 4. Proposed Solution

TruthLens uses a **shared Transformer encoder** (DistilBERT) with two classification heads (detection + severity), plus a separate **FLAN-T5** generator for corrections. Heuristic claim extraction and explanation layers provide transparency.

## 5. System Architecture

```
                    User Question
                         +
                    LLM Response
                         |
                         v
               +-------------------+
               | Text Preprocessing|
               +-------------------+
                         |
                         v
               +-------------------+
               | Transformer       |
               | Encoder           |
               | (DistilBERT)      |
               +-------------------+
                         |
                         v
               Shared Representation
                         |
              +----------+----------+
              |                     |
              v                     v
       Detection Head        Severity Head
              |                     |
              v                     v
        Hallucination           Low/Medium/
        Yes / No                High
              |                     |
              +----------+----------+
                         |
                         v
                  Claim Identification
                         |
                         v
                   Explanation Layer
                         |
                         v
                  Correction Prompt
                         |
                         v
               Transformer Generator
                  FLAN-T5 / T5
                         |
                         v
                Corrected Response
```

Loss: `L_total = λ1 * L_detection + λ2 * L_severity` (both CrossEntropy, configurable λ).

## 6. Features

- **Hallucination Detection** — Yes/No with confidence
- **Severity Assessment** — Low/Medium/High with confidence
- **Claim Identification** — isolates most suspicious sentence (never fabricates)
- **Explanation** — distinguishes Model Prediction vs Verified Evidence
- **Automated Correction** — FLAN-T5-small with safety warnings
- **Confidence Disclaimer** — clearly states confidence ≠ factual certainty
- **Three Model Modes** — Demonstration / Pretrained / Fine-Tuned (auto-detected)
- **History** — session-based (Streamlit state, no database)
- **Export** — TXT and JSON reports
- **Evaluation** — sklearn metrics + optional ROUGE/BLEU with caveats

## 7. Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.10+ | Language |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | Encoder (DistilBERT) & Generator (FLAN-T5) |
| Datasets | Dataset loading |
| scikit-learn | Metrics (accuracy, F1, confusion matrix) |
| pandas, NumPy | Data handling |
| Streamlit | Web UI |
| NLTK / rouge-score | Optional correction metrics |

No Docker/Kubernetes/FastAPI/React/DB required — runs on a student laptop (CPU by default, GPU auto-used if available).

## 8. Project Structure

```
TruthLens/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── sample_dataset.csv
├── models/
│   ├── __init__.py
│   ├── detector.py
│   ├── severity.py
│   ├── corrector.py
│   └── multitask_model.py
├── training/
│   ├── __init__.py
│   ├── train_detector.py
│   ├── train_severity.py
│   └── train_multitask.py
├── inference/
│   ├── __init__.py
│   └── pipeline.py
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── claim_extractor.py
│   ├── explanation.py
│   └── evaluation.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_pipeline.py
│   └── test_models.py
├── outputs/
│   └── .gitkeep
└── saved_models/
    └── .gitkeep
```

## 9. Dataset

`data/sample_dataset.csv` — **Demonstration Dataset** with 35 examples (exceeds 30 required).

Columns: `id, question, response, hallucination, severity, corrected_response`

Covers: geography, history, science, mathematics, technology, medicine, sports, literature. Includes both hallucinated and non-hallucinated examples with severity Low/Medium/High.

> ⚠️ 35 examples are insufficient for production-grade training — this is a demonstration dataset for academic illustration.

Types included: Wrong Entity, Wrong Number, Wrong Location, Wrong Historical Event, Wrong Scientific Claim, Wrong Mathematical Result, Wrong Technical Information, Medical misinformation (safe educational examples).

## 10. Installation

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

## 11. Run Application

```bash
# If `streamlit` not on PATH (Windows), use:
C:\Python314\python.exe -m streamlit run app.py
# or
python -m streamlit run app.py
# Or add to PATH: C:\Users\<you>\AppData\Roaming\Python\Python314\Scripts
```

Opens at `http://localhost:8501`.

### 11.1 Evidence-Based Fact Checking

Questions and answers are entered by the user and **do not need to appear in `data/sample_dataset.csv`**. The dataset-trained detector is a separate estimate and may generalize poorly to unseen topics. Enable **Check with hosted Llama using Wikipedia evidence** to search for evidence from both inputs and send the question, answer, and retrieved passages to the selected NVIDIA-hosted Llama model. Invention questions also search topic history, and compound questions search the additional context separately. The model is instructed to assess central claims and chronology, and classifies the answer as *supported*, *contradicted*, *off-topic*, or *inconclusive*. A contradicted or off-topic answer is shown as **HALLUCINATION DETECTED**. A valid NVIDIA API key is required; enter it in the sidebar.

The Llama verdict is grounded in retrieved evidence, not a guarantee that every fact on every topic can be verified. If Wikipedia has no relevant evidence, the API key is missing, or hosted Llama cannot complete a check, TruthLens reports **UNVERIFIED** rather than inventing a verdict. Wikipedia may be incomplete or outdated, and the model can make mistakes, so independently verify important claims. Disable the checkbox to skip evidence retrieval and use the existing trained-detector estimate. The question, answer, and evidence are sent to NVIDIA when this check is enabled; do not submit sensitive information.

Hosted requests time out after 25 seconds by default (`NVAPI_TIMEOUT` can override this). If the evidence check times out, the app does not make another hosted correction request or wait for slow local text generation during that analysis; it shows the detector estimate and a rule-based correction fallback.

If NVIDIA returns HTTP 404 or 410 because the selected Llama model is unavailable to the account, the evidence check automatically retries the configured `meta/llama-*` alternatives. Other API errors and timeouts are reported without retrying multiple models.

### 11.2 Llama (NVIDIA NVAPI - Hosted, No Local GPU Needed)

TruthLens now supports **hosted Llama via NVIDIA NVAPI** for high-quality corrections without downloading 2GB+ models.

**Get a free NVAPI key:**
1. Go to https://build.nvidia.com/explore/reasoning → pick a Llama (e.g., `meta/llama-3.2-11b-vision-instruct`) → **Get API Key**
2. Save it in one of three ways (priority: Streamlit > env var > .env):
   - **Streamlit UI (easiest):** Sidebar → paste `nvapi-...` into *NVAPI Key* field (session-only, never saved to disk)
   - **.env file:** Copy `.env.example` to `.env` and set `NVAPI_KEY=nvapi-...`
   - **Environment variable:** `set NVAPI_KEY=nvapi-...` (Windows) or `export NVAPI_KEY=nvapi-...` (Linux/macOS)

**Enable:**
```bash
# .env already contains your key (gitignored)
# Just restart the app:
python -m streamlit run app.py
# Sidebar will show: ✅ NVAPI Llama active: meta/llama-3.2-11b-vision-instruct
```

**Config (`config.py`):**
```python
NVAPI_MODEL = "meta/llama-3.2-11b-vision-instruct"  # default, working as of Sep 2026
# Fallbacks auto-tried: gemma-3-12b-it, mixtral-8x7b, qwen2.5-72b, nvidia nemotron
# EOL - do NOT use: meta/llama-3.1-8b-instruct (410 Gone)
```

**Local Llama alternative:** For offline/CPU use, set in `config.py`:
```python
CORRECTION_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # no auth, ~2GB, slower on CPU
# or FLAN baseline: "google/flan-t5-small"
```
This local model is used for correction generation; evidence-based hallucination verdicts use the hosted Llama selected above and require an NVAPI key.

Correction priority is: **NVAPI Llama > Local Transformer > Rule-based fallback** — if NVAPI fails/times out, it automatically falls back.

## 12. Training

```bash
# Multi-task (detection + severity, recommended)
python training/train_multitask.py

# Detection only
python training/train_detector.py

# Severity only
python training/train_severity.py
```

Training automatically uses CUDA if available, else CPU. Progress is displayed via `tqdm` and logs. The best model (by detection F1) is saved to `saved_models/multitask/`.

You can experiment with loss weights in `config.py`:

```python
LAMBDA_DETECTION = 1.0  # λ1
LAMBDA_SEVERITY = 1.0   # λ2
```

Or compare encoders by changing `BASE_ENCODER_MODEL` (e.g., `bert-base-uncased`, `roberta-base`, `deberta-base`).

## 13. Evaluation

`utils/evaluation.py` provides:

- **Detection:** Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report
- **Severity:** Accuracy, Macro Precision/Recall/F1, Confusion Matrix
- **Correction (optional):** BLEU (NLTK), ROUGE (rouge-score)

> Automatic text-generation metrics do not guarantee factual correctness. A generated correction can have high lexical similarity while still containing a factual error, or low similarity while being factually correct.

Example:

```python
from utils.evaluation import evaluate_detection, evaluate_severity
results_det = evaluate_detection(y_true, y_pred)
results_sev = evaluate_severity(y_true_sev, y_pred_sev)
```

## 14. Example

**Input**

- Question: "What is the capital of Australia?"
- Response: "The capital of Australia is Sydney."

**Output**

- Hallucination: Detected (96%)
- Severity: Medium (87%)
- Claim: "The capital of Australia is Sydney."
- Explanation: The response incorrectly identifies Sydney as the capital; Canberra is the capital.
- Correction: "The capital of Australia is Canberra."
- Warning: ⚠️ Generated corrections should be independently verified...

## 15. Severity Definitions

- **Low** — Minor factual error with limited impact (incorrect minor date, name spelling, small numerical detail)
- **Medium** — Meaningful factual error that could mislead (incorrect capital, historical fact, scientific explanation, technical spec)
- **High** — Potentially dangerous or highly consequential (serious medical misinformation, dangerous instructions, incorrect financial/legal claims, major safety misinformation)

## 16. Model Modes

| Mode | When | Display |
|---|---|---|
| **Demonstration / Baseline** | No fine-tuned model, no cached pretrained | Demonstration Mode — rule-based heuristics, for illustration only |
| **Pretrained Transformer** | Pretrained encoder available but heads untrained | Pretrained Encoder / Untrained Classification Head |
| **Fine-Tuned TruthLens** | `saved_models/multitask/pytorch_model.bin` exists | Fine-Tuned TruthLens Model |

The app auto-detects the available mode at startup.

## 17. Limitations

- Small demonstration dataset (35 examples) — not production-grade
- Model uncertainty — hallucinations in open-ended answers are inherently hard
- No guaranteed factual verification — no web browsing / external KB by default
- Generated corrections are AI-generated and may still be wrong
- Domain-specific hallucinations may not be covered
- Difficulty evaluating open-ended answers with automatic metrics
- Confidence scores reflect model preference, not factual certainty

## 18. Future Enhancements

- Larger, diverse datasets
- Retrieval-Augmented Generation (RAG) with external knowledge
- Web-based fact checking & claim-level verification
- Better calibration & confidence calibration
- Human feedback & human-in-the-loop verification
- Explainable AI (attention visualization, LIME/SHAP)
- Domain-specific models
- Multilingual support
- More advanced severity modeling
- Benchmark evaluation (e.g., HaluEval, FEVER)

## 19. Inference Pipeline Usage

```python
from inference.pipeline import analyze

result = analyze(
    question="What is the capital of Australia?",
    response="The capital of Australia is Sydney."
)
print(result["hallucination"])          # True
print(result["severity"])                # Medium
print(result["claim"])                   # The capital of Australia is Sydney.
print(result["correction"])              # The capital of Australia is Canberra.
```

Or via class:

```python
from inference.pipeline import TruthLensPipeline
pipeline = TruthLensPipeline()
result = pipeline.analyze(question, response)
```

## 20. Testing

```bash
python -m pytest tests/ -v
# or
python tests/test_preprocessing.py
python tests/test_pipeline.py
python tests/test_models.py
```

Tests cover preprocessing, claim extraction, pipeline (valid/missing inputs), and model output shapes — all CPU-only.

## 21. Scientific Honesty

TruthLens **estimates** whether a response *may* contain hallucinated information based on its trained or baseline model. It does **not** guarantee truth.

- Generated corrections are described as **"Suggested AI-generated correction"**, not guaranteed truth.
- If no external verification is configured, the UI states: *"TruthLens has not independently verified this claim against an external knowledge source."*
- Confidence is labeled as model preference, not factual certainty.

## 22. License

MIT — for academic and educational use.

---

**TruthLens** — *Not a truth oracle, but a lens for critical evaluation.*
