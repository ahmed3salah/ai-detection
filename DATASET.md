# Dataset: human + AI data for training (Option B)

Training uses **local data only** by default. Use **separate files** for clearer organization:

| File | Content | Label |
|------|---------|--------|
| `dataset/human_abstracts.txt` | Human-written abstracts | 0 |
| `dataset/human_reviews.txt`  | Human-written reviews  | 0 |
| `dataset/ai_abstracts.txt`   | AI-generated abstracts | 1 |
| `dataset/ai_reviews.txt`     | AI-generated reviews   | 1 |

All four are optional; any combination is merged.  
**Backward compatibility:** `human_text.txt` and `ai_text.txt` are still loaded (same labels) if you prefer the old layout.

---

## 1. File format

Each file: **one document per line**, or documents separated by **`---DOC---`** for multi-line text.

Example `human_abstracts.txt`:

```
We present a novel method for detecting machine-generated text. Experiments on three benchmarks show consistent gains.
This study examines the effect of temperature on reaction rates. Our results suggest that existing models may need revision.
```

Example `human_reviews.txt` (multi-line with separator):

```
This paper proposes a new method for X. The experiments are well designed. However, the baseline comparison is limited.
---DOC---
The contribution is clear. I suggest expanding the related work section. Overall I recommend acceptance.
```

---

## 2. Where to get human text

| Source | Output file | Command |
|--------|-------------|---------|
| **Your own** | `human_abstracts.txt` or `human_reviews.txt` | Paste/copy (one per line or `---DOC---`) |
| **arXiv** | `human_abstracts.txt` | `python scripts/fetch_human_data.py --source arxiv --max 200 --output dataset/human_abstracts.txt` |
| **PubMed** | `human_abstracts.txt` | `python scripts/fetch_human_data.py --source pubmed --max 200 --output dataset/human_abstracts.txt` |

Use `--before-year 2021` to reduce chance of LLM-assisted writing. Use `--append` to add to an existing file.

**Large ArXiv dump (`arxiv-metadata-pre-llm.jsonl`):** If you place this file in `dataset/`, training will automatically use it as **human** (label 0) abstracts. The file is streamed and **sampled** (default 100,000) so it is not loaded entirely into memory. Use `--arxiv-max-samples N` when running `training.py` to change the cap, or `--arxiv-max-samples 0` to disable. No need to shorten the file yourself.

---

## 3. Generating AI text

**Multiple providers:** `--provider openai | anthropic | google | gemini | ollama | openai_compatible`

| Provider | Env vars | Install |
|----------|----------|---------|
| **openai** (default) | `OPENAI_API_KEY` | `pip install openai` |
| **anthropic** (Claude) | `ANTHROPIC_API_KEY` | `pip install anthropic` |
| **google** / **gemini** | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | `pip install google-generativeai` |
| **ollama** (local) | (none); `OLLAMA_BASE_URL` optional | `pip install openai` (Ollama speaks OpenAI API) |
| **openai_compatible** | `OPENAI_API_KEY` + `OPENAI_API_BASE` | Together, Groq, vLLM, etc. |

| Goal | Output file | Command |
|------|-------------|---------|
| **Abstracts from topics** | `ai_abstracts.txt` | `python scripts/generate_ai_data.py --topics "topic1" "topic2" --output dataset/ai_abstracts.txt` |
| **Abstracts paired to human** | `ai_abstracts.txt` | `python scripts/generate_ai_data.py --pair-from dataset/human_abstracts.txt --output dataset/ai_abstracts.txt` |
| **Reviews paired to human** | `ai_reviews.txt` | `python scripts/generate_ai_data.py --pair-from dataset/human_reviews.txt --output dataset/ai_reviews.txt` |
| **Using Claude** | any | `--provider anthropic` and set `ANTHROPIC_API_KEY` |
| **Using local Ollama** | any | `--provider ollama --model llama3.2` (Ollama running on localhost) |
| **From topic file** | e.g. `ai_abstracts.txt` | `python scripts/generate_ai_data.py --topic-file dataset/topics.txt --output dataset/ai_abstracts.txt` |

---

## 4. Single JSONL (alternative)

If you prefer one file: **`dataset/data.jsonl`** – one JSON object per line:

```json
{"text": "Your abstract or review here.", "label": 0}
{"text": "AI-generated text...", "label": 1}
```

If `data.jsonl` exists, it is used **instead of** the separate text files.

---

## 5. Train and evaluate

```bash
python training.py
python evaluate.py --dataset-dir dataset
```

Training loads all present files under `dataset/` (separate or JSONL). No IDMGSP or Hugging Face required.
