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

### AI arXiv-style JSONL (`--output` ending in `.jsonl`)

Topic-file runs that write JSONL assign `id` values `ai-0`, `ai-1`, … aligned with the task index (cycling topics and, with `--multi-model`, models in order). Rows must pass shared checks in `scripts/ai_jsonl_quality.py`: non-empty **title** and usable **abstract** (length, no model JSON narration, heavy LaTeX/homework patterns, etc.), and by default non-empty **categories**. Use `--no-require-categories` to allow empty categories.

- **Retries:** `--max-retries` (default 3) re-calls the model when validation fails.
- **Failure log:** JSON lines are appended to `<output_stem>.failures.jsonl` by default for topic-file JSONL runs (override with `--failure-log`).
- **Placeholder rows:** If validation still fails, a row with `generation_failed: 1` is written so line indices stay stable unless you pass `--skip-failure-rows` (not recommended if you rely on `ai-N` ids).

**Audit (split clean vs rejected)**

```bash
python scripts/audit_ai_jsonl.py audit --input dataset/ai_abstracts_20k.jsonl
```

Default outputs: `dataset/ai_abstracts_20k.cleaned.jsonl` and `dataset/ai_abstracts_20k.rejected.jsonl` (override with `--output-clean` / `--output-rejects`).

**Regenerate only some indices** (use the same `--topic-file`, `--target`, `--multi-model`, and `--models` ordering as the original batch). `reject_indices.txt` is one integer per line (or pass a comma-separated list):

```bash
python scripts/generate_ai_data.py --topic-file dataset/topics.txt --regenerate-indices reject_indices.txt \
  --output dataset/ai_regen_patch.jsonl --provider openrouter --multi-model
```

**Merge patch back into the full file**

```bash
python scripts/audit_ai_jsonl.py merge --base dataset/ai_abstracts_20k.jsonl --patch dataset/ai_regen_patch.jsonl \
  --output dataset/ai_abstracts_20k.merged.jsonl --validate --still-bad dataset/still_bad.jsonl
```

Save the exact command line used for large runs (topic file path, `--target`, `--models`, `--workers`, `--delay`) so regeneration stays reproducible.

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
