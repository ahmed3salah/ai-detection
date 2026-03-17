#!/usr/bin/env python3
"""
Generate AI-written scientific text for training (Option B).

Supports multiple providers: OpenAI, Anthropic (Claude), Google (Gemini), Ollama (local),
OpenRouter (unified gateway to many models), and any OpenAI-compatible API (Together, Groq, vLLM, etc.).

- From topics: generate N abstracts with prompts like "Write a scientific abstract about [topic]."
- Synthetic pairing: for each human abstract (from a file), generate 1–2 AI versions.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

# Add project root
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

# Load .env so API keys can live in .env (which should be in .gitignore)
def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(path)
        return
    except ImportError:
        pass
    # Fallback: parse simple KEY=VALUE lines when python-dotenv not installed
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value

for _env_path in (_project_root / ".env", Path.cwd() / ".env"):
    _load_dotenv(_env_path)


def _complete_openai(client, prompt: str, model: str, max_tokens: int = 300, temperature: float = 0.7) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def get_client_openai(api_key: Optional[str] = None, base_url: Optional[str] = None):
    try:
        import openai
    except ImportError:
        raise ImportError("Install openai: pip install openai")
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY or pass --api-key")
    client = openai.OpenAI(api_key=key, base_url=os.environ.get("OPENAI_API_BASE") or base_url or None)
    def complete(prompt, model, max_tokens=300, temperature=0.7):
        return _complete_openai(client, prompt, model, max_tokens, temperature)
    return complete


def get_client_anthropic(api_key: Optional[str] = None, **kwargs):
    try:
        import anthropic
    except ImportError:
        raise ImportError("Install anthropic: pip install anthropic")
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("Set ANTHROPIC_API_KEY or pass --api-key")
    client = anthropic.Anthropic(api_key=key)
    def complete(prompt, model, max_tokens=300, temperature=0.7):
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text if msg.content else ""
        return text.strip()
    return complete


def get_client_google(api_key: Optional[str] = None, **kwargs):
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Install google-generativeai: pip install google-generativeai")
    key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY or pass --api-key")
    genai.configure(api_key=key)
    def complete(prompt, model, max_tokens=300, temperature=0.7):
        gen_model = genai.GenerativeModel(model)
        resp = gen_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        if resp.text is None:
            return ""
        return resp.text.strip()
    return complete


def get_client_ollama(base_url: Optional[str] = None, **kwargs):
    # Ollama is local; no API key. Uses OpenAI-compatible API at default http://localhost:11434/v1
    try:
        import openai
    except ImportError:
        raise ImportError("Install openai for Ollama: pip install openai")
    url = (base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1").rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1" if not url.endswith("/v1") else url
    client = openai.OpenAI(api_key="ollama", base_url=url)
    def complete(prompt, model, max_tokens=300, temperature=0.7):
        return _complete_openai(client, prompt, model, max_tokens, temperature)
    return complete


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_client_openrouter(api_key: Optional[str] = None, base_url: Optional[str] = None):
    """Use OpenRouter (openrouter.ai) — single API for many models (OpenAI, Anthropic, Google, etc.)."""
    try:
        import openai
    except ImportError:
        raise ImportError("Install openai: pip install openai")
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("Set OPENROUTER_API_KEY or pass --api-key")
    url = (base_url or os.environ.get("OPENROUTER_API_BASE") or OPENROUTER_BASE_URL).rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1" if "/v1" not in url else url
    client = openai.OpenAI(api_key=key, base_url=url)
    def complete(prompt, model, max_tokens=300, temperature=0.7):
        return _complete_openai(client, prompt, model, max_tokens, temperature)
    return complete


# Provider registry: name -> (get_client_fn, default_model)
PROVIDERS = {
    "openai": (get_client_openai, "gpt-4o-mini"),
    "anthropic": (get_client_anthropic, "claude-3-5-haiku-20241022"),
    "google": (get_client_google, "gemini-1.5-flash"),
    "gemini": (get_client_google, "gemini-1.5-flash"),
    "ollama": (get_client_ollama, "llama3.2"),
    "openrouter": (get_client_openrouter, "openai/gpt-4o-mini"),  # OpenRouter: use any model id e.g. anthropic/claude-3-5-haiku
    "openai_compatible": (get_client_openai, "gpt-4o-mini"),  # Any OpenAI-compatible (Together, Groq, vLLM)
}

# Default models for --multi-model (OpenRouter IDs; split across providers for diverse training data)
# Kept current as of 2025: Gemini 2.5, Llama 3.3, DeepSeek R1, and latest Claude/GPT-4o.
DEFAULT_OPENROUTER_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "anthropic/claude-3-5-haiku",
    "anthropic/claude-3-5-sonnet",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-large",
    "deepseek/deepseek-r1",
]


def get_client(provider: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
    """Return a callable complete(prompt, model, max_tokens=300, temperature=0.7) -> str."""
    provider = provider.lower().strip()
    if provider not in PROVIDERS:
        raise ValueError("Unknown provider %r. Choose: %s" % (provider, ", ".join(PROVIDERS)))
    get_fn, _ = PROVIDERS[provider]
    return get_fn(api_key=api_key, base_url=base_url)


# Prompt templates to simulate varied real author requests (one chosen at random per call).
ABSTRACT_PROMPT_TEMPLATES = [
    "Write a short scientific abstract (2–4 sentences) about the following topic, in the style of a %s paper. Do not use bullet points or headings. Topic: %s",
    "Draft a scientific abstract for a %s paper on this topic. 2–4 sentences, no bullets. Topic: %s",
    "Generate a concise abstract (2–4 sentences) for a research paper, %s style. Topic: %s",
    "I need an abstract for my %s paper. Please write 2–4 sentences on: %s",
    "Can you write a brief scientific abstract about this? Style: %s. 2–4 sentences. Topic: %s",
    "Produce a short abstract suitable for a %s conference/journal. Topic: %s",
    "Write an abstract (2–4 sentences) in the style of a %s paper. Topic: %s",
    "Create a scientific abstract for the following topic. %s style, 2–4 sentences, no bullet points. Topic: %s",
]

# System instruction prepended when asking for JSON (reduces markdown/truncation from models like Gemini).
# Many APIs only get a user message, so we bake this into the prompt.
JSON_OUTPUT_SYSTEM_INSTRUCTION = (
    "You must respond with exactly one valid JSON object. "
    "Do not wrap in markdown or code fences (no ```). Do not add any text before or after the JSON. "
    "Use double quotes for keys and string values. Keep the abstract to 2-4 sentences (under 350 characters). "
)

# Robust single prompt for metadata (used for Gemini and optionally others) to avoid truncation/format issues.
ABSTRACT_METADATA_STRICT_PROMPT = (
    "Output a single line of valid JSON with these three keys only: \"title\", \"categories\", \"abstract\". "
    "title: one short paper title (under 150 chars). categories: one or two arxiv codes, e.g. \"cs.LG\" or \"cs.CV, cs.AI\". "
    "abstract: 2-4 sentences summarizing the paper, under 350 characters. "
    "For a %s paper on this topic: %s"
)

# Prompts that ask for title + categories + abstract (for arxiv-style JSONL); response parsed as JSON.
ABSTRACT_WITH_METADATA_TEMPLATES = [
    "For a %s paper on this topic, respond with exactly one JSON object (no other text) with keys: title, categories, abstract. title: one short paper title. categories: one or more arxiv-style codes (e.g. cs.LG, cs.CL, stat.ML). abstract: 2–4 sentences. Topic: %s",
    "Generate a %s paper stub as JSON only. Keys: title (string), categories (string, arxiv codes like cs.AI), abstract (2–4 sentences). Topic: %s",
    "Output a single JSON object with title, categories (e.g. cs.LG), and abstract for a %s paper on: %s",
]


def generate_abstract(complete_fn: Callable, topic: str, model: str, style: str = "CS") -> str:
    """Generate one scientific abstract for the given topic (random prompt template)."""
    template = random.choice(ABSTRACT_PROMPT_TEMPLATES)
    prompt = template % (style, topic)
    text = complete_fn(prompt, model=model, max_tokens=300, temperature=0.7)
    return text.replace("\n", " ")


def _norm_str(val: Any, max_len: Optional[int] = None) -> str:
    """Coerce a value to a single-line string; handle lists (e.g. categories as ["cs.LG"])."""
    if val is None:
        return ""
    if isinstance(val, str):
        s = val.strip().replace("\n", " ")
    elif isinstance(val, list):
        s = " ".join(str(x).strip() for x in val).replace("\n", " ")
    else:
        s = str(val).strip().replace("\n", " ")
    return s[:max_len] if max_len else s


# Minimum length for an abstract to be considered usable (avoids storing "AI-Physicist: A" or "is").
MIN_ABSTRACT_LEN = 50
# Substrings that indicate garbage/corruption (markdown fragments, malformed keys, obvious glitches).
_GARBAGE_SUBSTRINGS = (
    "**Categories**",
    "titlerc",
    "categories\\",
    "  *   *",
    "Pert-urbOfUrbativeive",
    "Order Pert-urb",
    "OfUrbativeive",
    "quququ",
    "pertpert",
)
# Meta/colloquial responses where the model talks about generating JSON instead of outputting it (e.g. Gemini).
_META_JSON_PHRASES = (
    "i've assembled",
    "json output",
    "my title is now confirmed",
    "necessary components",
    "here is the json",
    "here's the json",
    "as requested",
    "below is the json",
)


def _is_usable_abstract(text: str) -> bool:
    """Return False if abstract is too short or looks like garbage/corruption; otherwise True."""
    if not text or not isinstance(text, str):
        return False
    s = text.strip().replace("\n", " ")
    if len(s) < MIN_ABSTRACT_LEN:
        return False
    s_lower = s.lower()
    for bad in _GARBAGE_SUBSTRINGS:
        if bad.lower() in s_lower:
            return False
    for phrase in _META_JSON_PHRASES:
        if phrase in s_lower:
            return False
    # Reject if it's just a title fragment (e.g. "AI-Physicist: A" or "SpectraGuard: A Frequency-")
    if s.endswith(": A") or s.endswith(": A ") or (len(s) < 80 and s.count(" ") < 4):
        return False
    # Reject if it looks like unparsed JSON blob (raw/truncated API response stored as abstract)
    # Normalize escaped quotes so we catch both "title" and \"title\"
    s_norm = s.replace('\\"', '"')
    if s_norm.startswith("{"):
        # Any JSON-like blob (has "title" key) or short truncated fragment
        if '"title"' in s_norm and s_norm.count('"') >= 4:
            return False
        if len(s_norm) < 200 and "title" in s_norm and (":" in s_norm or '"' in s_norm):
            return False
    return True


def _extract_json_fields_regex(raw: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """When json.loads fails (e.g. newlines in abstract), try to extract title, categories, abstract with regex. Returns (title, categories, abstract) or (None, None, None)."""
    title, categories, abstract = None, None, None
    # "title": "..." - take first capture
    mt = re.search(r'"title"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if mt:
        title = mt.group(1).replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\\\", "\\").strip().replace("\n", " ")[:500]
    # "categories": "..." or "categories": ["a", "b"] - try string first, then array
    mc = re.search(r'"categories"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if mc:
        categories = mc.group(1).replace("\\\\", "\\").replace('\\"', '"').strip().replace("\n", " ")[:200]
    else:
        ma = re.search(r'"categories"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
        if ma:
            inner = ma.group(1)
            parts = re.findall(r'"((?:[^"\\]|\\.)*)"', inner)
            categories = " ".join(p.replace('\\"', '"') for p in parts).strip()[:200] if parts else None
    # "abstract": "..." - last occurrence is usually the full abstract (in case "abstract" appears in title)
    abstracts = list(re.finditer(r'"abstract"\s*:\s*"((?:[^"\\]|\\.)*)"', raw))
    if abstracts:
        a = abstracts[-1].group(1)
        abstract = a.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\\\", "\\").strip().replace("\n", " ")
    if not abstract:
        # Truncated: "abstract": "Some text to end of string (no closing quote)
        mat = re.search(r'"abstract"\s*:\s*"((?:[^"\\]|\\.)*)$', raw)
        if mat:
            abstract = mat.group(1).replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\\\", "\\").strip().replace("\n", " ")
    if not categories:
        # Truncated categories string to end (e.g. "categories": "cs.CV, cs.AI" with no closing quote)
        mcat = re.search(r'"categories"\s*:\s*"((?:[^"\\]|\\.)*)$', raw)
        if mcat:
            categories = mcat.group(1).replace("\\\\", "\\").replace('\\"', '"').strip().replace("\n", " ")[:200]
    if abstract:
        return (title, categories, abstract)
    return (None, None, None)


def _extract_truncated_title(raw: str) -> Optional[str]:
    """When JSON is truncated (e.g. no closing quote on title), extract partial title from \"title\": \"... to end."""
    # Truncated: "title": "Graph-Accelerated Discovery of" with no closing quote or abstract
    m = re.search(r'"title"\s*:\s*"((?:[^"\\]|\\.)*)$', raw)
    if m:
        s = m.group(1).replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\\\", "\\").strip().replace("\n", " ")[:500]
        if s:
            return s
    return None


def _parse_json_paper_stub(raw: str) -> Tuple[Optional[str], Optional[str], str]:
    """Extract title, categories, abstract from model JSON or plain text. Returns (title, categories, abstract)."""
    raw = (raw or "").strip()
    # Strip markdown code block if present (full block with closing ```)
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if m:
        raw = m.group(1).strip()
    else:
        # Some models (e.g. Gemini) return only opening fence with no closing: "```json\n{ ..."
        raw = re.sub(r"^\s*```(?:json)?\s*", "", raw).strip()
    # If still not starting with {, try from first { (trailing text before JSON)
    if raw and not raw.startswith("{"):
        idx = raw.find("{")
        if idx >= 0:
            raw = raw[idx:]
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            title = _norm_str(obj.get("title"), 500)
            categories = _norm_str(obj.get("categories"), 200)
            abstract = _norm_str(obj.get("abstract"))
            if abstract:
                return (title or None, categories or None, abstract)
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback: model returned valid-looking JSON that didn't parse (e.g. newlines in abstract string)
    if raw.strip().startswith("{"):
        t, c, a = _extract_json_fields_regex(raw)
        if a:
            return (t, c, a)
        # Truncated JSON (e.g. Gemini cut off): use partial title as title+abstract only if long enough
        partial_title = _extract_truncated_title(raw)
        if partial_title and len(partial_title) >= MIN_ABSTRACT_LEN and _is_usable_abstract(partial_title):
            return (partial_title, None, partial_title)
    return (None, None, raw.replace("\n", " "))


def generate_abstract_with_metadata(
    complete_fn: Callable, topic: str, model: str, style: str = "CS"
) -> Tuple[Optional[str], Optional[str], str]:
    """Generate title, categories, and abstract for arxiv-style output. Returns (title, categories, abstract)."""
    model_lower = (model or "").lower()
    is_gemini = "gemini" in model_lower
    # Use strict prompt and higher token limit for Gemini to reduce truncation and markdown wrapping
    if is_gemini:
        user_part = ABSTRACT_METADATA_STRICT_PROMPT % (style, topic)
        max_tokens = 550
    else:
        template = random.choice(ABSTRACT_WITH_METADATA_TEMPLATES)
        user_part = template % (style, topic)
        max_tokens = 500
    prompt = JSON_OUTPUT_SYSTEM_INSTRUCTION + "\n\n" + user_part
    text = complete_fn(prompt, model=model, max_tokens=max_tokens, temperature=0.7)
    return _parse_json_paper_stub(text or "")


def generate_paired_abstract(complete_fn: Callable, human_abstract: str, model: str, style: str = "CS") -> str:
    """Generate an AI abstract on the same topic as the human one (synthetic pairing)."""
    prompt = (
        "Based on the following scientific abstract, write a different scientific abstract "
        "that covers the same or a very similar topic and contribution, in the style of a %s paper. "
        "Use different wording and structure. 2–4 sentences, no bullet points.\n\nAbstract:\n%s"
    ) % (style, human_abstract[:1500])
    text = complete_fn(prompt, model=model, max_tokens=300, temperature=0.7)
    return text.replace("\n", " ")


def read_human_texts(path: Path):
    """Read human texts: one per line, or ---DOC--- separated."""
    raw = path.read_text(encoding="utf-8")
    if "---DOC---" in raw:
        return [b.strip() for b in raw.split("---DOC---") if b.strip()]
    return [line.strip() for line in raw.splitlines() if line.strip()]


def load_jsonl_for_pairing(path: Path, text_mode: str = "abstract") -> List[Tuple[str, Optional[Dict[str, Any]]]]:
    """
    Load records from an arxiv-style JSONL (id, title, abstract, categories, etc.).
    Returns list of (human_text, source_record). source_record is None for non-JSONL.
    text_mode: "abstract" | "title_abstract" (use title + " " + abstract as prompt text).
    """
    if path.suffix.lower() != ".jsonl":
        return []
    pairs: List[Tuple[str, Optional[Dict[str, Any]]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            abstract = (obj.get("abstract") or "").strip()
            if not abstract:
                continue
            if text_mode == "title_abstract":
                title = (obj.get("title") or "").strip().replace("\n", " ")
                human_text = (title + " " + abstract).strip() or abstract
            else:
                human_text = abstract
            pairs.append((human_text, obj))
    return pairs


def build_arxiv_style_record(
    ai_abstract: str,
    source: Optional[Dict[str, Any]],
    index: int,
    model: str,
    provider: str,
) -> Dict[str, Any]:
    """Build one arxiv-style JSONL record (id, title, abstract, categories, authors_parsed, update_date, ai_written, model, provider).
    Matches arxiv-metadata-pre-llm-sample-15k.jsonl so output can be merged for training."""
    from datetime import date
    record: Dict[str, Any] = {
        "id": "",
        "title": "",
        "abstract": (ai_abstract or "").strip().replace("\n", " "),
        "categories": "",
        "authors_parsed": [],
        "update_date": date.today().isoformat(),
        "ai_written": 1,
        "model": model,
        "provider": provider,
    }
    if source:
        record["id"] = (str(source.get("id", "")) + "-ai-" + str(index)) if source.get("id") else ("ai-%d" % index)
        record["title"] = _norm_str(source.get("title"), 500)
        record["categories"] = _norm_str(source.get("categories"), 200)
    else:
        record["id"] = "ai-%d" % index
    return record


def _run_one_topic(
    complete_fn: Callable,
    topic: str,
    model: str,
    style: str,
    provider: str,
    want_metadata: bool = False,
) -> Optional[Tuple[str, str, str, Optional[Dict[str, Any]], str, str]]:
    """Generate one abstract for a topic. If want_metadata, also get title and categories (arxiv-style). Returns (text, source, meta, source_record, model_used, provider_used) or None."""
    if want_metadata:
        title, categories, abstract = generate_abstract_with_metadata(complete_fn, topic, model=model, style=style)
        if not abstract or not _is_usable_abstract(abstract):
            return None
        source_record: Optional[Dict[str, Any]] = None
        if title is not None or categories is not None:
            source_record = {"title": title or "", "categories": categories or ""}
        return (abstract, "topic", topic[:50], source_record, model, provider)
    text = generate_abstract(complete_fn, topic, model=model, style=style)
    return (text, "topic", topic[:50], None, model, provider) if text else None


def _run_one_paired(
    complete_fn: Callable,
    human: str,
    model: str,
    style: str,
    index: int,
    source_record: Optional[Dict[str, Any]] = None,
    provider: str = "openrouter",
) -> Optional[Tuple[str, str, str, Optional[Dict[str, Any]], str, str]]:
    """Generate one paired abstract. Returns (text, source, meta, source_record, model_used, provider_used) or None."""
    text = generate_paired_abstract(complete_fn, human, model=model, style=style)
    if not text:
        return None
    return (text, "paired", str(index), source_record, model, provider)


def _write_one_result(
    f,
    item: Tuple,
    write_idx: int,
    is_jsonl: bool,
    model_fallback: str,
    provider_fallback: str,
) -> None:
    """Append one generated result to the open file and flush so it is persisted immediately."""
    text = item[0]
    source_record = item[3] if len(item) > 3 else None
    model_used = item[4] if len(item) > 4 else model_fallback
    provider_used = item[5] if len(item) > 5 else provider_fallback
    if is_jsonl:
        record = build_arxiv_style_record(text, source_record, write_idx, model_used, provider_used)
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        f.write(text.replace("\n", " ") + "\n")
    f.flush()


def _print_progress(written: int, total: int, start_time: Optional[float] = None, last_model: Optional[str] = None) -> None:
    """Print a single-line progress update (overwrites previous line)."""
    if total <= 0:
        return
    pct = 100 * written // total
    parts = ["  Progress: %d/%d (%d%%)" % (written, total, pct)]
    if start_time is not None:
        elapsed = int(time.time() - start_time)
        parts.append(" | %ds" % elapsed)
    if last_model:
        # Show short model id (e.g. last part after /)
        short = last_model.split("/")[-1] if "/" in last_model else last_model
        parts.append(" | %s" % short[:24])
    sys.stdout.write("\r" + "".join(parts) + "    ")
    sys.stdout.flush()


def _progress_state_path(out_path: Path) -> Path:
    """Path to the progress/resume state file for this output."""
    return out_path.with_suffix(out_path.suffix + ".progress.json")


def _load_progress_state(out_path: Path) -> Optional[Dict[str, Any]]:
    """Load progress state if it exists and matches current output. Returns None if not resumable."""
    path = _progress_state_path(out_path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_progress_state(
    out_path: Path,
    written_count: int,
    total_planned: int,
    pair_from: Optional[Path] = None,
    topic_file: Optional[Path] = None,
) -> None:
    """Write current progress so the run can be resumed later (pair-from or topic-file mode)."""
    path = _progress_state_path(out_path)
    state: Dict[str, Any] = {
        "output": str(out_path.resolve()),
        "written_count": written_count,
        "total_planned": total_planned,
    }
    if pair_from is not None:
        state["pair_from"] = str(pair_from.resolve())
    if topic_file is not None:
        state["topic_file"] = str(topic_file.resolve())
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=0)
    except OSError:
        pass


def _normalize_path(p: Path) -> str:
    return str(Path(p).resolve())


def _prompt_resume(state: Dict[str, Any], out_path: Path, total_planned: int) -> bool:
    """Ask user to resume. Returns True to resume, False to start over. Uses stdin; if not a TTY, defaults to no."""
    written = state["written_count"]
    if written <= 0 or written >= total_planned:
        return False
    msg = "Resume? %d/%d already written to %s. [y/N] " % (written, total_planned, out_path)
    if sys.stdin.isatty():
        try:
            answer = input(msg).strip().lower()
            return answer in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False
    return False


def _prompt_overwrite_complete(state: Dict[str, Any], total_planned: int, out_path: Path) -> bool:
    """When state shows run already complete, ask to start over. Returns True to overwrite, False to exit."""
    if state["written_count"] < total_planned:
        return True
    msg = "Run already complete (%d written). Start over (overwrite)? [y/N] " % state["written_count"]
    if sys.stdin.isatty():
        try:
            answer = input(msg).strip().lower()
            return answer in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False
    return False


def main():
    p = argparse.ArgumentParser(
        description="Generate AI scientific abstracts for training (multiple providers supported)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Providers and env vars:
  openai             OPENAI_API_KEY          (default)
  anthropic          ANTHROPIC_API_KEY       Claude
  google / gemini    GOOGLE_API_KEY or GEMINI_API_KEY
  ollama             (no key)                local, OLLAMA_BASE_URL=http://localhost:11434/v1
  openrouter         OPENROUTER_API_KEY     single API for many models (e.g. openai/gpt-4o-mini, anthropic/claude-3-5-haiku)
  openai_compatible  OPENAI_API_KEY + OPENAI_API_BASE   Together, Groq, vLLM, etc.
""",
    )
    p.add_argument("--provider", choices=list(PROVIDERS), default="openai",
                   help="AI provider (default: openai)")
    p.add_argument("--api-key", default=None, help="API key (overrides env for the chosen provider)")
    p.add_argument("--base-url", default=None,
                   help="API base URL (OpenAI-compatible and Ollama only)")
    p.add_argument("--model", default=None,
                   help="Model name (default depends on provider)")
    p.add_argument("--style", default="CS", help="Paper style: CS, biology, etc. (default CS)")
    p.add_argument("--topics", nargs="*", help="Topics to generate abstracts for")
    p.add_argument("--topic-file", type=Path, help="File with one topic per line")
    p.add_argument("--pair-from", "--generate-from", type=Path, dest="pair_from",
                   help="Path to human texts or arxiv-style .jsonl; generate AI version(s) per item")
    p.add_argument("--pair-count", type=int, default=1,
                   help="AI abstracts per human when using --pair-from (default 1)")
    p.add_argument("--jsonl-text-mode", default="abstract", choices=("abstract", "title_abstract"),
                   help="When reading .jsonl: use 'abstract' only or 'title_abstract' (default abstract)")
    p.add_argument("--output", default="dataset/ai_abstracts.txt",
                   help="Output file (default: dataset/ai_abstracts.txt). Use .jsonl for arxiv-style training data.")
    p.add_argument("--target", type=int, default=None,
                   help="When using --topic-file: generate this many abstracts by cycling through topics (e.g. 5000 from 2135 topics).")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after generating this many abstracts (default: no limit). With --target, effective count is min(target, limit).")
    p.add_argument("--append", action="store_true", help="Append to output")
    p.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls when --workers=1 (default 0.5)")
    p.add_argument("--workers", type=int, default=1, metavar="N",
                   help="Number of concurrent API calls (default 1). Increase for higher throughput.")
    p.add_argument("--multi-model", action="store_true",
                   help="Use multiple models via OpenRouter; samples are split round-robin across models for diverse training.")
    p.add_argument("--models", default=None,
                   help="Comma-separated OpenRouter model IDs for --multi-model (e.g. openai/gpt-4o-mini,anthropic/claude-3-5-haiku). "
                        "Default: 9 popular models (OpenAI, Anthropic, Google, Meta, Mistral, DeepSeek).")
    args = p.parse_args()

    # Resolve model list when multi-model: use OpenRouter and cycle through models (pair-from and topic-file)
    if args.multi_model:
        args.provider = "openrouter"
        if args.models:
            models_list = [m.strip() for m in args.models.split(",") if m.strip()]
        else:
            models_list = list(DEFAULT_OPENROUTER_MODELS)
        if not models_list:
            print("Error: --multi-model requires at least one model (use --models or default list).")
            return 1
        model = models_list[0]  # fallback for write when result has no model
    else:
        models_list = None
        _, default_model = PROVIDERS[args.provider]
        model = args.model or default_model

    try:
        complete_fn = get_client(
            provider=args.provider,
            api_key=args.api_key,
            base_url=args.base_url,
        )
    except (ImportError, ValueError) as e:
        print("Error: %s" % e)
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    is_jsonl = out_path.suffix.lower() == ".jsonl"

    written_count = 0
    workers = max(1, int(args.workers))
    use_parallel = workers > 1

    if args.pair_from and args.pair_from.exists():
        jsonl_pairs = load_jsonl_for_pairing(args.pair_from, text_mode=args.jsonl_text_mode)
        if jsonl_pairs:
            human_texts = [p[0] for p in jsonl_pairs]
            source_records = [p[1] for p in jsonl_pairs]
            total = min(len(human_texts) * args.pair_count, args.limit) if args.limit else len(human_texts) * args.pair_count
            if models_list:
                print("Using provider=openrouter with %d models. Generating up to %d AI abstract(s) from %d JSONL records (split across models)%s (workers=%d)..." % (
                    len(models_list), total, len(jsonl_pairs), " (limit %d)" % args.limit if args.limit else "", workers))
            else:
                print("Using provider=%s model=%s. Generating up to %d AI abstract(s) from %d JSONL records in %s%s (workers=%d)..." % (
                    args.provider, model, total, len(jsonl_pairs), args.pair_from, " (limit %d)" % args.limit if args.limit else "", workers))
            tasks: List[Tuple] = []
            task_idx = 0
            for i, (human, src) in enumerate(zip(human_texts, source_records)):
                if args.limit and len(tasks) >= args.limit:
                    break
                for c in range(args.pair_count):
                    if args.limit and len(tasks) >= args.limit:
                        break
                    model_k = models_list[task_idx % len(models_list)] if models_list else model
                    provider_used = "openrouter" if models_list else args.provider
                    tasks.append((complete_fn, human, model_k, args.style, i, src, provider_used))
                    task_idx += 1
            # Resume logic: same pair-from + output and more records in source than written
            resume_from = 0
            if not args.append:
                state = _load_progress_state(out_path)
                pair_norm = _normalize_path(args.pair_from)
                out_norm = _normalize_path(out_path)
                if state and _normalize_path(state.get("pair_from", "")) == pair_norm and _normalize_path(state.get("output", "")) == out_norm:
                    prev_written = state.get("written_count", 0)
                    prev_total = state.get("total_planned", total)
                    if prev_written >= total:
                        if _prompt_overwrite_complete(state, total, out_path):
                            mode = "w"
                        else:
                            print("Exiting.")
                            return 0
                    elif prev_written > 0:
                        if _prompt_resume(state, out_path, total):
                            mode = "a"
                            resume_from = prev_written
                            tasks = tasks[resume_from:]
                            written_count = resume_from
                            print("Resuming from %d/%d..." % (resume_from, total))
                        else:
                            mode = "w"
            elif out_path.exists() and out_path.stat().st_size > 0:
                # Append mode: start write index after existing lines so IDs don't collide
                with out_path.open("r", encoding="utf-8") as cf:
                    written_count = sum(1 for _ in cf)
            start_time = time.time()
            with out_path.open(mode, encoding="utf-8") as f:
                if use_parallel:
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        futures = {
                            executor.submit(_run_one_paired, t[0], t[1], t[2], t[3], t[4], t[5], t[6]): t
                            for t in tasks
                        }
                        for fut in as_completed(futures):
                            try:
                                result = fut.result()
                                if result:
                                    _write_one_result(f, result, written_count, is_jsonl, model, args.provider)
                                    written_count += 1
                                    _print_progress(written_count, total, start_time, result[4] if len(result) > 4 else None)
                                    _save_progress_state(out_path, written_count, total, pair_from=args.pair_from)
                            except Exception as e:
                                print("  Warning: %s" % e)
                    print()
                else:
                    for t in tasks:
                        try:
                            result = _run_one_paired(t[0], t[1], t[2], t[3], t[4], t[5], t[6])
                            if result:
                                _write_one_result(f, result, written_count, is_jsonl, model, args.provider)
                                written_count += 1
                                _print_progress(written_count, total, start_time, result[4] if len(result) > 4 else None)
                                _save_progress_state(out_path, written_count, total, pair_from=args.pair_from)
                        except Exception as e:
                            print("  Warning: %s" % e)
                        time.sleep(args.delay)
                    print()
        else:
            human_texts = read_human_texts(args.pair_from)
            total = min(len(human_texts) * args.pair_count, args.limit) if args.limit else len(human_texts) * args.pair_count
            if models_list:
                print("Using provider=openrouter with %d models. Generating up to %d AI abstract(s) from %s (split across models)%s (workers=%d)..." % (
                    len(models_list), total, args.pair_from, " (limit %d)" % args.limit if args.limit else "", workers))
            else:
                print("Using provider=%s model=%s. Generating up to %d AI abstract(s) from %s%s (workers=%d)..." % (
                    args.provider, model, total, args.pair_from, " (limit %d)" % args.limit if args.limit else "", workers))
            tasks = []
            task_idx = 0
            for i, human in enumerate(human_texts):
                if args.limit and len(tasks) >= args.limit:
                    break
                for _ in range(args.pair_count):
                    if args.limit and len(tasks) >= args.limit:
                        break
                    model_k = models_list[task_idx % len(models_list)] if models_list else model
                    provider_used = "openrouter" if models_list else args.provider
                    tasks.append((complete_fn, human, model_k, args.style, i, None, provider_used))
                    task_idx += 1
            resume_from = 0
            if not args.append:
                state = _load_progress_state(out_path)
                pair_norm = _normalize_path(args.pair_from)
                out_norm = _normalize_path(out_path)
                if state and _normalize_path(state.get("pair_from", "")) == pair_norm and _normalize_path(state.get("output", "")) == out_norm:
                    prev_written = state.get("written_count", 0)
                    if prev_written >= total:
                        if _prompt_overwrite_complete(state, total, out_path):
                            mode = "w"
                        else:
                            print("Exiting.")
                            return 0
                    elif prev_written > 0:
                        if _prompt_resume(state, out_path, total):
                            mode = "a"
                            resume_from = prev_written
                            tasks = tasks[resume_from:]
                            written_count = resume_from
                            print("Resuming from %d/%d..." % (resume_from, total))
                        else:
                            mode = "w"
            elif out_path.exists() and out_path.stat().st_size > 0:
                with out_path.open("r", encoding="utf-8") as cf:
                    written_count = sum(1 for _ in cf)
            start_time = time.time()
            with out_path.open(mode, encoding="utf-8") as f:
                if use_parallel:
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        futures = {
                            executor.submit(_run_one_paired, t[0], t[1], t[2], t[3], t[4], t[5], t[6]): t
                            for t in tasks
                        }
                        for fut in as_completed(futures):
                            try:
                                result = fut.result()
                                if result:
                                    _write_one_result(f, result, written_count, is_jsonl, model, args.provider)
                                    written_count += 1
                                    _print_progress(written_count, total, start_time, result[4] if len(result) > 4 else None)
                                    _save_progress_state(out_path, written_count, total, pair_from=args.pair_from)
                            except Exception as e:
                                print("  Warning: %s" % e)
                    print()
                else:
                    for t in tasks:
                        try:
                            result = _run_one_paired(t[0], t[1], t[2], t[3], t[4], t[5], t[6])
                            if result:
                                _write_one_result(f, result, written_count, is_jsonl, model, args.provider)
                                written_count += 1
                                _print_progress(written_count, total, start_time, result[4] if len(result) > 4 else None)
                                _save_progress_state(out_path, written_count, total, pair_from=args.pair_from)
                        except Exception as e:
                            print("  Warning: %s" % e)
                        time.sleep(args.delay)
                    print()
    elif args.topics or args.topic_file:
        topics = list(args.topics or [])
        if args.topic_file:
            topics.extend(read_human_texts(args.topic_file))
        if not topics:
            print("Provide --topics or --topic-file or --pair-from")
            return 1
        # With --target: generate that many by cycling through topics; otherwise one per topic (capped by --limit).
        total = (args.target if args.target is not None else len(topics))
        if args.limit is not None:
            total = min(total, args.limit)
        # Build tasks: cycle topics (and models) so we get exactly `total` items. When JSONL output, request title+categories too.
        want_metadata = is_jsonl
        if models_list:
            print("Using provider=openrouter with %d models. Generating %d abstract(s) from %d topics (cycling)%s (workers=%d)..." % (
                len(models_list), total, len(topics), " (limit %d)" % args.limit if args.limit else "", workers))
            tasks = [(complete_fn, topics[i % len(topics)], models_list[i % len(models_list)], args.style, "openrouter", want_metadata) for i in range(total)]
        else:
            print("Using provider=%s model=%s. Generating %d abstract(s) from %d topics (cycling)%s (workers=%d)..." % (
                args.provider, model, total, len(topics), " (limit %d)" % args.limit if args.limit else "", workers))
            tasks = [(complete_fn, topics[i % len(topics)], model, args.style, args.provider, want_metadata) for i in range(total)]
        # Resume: when using --topic-file, same output and total allow resuming from written_count
        mode = "w"
        written_count = 0
        if not args.append and args.topic_file:
            state = _load_progress_state(out_path)
            out_norm = _normalize_path(out_path)
            topic_norm = _normalize_path(args.topic_file)
            if state and _normalize_path(state.get("output", "")) == out_norm and _normalize_path(state.get("topic_file", "")) == topic_norm:
                prev_written = state.get("written_count", 0)
                prev_total = state.get("total_planned", total)
                if prev_total == total:
                    if prev_written >= total:
                        if _prompt_overwrite_complete(state, total, out_path):
                            mode = "w"
                        else:
                            print("Exiting.")
                            return 0
                    elif prev_written > 0:
                        if _prompt_resume(state, out_path, total):
                            mode = "a"
                            written_count = prev_written
                            tasks = tasks[prev_written:]
                            print("Resuming from %d/%d..." % (written_count, total))
                        else:
                            mode = "w"
        start_time = time.time()
        with out_path.open(mode, encoding="utf-8") as f:
            if use_parallel:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(_run_one_topic, t[0], t[1], t[2], t[3], t[4], t[5]): t
                        for t in tasks
                    }
                    for fut in as_completed(futures):
                        try:
                            result = fut.result()
                            if result:
                                _write_one_result(f, result, written_count, is_jsonl, model, args.provider)
                                written_count += 1
                                _print_progress(written_count, total, start_time, result[4] if len(result) > 4 else None)
                                if args.topic_file:
                                    _save_progress_state(out_path, written_count, total, topic_file=args.topic_file)
                        except Exception as e:
                            print("  Warning: %s" % e)
                print()
            else:
                for (complete_fn_t, topic, model_t, style_t, provider_t, want_meta) in tasks:
                    try:
                        result = _run_one_topic(complete_fn_t, topic, model_t, style_t, provider_t, want_meta)
                        if result:
                            _write_one_result(f, result, written_count, is_jsonl, model, args.provider)
                            written_count += 1
                            _print_progress(written_count, total, start_time, result[4] if len(result) > 4 else None)
                            if args.topic_file:
                                _save_progress_state(out_path, written_count, total, topic_file=args.topic_file)
                    except Exception as e:
                        print("  Warning: %s" % e)
                    time.sleep(args.delay)
                print()
    else:
        print("Provide --topics, --topic-file, or --pair-from (or --generate-from .jsonl). See --help.")
        return 1

    if written_count == 0:
        print("No texts generated.")
        return 1

    print("Wrote %d AI abstracts to %s" % (written_count, out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
