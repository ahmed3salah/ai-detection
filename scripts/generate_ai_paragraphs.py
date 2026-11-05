"""
Generate the AI side of the paragraph dataset from the human paragraphs.

Every AI paragraph replaces a real one: same paper, same section, same length. Topic, domain and
length then match across the classes, so a model can only separate them on the writing. (The old
generator capped AI text at 350 characters, and the model learned length.)

Spend is capped: --budget-usd is a hard stop, checked before every call against what has actually
been spent, across restarts. --dry-run prints the estimate and spends nothing.

Safe to stop and rerun: every attempt is in the ledger and is not repeated.

    python scripts/generate_ai_paragraphs.py --count 2000 --budget-usd 5 --dry-run
    python scripts/generate_ai_paragraphs.py --count 2000 --budget-usd 5
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from segment_dataset import TEST, assign_paper  # noqa: E402

OPENROUTER_CHAT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"

# Across vendors, from frontier to cheap. Checked against OpenRouter's catalogue on each run.
TRAINING_MODELS = [
    "openai/gpt-5.4-mini",
    "openai/gpt-4o-mini",
    "anthropic/claude-sonnet-5.5",
    "anthropic/claude-haiku-5.5",
    "google/gemini-2.5-flash",
    "google/gemini-3.1-flash-lite",
    "meta-llama/llama-4-maverick",
    "deepseek/deepseek-v4-flash",
]
# Whole vendors the model never trains on, to measure how it copes with a generator it hasn't seen.
HELDOUT_MODELS = ["mistralai/mistral-large-2512", "qwen/qwen3-235b-a22b-2507"]

LENGTH_BUCKETS = ((40, 80), (80, 120), (120, 180), (180, 300))
WORD_TOLERANCE = 0.3
TOKENS_PER_WORD = 1.5
MAX_TOKENS_FACTOR = 3
CHARS_PER_TOKEN = 4
HELDOUT_SHARE_OF_TEST_PAPERS = 0.5
OUTLINE_SHARE = 0.3
MAX_CONSECUTIVE_MODEL_ERRORS = 5

INSTRUCTIONS = (
    "Write about {words} words of plain academic prose as one paragraph. No heading, list, markdown, "
    "citations or references. Do not mention these instructions. Reply with the paragraph only."
)

_META = re.compile(r"^\s*(here is|here's|sure|certainly|paragraph:|abstract:)|as an ai\b|\*\*|^\s*#|\n\s*[-*] ", re.I | re.M)


@dataclass(frozen=True)
class Task:
    id: str
    model: str
    mode: str
    prompt: str
    target_words: int
    source: Dict


def stable_fraction(*parts: object) -> float:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:12], 16) / 16**12


def task_id(prefix: str, *parts: str) -> str:
    """Deterministic, so a rerun recognises what it has already tried."""
    return f"{prefix}-{hashlib.sha256('|'.join(parts).encode()).hexdigest()[:16]}"


def load_human(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def word_count(text: str) -> int:
    return len(text.split())


def select_targets(rows: Sequence[Dict], count: int, seed: int) -> List[Dict]:
    """Paragraphs to replace, the same number from each length bucket so every length gets tested."""
    rng = random.Random(seed)
    buckets: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    for row in rows:
        words = word_count(row["text"])
        for low, high in LENGTH_BUCKETS:
            if low <= words < high or (high == LENGTH_BUCKETS[-1][1] and words == high):
                buckets[(low, high)].append(row)
    picked: List[Dict] = []
    each = count // len(LENGTH_BUCKETS)
    leftovers: List[Dict] = []
    for bucket in LENGTH_BUCKETS:
        members = sorted(buckets[bucket], key=lambda row: row["id"])
        rng.shuffle(members)
        picked.extend(members[:each])
        leftovers.extend(members[each:])
    rng.shuffle(leftovers)
    return picked + leftovers[: max(0, count - len(picked))]


def choose_model(row: Dict, seed: int, training_models: Sequence[str], heldout_models: Sequence[str]) -> str:
    """Held-out generators write only for test papers: on any other paper the model has seen the text."""
    def pick(models: Sequence[str]) -> str:
        return models[int(stable_fraction("model", seed, row["id"]) * len(models))]

    in_test = assign_paper(row["paper_id"], seed) == TEST
    if heldout_models and in_test and stable_fraction("heldout", seed, row["id"]) < HELDOUT_SHARE_OF_TEST_PAPERS:
        return pick(heldout_models)
    return pick(training_models)


def build_prompt(row: Dict, previous: Optional[Dict], words: int, seed: int) -> Tuple[str, str]:
    """The mode and prompt for replacing a paragraph. The paragraph itself is never shown."""
    rules = INSTRUCTIONS.format(words=words)
    title = row.get("title") or "an academic paper"
    if row["section"] == "abstract":
        return "abstract", f'Write the abstract of a research paper titled "{title}". {rules}'
    heading = row.get("heading") or row["section"]
    if previous is not None and stable_fraction("mode", seed, row["id"]) >= OUTLINE_SHARE:
        return "continue", (
            f'You are writing the section "{heading}" of a research paper titled "{title}". '
            f"The previous paragraph is:\n\n{previous['text']}\n\nWrite the next paragraph. {rules}"
        )
    return "outline", f'Write a paragraph for the section "{heading}" of a research paper titled "{title}". {rules}'


def build_tasks(rows: Sequence[Dict], count: int, seed: int, training_models: Sequence[str], heldout_models: Sequence[str], polish_count: int = 0) -> List[Task]:
    by_paper: Dict[str, Dict[int, Dict]] = defaultdict(dict)
    for row in rows:
        by_paper[row["paper_id"]][row.get("position", -1)] = row

    tasks: List[Task] = []
    targets = select_targets(rows, count, seed)
    for row in targets:
        words = word_count(row["text"])
        previous = by_paper[row["paper_id"]].get(row.get("position", -1) - 1)
        mode, prompt = build_prompt(row, previous, words, seed)
        model = choose_model(row, seed, training_models, heldout_models)
        tasks.append(Task(task_id("ai", row["id"], model, mode), model, mode, prompt, words, row))

    # Polished human text is measured, never trained on. These don't overlap the replaced paragraphs.
    used = {row["id"] for row in targets}
    spare = sorted((row for row in rows if row["id"] not in used and row["section"] != "abstract" and 40 <= word_count(row["text"]) <= 300), key=lambda row: row["id"])
    random.Random(seed + 1).shuffle(spare)
    for row in spare[:polish_count]:
        model = training_models[int(stable_fraction("polish", seed, row["id"]) * len(training_models))]
        prompt = (
            "Correct the grammar and improve the clarity of this paragraph from a research paper. Keep its meaning, "
            f"its claims and its length. Reply with the edited paragraph only, no markdown.\n\n{row['text']}"
        )
        tasks.append(Task(task_id("polish", row["id"], model), model, "polish", prompt, word_count(row["text"]), row))
    return tasks


def rejection_reason(text: str, finish_reason: Optional[str], target_words: int) -> Optional[str]:
    """Why a reply can't go in the dataset. The old set kept 2,121 replies cut off mid-sentence."""
    if not text.strip():
        return "empty"
    if finish_reason not in ("stop", "end_turn", None):
        return f"finish_reason:{finish_reason}"
    if _META.search(text):
        return "meta_or_markdown"
    if not text.rstrip().endswith((".", "!", "?", '"', ")")):
        return "no_sentence_end"
    words = word_count(text)
    if abs(words - target_words) > WORD_TOLERANCE * target_words:
        return "word_count"
    return None


def fetch_pricing(session: Optional[requests.Session] = None) -> Dict[str, Tuple[float, float]]:
    """Each model's USD per token, in and out, from OpenRouter's public catalogue."""
    response = (session or requests).get(OPENROUTER_MODELS, timeout=60)
    response.raise_for_status()
    return {
        model["id"]: (float(model["pricing"]["prompt"]), float(model["pricing"]["completion"]))
        for model in response.json()["data"]
        if model.get("pricing")
    }


def max_tokens_for(task: Task) -> int:
    """Roomy: on a reasoning model thinking counts against this, and a reply cut short is paid for."""
    return int(task.target_words * TOKENS_PER_WORD * MAX_TOKENS_FACTOR) + 256


def worst_case_cost(task: Task, pricing: Dict[str, Tuple[float, float]]) -> float:
    per_in, per_out = pricing[task.model]
    return len(task.prompt) / CHARS_PER_TOKEN * per_in + max_tokens_for(task) * per_out


def estimated_cost(task: Task, pricing: Dict[str, Tuple[float, float]]) -> float:
    per_in, per_out = pricing[task.model]
    return len(task.prompt) / CHARS_PER_TOKEN * per_in + task.target_words * TOKENS_PER_WORD * per_out


_MODELS_THAT_REFUSE_REASONING_SETTING = set()


def openrouter_completion(session: requests.Session, api_key: str, task: Task, pricing: Dict[str, Tuple[float, float]]) -> Dict:
    """One call: the text, why it stopped and what it cost. Retries while the provider is busy."""
    body = {
        "model": task.model,
        "messages": [{"role": "user", "content": task.prompt}],
        "max_tokens": max_tokens_for(task),
        "temperature": 0.7,
        "usage": {"include": True},
    }
    # A paragraph needs no reasoning. Most models take this; a few answer 400 and go without it.
    if task.model not in _MODELS_THAT_REFUSE_REASONING_SETTING:
        body["reasoning"] = {"enabled": False}
    for attempt in range(4):
        try:
            response = session.post(OPENROUTER_CHAT, json=body, headers={"Authorization": f"Bearer {api_key}"}, timeout=180)
        except requests.RequestException as e:
            if attempt == 3:
                raise RuntimeError(type(e).__name__) from None
            time.sleep(5 * (attempt + 1))
            continue
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
            time.sleep(10 * (attempt + 1))
            continue
        if response.status_code == 400 and "reasoning" in body:
            _MODELS_THAT_REFUSE_REASONING_SETTING.add(task.model)
            del body["reasoning"]
            continue
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}")
        payload = response.json()
        if "error" in payload or not payload.get("choices"):
            if attempt < 3:
                time.sleep(10 * (attempt + 1))
                continue
            raise RuntimeError(f"provider error {str(payload.get('error', {}).get('code', ''))[:40]}")
        usage = payload.get("usage") or {}
        per_in, per_out = pricing[task.model]
        cost = usage.get("cost")
        if not isinstance(cost, (int, float)):
            cost = usage.get("prompt_tokens", 0) * per_in + usage.get("completion_tokens", 0) * per_out
        choice = payload["choices"][0]
        return {"text": (choice.get("message") or {}).get("content") or "", "finish_reason": choice.get("finish_reason"), "cost": float(cost)}
    raise RuntimeError("no answer")


class Ledger:
    """Every attempt and what it cost, on disk, so a restart knows what was spent and tried."""

    def __init__(self, path: Path):
        self.path = path
        self.attempted = set()
        self.spent = 0.0
        if path.exists():
            for line in path.read_text().splitlines():
                entry = json.loads(line)
                self.spent += entry.get("cost", 0.0)
                # A provider failure can be tried again; a reply that was rejected was paid for.
                if entry["status"] != "error":
                    self.attempted.add(entry["id"])

    def write(self, entry: Dict) -> None:
        self.spent += entry.get("cost", 0.0)
        if entry["status"] != "error":
            self.attempted.add(entry["id"])
        with self.path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")


def row_for(task: Task, text: str) -> Dict:
    source = task.source
    row = {
        "id": task.id,
        "text": text.strip(),
        "paper_id": source["paper_id"],
        "domain": source["domain"],
        "section": source["section"],
        "generator": task.model,
        "mode": task.mode,
        "source_segment_id": source["id"],
        "target_words": task.target_words,
    }
    if task.mode == "polish":
        return {**row, "label": 0, "test_set": "polished"}
    return {**row, "label": 1}


def run(
    tasks: Sequence[Task],
    *,
    output: Path,
    ledger: Ledger,
    budget: float,
    pricing: Dict[str, Tuple[float, float]],
    complete: Callable[[Task], Dict],
    workers: int = 6,
) -> Dict:
    """Run tasks until they are done or the budget is. Never starts a call the budget can't cover."""
    lock = threading.Lock()
    in_flight = {"reserved": 0.0}
    statuses: Counter = Counter()
    reasons: Counter = Counter()
    errors_in_a_row: Counter = Counter()
    disabled = set()
    stopped = threading.Event()

    def attempt(task: Task) -> None:
        worst = worst_case_cost(task, pricing)
        with lock:
            if stopped.is_set() or task.model in disabled:
                statuses["skipped"] += 1
                return
            if ledger.spent + in_flight["reserved"] + worst > budget:
                stopped.set()
                statuses["skipped"] += 1
                return
            in_flight["reserved"] += worst

        entry = {"id": task.id, "model": task.model, "mode": task.mode, "source": task.source["id"]}
        try:
            reply = complete(task)
        except Exception as e:  # The text of a failure can carry the request, so keep only its kind.
            with lock:
                in_flight["reserved"] -= worst
                errors_in_a_row[task.model] += 1
                if errors_in_a_row[task.model] >= MAX_CONSECUTIVE_MODEL_ERRORS:
                    disabled.add(task.model)
                ledger.write({**entry, "status": "error", "reason": str(e)[:80], "cost": 0.0})
                statuses["error"] += 1
                reasons[f"{task.model}: {str(e)[:50]}"] += 1
            return

        reason = rejection_reason(reply["text"], reply.get("finish_reason"), task.target_words)
        with lock:
            in_flight["reserved"] -= worst
            errors_in_a_row[task.model] = 0
            if reason is None:
                with output.open("a") as handle:
                    handle.write(json.dumps(row_for(task, reply["text"]), ensure_ascii=False) + "\n")
                ledger.write({**entry, "status": "ok", "cost": reply["cost"], "words": word_count(reply["text"])})
                statuses["ok"] += 1
            else:
                ledger.write({**entry, "status": "rejected", "reason": reason, "cost": reply["cost"]})
                statuses["rejected"] += 1
                reasons[f"{task.model}: {reason}"] += 1

    todo = [task for task in tasks if task.id not in ledger.attempted]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(attempt, todo))

    return {
        "tasks": len(tasks),
        "already_done": len(tasks) - len(todo),
        "statuses": dict(statuses),
        "spent_usd": round(ledger.spent, 4),
        "budget_usd": budget,
        "stopped_on_budget": stopped.is_set(),
        "models_disabled": sorted(disabled),
        "reasons": dict(reasons.most_common(25)),
    }


def dry_run_summary(tasks: Sequence[Task], pricing: Dict[str, Tuple[float, float]]) -> Dict:
    per_model: Dict[str, Dict] = defaultdict(lambda: {"calls": 0, "usd": 0.0})
    for task in tasks:
        per_model[task.model]["calls"] += 1
        per_model[task.model]["usd"] += estimated_cost(task, pricing)
    total = sum(entry["usd"] for entry in per_model.values())
    return {
        "calls": len(tasks),
        "estimate_usd": round(total, 2),
        "worst_case_usd": round(sum(worst_case_cost(task, pricing) for task in tasks), 2),
        "per_model": {model: {"calls": entry["calls"], "usd": round(entry["usd"], 3)} for model, entry in sorted(per_model.items())},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", default="dataset/segments")
    parser.add_argument("--count", type=int, required=True, help="AI paragraphs to attempt")
    parser.add_argument("--polish-count", type=int, default=0, help="Polished human paragraphs for the test-only set")
    parser.add_argument("--budget-usd", type=float, required=True, help="Hard stop, across restarts")
    parser.add_argument("--dry-run", action="store_true", help="Print the estimate. Spends nothing, needs no key.")
    parser.add_argument("--models", default=",".join(TRAINING_MODELS))
    parser.add_argument("--heldout-models", default=",".join(HELDOUT_MODELS))
    parser.add_argument("--seed", type=int, default=42, help="Must match train_segments.py --seed")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    workdir = Path(args.workdir)
    human = workdir / "human_segments.jsonl"
    if not human.exists():
        print(f"{human} doesn't exist. Run build_human_paragraphs.py first.")
        return 1

    training = [m for m in args.models.split(",") if m]
    heldout = [m for m in args.heldout_models.split(",") if m]
    pricing = fetch_pricing()
    missing = [model for model in training + heldout if model not in pricing]
    if missing:
        print(f"Not in OpenRouter's catalogue any more: {', '.join(missing)}. Choose others with --models / --heldout-models.")
        return 1

    tasks = build_tasks(load_human(human), args.count, args.seed, training, heldout, args.polish_count)
    summary = dry_run_summary(tasks, pricing)
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return 0

    load_dotenv()
    api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        print("OPENROUTER_API_KEY isn't set. Put it in .env; nothing was spent.")
        return 1

    session = requests.Session()
    report = run(
        tasks,
        output=workdir / "ai_segments.jsonl",
        ledger=Ledger(workdir / "ai_ledger.jsonl"),
        budget=args.budget_usd,
        pricing=pricing,
        complete=lambda task: openrouter_completion(session, api_key, task, pricing),
        workers=args.workers,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
