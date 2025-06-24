"""Dataset loading utilities."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

from .morph import analyze
from .preprocess import extract_concepts, infer_domain

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """Raw QA sample as loaded from JSON."""

    question: dict
    answer: dict
    concepts: List[str]
    domain: str


def load_all(path: Path) -> List[Sample]:
    """Recursively load all JSON files under ``path``."""

    root = path.resolve()
    files = list(root.rglob("*.json"))
    samples: List[Sample] = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            q = item.get("question", {})
                            a = item.get("answer", {})
                            conc = item.get("concepts", [])
                            dom = item.get("domain", "")
                            if q.get("text", "").strip() == a.get("text", "").strip():
                                q = {"text": "", "tokens": []}
                        elif isinstance(item, str):
                            tokens = analyze(item)
                            q = {"text": "", "tokens": []}
                            a = {"text": item, "tokens": tokens}
                            conc = extract_concepts(tokens)
                            dom = infer_domain(tokens)
                        else:
                            continue
                        samples.append(
                            Sample(
                                question=q,
                                answer=a,
                                concepts=conc,
                                domain=dom,
                            )
                        )
            except Exception as exc:
                logger.warning("skip %s | %s", fp.name, exc)
    logger.info("files used: %s", ", ".join(p.name for p in files))
    logger.info(
        "dataset loaded: %d samples from %d files", len(samples), len(files)
    )
    return samples

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional
    pd = None


@dataclass
class QAPair:
    """Simple question-answer pair."""

    question: str
    answer: str
    tokens_q: List[dict]
    tokens_a: List[dict]
    concepts: List[str]
    domain: str


class QADataset:
    """Dataset loader for JSON QA pairs.

    This class accepts either a single JSON file or a directory containing
    multiple JSON files. All files are merged into one dataset.
    """

    def __init__(self, path: Path) -> None:
        self.paths: List[Path] = []
        if path.is_file():
            self.paths = [path]
        elif path.is_dir():
            self.paths = sorted(path.glob("*.json"))
        else:
            raise FileNotFoundError(path)

        self.pairs: List[QAPair] = []
        self.load()

    def load(self) -> None:
        for file_path in self.paths:
            if not file_path.exists():
                raise FileNotFoundError(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                if isinstance(item, dict):
                    q_text = item.get("question", {}).get("text", "")
                    a_text = item.get("answer", {}).get("text", "")
                    q_tokens = item.get("question", {}).get("tokens") or (analyze(q_text) if q_text else [])
                    a_tokens = item.get("answer", {}).get("tokens") or (analyze(a_text) if a_text else [])
                    concepts = item.get("concepts")
                    if not concepts:
                        concepts = extract_concepts(q_tokens + a_tokens)
                    domain = item.get("domain") or infer_domain(q_tokens + a_tokens)
                    if q_text.strip() == a_text.strip():
                        q_text = ""
                        q_tokens = []
                elif isinstance(item, str):
                    q_text = ""
                    a_text = item
                    q_tokens = []
                    a_tokens = analyze(item)
                    concepts = extract_concepts(a_tokens)
                    domain = infer_domain(a_tokens)
                else:
                    continue
                if not q_tokens:
                    q_tokens = a_tokens[:]
                if not a_tokens:
                    continue
                self.pairs.append(
                    QAPair(
                        question=q_text,
                        answer=a_text,
                        tokens_q=q_tokens,
                        tokens_a=a_tokens,
                        concepts=concepts,
                        domain=domain,
                    )
                )
        if len(self.pairs) < 100:
            print(
                f"Warning: dataset has only {len(self.pairs)} pairs (<100)."
            )

    def to_dataframe(self) -> Any:
        """Convert dataset to pandas DataFrame or fallback object."""

        records = [
            {
                "question": p.question,
                "answer": p.answer,
                "concepts": ",".join(p.concepts),
                "domain": p.domain,
            }
            for p in self.pairs
        ]
        if pd is None:
            class Dummy:
                def __init__(self, rec):
                    self._rec = rec

                @property
                def empty(self) -> bool:
                    return not self._rec

            return Dummy(records)
        return pd.DataFrame(records)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        pair = self.pairs[idx]
        return pair.question, pair.answer
