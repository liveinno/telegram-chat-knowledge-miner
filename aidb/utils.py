import json
import os
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json_atomic(path: str, data: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def load_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str, items: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")


def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def token_estimate(text: str) -> int:
    # универсальная оценка: ~ 4 символа на токен
    if not text:
        return 0
    return max(1, len(text) // 4)


def hash_fact(statement: str, sources: List[Dict[str, str]], category: Optional[str] = None) -> str:
    src_ids = ",".join(sorted(s.get("source_id", "") for s in sources))
    basis = f"{statement.strip()}|{category or ''}|{src_ids}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


@dataclass
class Checkpoint:
    last_id: int = 0
    seen_hashes: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["seen_hashes"] is None:
            d["seen_hashes"] = []
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Checkpoint":
        return Checkpoint(
            last_id=int(d.get("last_id", 0)),
            seen_hashes=list(d.get("seen_hashes", [])),
        )


class CheckpointManager:
    def __init__(self, state_dir: str, state_file: str = "checkpoint.json") -> None:
        self.state_dir = state_dir
        self.state_path = os.path.join(state_dir, state_file)
        ensure_dir(state_dir)

    def load(self) -> Checkpoint:
        data = load_json(self.state_path, default=None)
        if data is None:
            return Checkpoint(last_id=0, seen_hashes=[])
        return Checkpoint.from_dict(data)

    def save(self, cp: Checkpoint) -> None:
        save_json_atomic(self.state_path, cp.to_dict())
