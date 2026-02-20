import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import sys
import time

from .utils import (
    ensure_dir,
    append_jsonl,
    now_ts,
    token_estimate,
    hash_fact,
    Checkpoint,
    CheckpointManager,
    save_json_atomic,
)
from .prompting import build_messages
from .extract import collect_items_from_db, count_messages
from .cloudru import CloudRuBackend


@dataclass
class BuilderConfig:
    # Параметры генерации (зарезервированы для реального LLM-клиента)
    gen_max_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.95

    # Бюджеты контекста
    context_budget: int = 180_000
    per_item_overhead: int = 400

    # Артефакты
    output_dir: str = "out"
    state_dir: str = ".state"
    state_file: str = "checkpoint.json"

    # Поведение
    hints: str = ""
    backend: str = "mock"  # 'mock' | 'cloudru'
    primary_model: str = "openai/gpt-oss-120b"
    fallback_model: Optional[str] = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    dotenv_path: Optional[str] = None


class MockBackend:
    """
    Простая доменно-нейтральная имитация LLM.
    Из каждого источника пытается извлечь одно короткое утверждение.
    """

    SENT_SPLIT = re.compile(r"[.!?]\s+")

    def _first_sentence(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        parts = self.SENT_SPLIT.split(text)
        sent = parts[0].strip() if parts else text
        # ограничим по словам ~60
        words = sent.split()
        if len(words) > 60:
            sent = " ".join(words[:60])
        return sent

    def generate(self, sources: List[Dict[str, Any]], gen_max_tokens: int, temperature: float, top_p: float, messages: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        facts: List[Dict[str, Any]] = []
        for item in sources:
            sid = item.get("source_id", "")
            text = item.get("text", "")
            sent = self._first_sentence(text)
            if not sent or not sid:
                continue
            # Короткая цитата (до ~200 символов)
            quote = text[:200].strip() if text else ""
            fact: Dict[str, Any] = {
                "statement": sent,
                "sources": [{"source_id": sid, "quote": quote}] if quote else [{"source_id": sid}],
                # category опциональна — не заполняем в mock
            }
            facts.append(fact)
        return {"facts": facts}


class KnowledgeBuilder:
    def __init__(self, config: BuilderConfig) -> None:
        self.cfg = config
        # Подготовка директорий
        self.art_out_dir = os.path.abspath(self.cfg.output_dir)
        self.raw_dir = os.path.join(self.art_out_dir, "raw")
        ensure_dir(self.raw_dir)
        self.state_dir_abs = os.path.abspath(self.cfg.state_dir)
        ensure_dir(self.state_dir_abs)

        # Бэкенд
        if self.cfg.backend == "mock":
            self.backend = MockBackend()
        elif self.cfg.backend == "cloudru":
            self.backend = CloudRuBackend(
                primary_model=self.cfg.primary_model,
                fallback_model=self.cfg.fallback_model,
                dotenv_path=self.cfg.dotenv_path,
            )
        else:
            self.backend = MockBackend()

        # Чекпоинты
        self.cp_mgr = CheckpointManager(self.state_dir_abs, self.cfg.state_file)
        self.cp: Optional[Checkpoint] = None

        # Файлы артефактов
        self.knowledge_path = os.path.join(self.art_out_dir, "knowledge.jsonl")

        # Статистика
        self.batches = 0
        self.accepted_facts = 0
        self.empty_batches = 0
        self.errors = 0

        # Прогресс
        self.total_to_process = 0
        self.processed_count = 0
        self.start_time = None

    def _batch_budget_ok(self, current_tokens: int, next_item_text: str) -> bool:
        add = self.cfg.per_item_overhead + token_estimate(next_item_text)
        return (current_tokens + add) <= self.cfg.context_budget

    def _finalize_and_run_batch(
        self,
        batch: List[Dict[str, Any]],
    ) -> Tuple[int, int]:
        """
        Возвращает (accepted_in_batch, errors_in_batch)
        """
        if not batch:
            return (0, 0)

        # Подготовим источники
        sources = [{"source_id": it["source_id"], "text": it["text"]} for it in batch]

        # Сформируем prompt (для сохранения в raw)
        messages = build_messages(sources, hints=self.cfg.hints)

        # Диапазон текущего батча (для логирования прогресса)
        first_id = batch[0]["id"]
        last_id = batch[-1]["id"]
        self._log(f"[processing] диапазон: {first_id}-{last_id}")

        # Вызов модели
        model_output = self.backend.generate(
            sources=sources,
            messages=messages,
            gen_max_tokens=self.cfg.gen_max_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
        )

        # Сохранение сырого вывода
        raw_payload = {
            "ts": now_ts(),
            "range": {"first_id": first_id, "last_id": last_id},
            "messages": messages,
            "model_params": {
                "gen_max_tokens": self.cfg.gen_max_tokens,
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
            },
            "output": model_output,
        }
        raw_name = f"{now_ts()}_{first_id}-{last_id}.json"
        save_json_atomic(os.path.join(self.raw_dir, raw_name), raw_payload)

        # Валидация/приёмка фактов
        accepted = self._accept_facts(model_output, batch)

        # При пустой приёмке попробуем второй проход с уточняющими подсказками
        if accepted == 0:
            retry_hints = (self.cfg.hints + "\nЕсли есть хотя бы один проверяемый факт (число/дата/процент, ссылка на документ/версию/изменение, условие/запрет/допустимость), извлеки его. Если фактов нет — верни {\"facts\": []}.").strip()
            messages_retry = build_messages(sources, hints=retry_hints)
            model_output_retry = self.backend.generate(
                sources=sources,
                messages=messages_retry,
                gen_max_tokens=self.cfg.gen_max_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
            )
            # сохраним сырой вывод ретрая
            raw_payload_retry = {
                "ts": now_ts(),
                "range": {"first_id": first_id, "last_id": last_id},
                "messages": messages_retry,
                "model_params": {
                    "gen_max_tokens": self.cfg.gen_max_tokens,
                    "temperature": self.cfg.temperature,
                    "top_p": self.cfg.top_p,
                },
                "output": model_output_retry,
                "retry": True,
            }
            raw_name_retry = f"{now_ts()}_{first_id}-{last_id}_retry.json"
            save_json_atomic(os.path.join(self.raw_dir, raw_name_retry), raw_payload_retry)

            accepted = self._accept_facts(model_output_retry, batch)

        # Продвинем чекпоинт
        if self.cp:
            self.cp.last_id = last_id
            self.cp_mgr.save(self.cp)

        self.batches += 1
        if accepted == 0:
            self.empty_batches += 1
        self.accepted_facts += accepted

        # Обновим прогресс
        self.processed_count += len(batch)
        self._update_progress(first_id, last_id)

        return (accepted, 0)

    def _accept_facts(self, output: Any, batch: List[Dict[str, Any]]) -> int:
        if not isinstance(output, dict):
            return 0
        facts = output.get("facts")
        if not isinstance(facts, list):
            return 0

        # Быстрый доступ к текстам по source_id
        text_by_sid = {it["source_id"]: it["text"] for it in batch}
        item_by_sid = {it["source_id"]: it for it in batch}
        known_sids = set(text_by_sid.keys())

        accepted_items: List[Dict[str, Any]] = []
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            statement = fact.get("statement")
            sources = fact.get("sources")
            category = fact.get("category") if isinstance(fact.get("category"), str) else None

            if not isinstance(statement, str) or not statement.strip():
                continue
            if not isinstance(sources, list) or not sources:
                continue

            # Валидация источников
            valid_sources: List[Dict[str, str]] = []
            ok = False
            for s in sources:
                if not isinstance(s, dict):
                    continue
                sid = s.get("source_id")
                quote = s.get("quote")
                if not isinstance(sid, str) or not re.match(r"^msg:\d+$", sid):
                    continue
                if sid not in known_sids:
                    continue
                if quote is not None and not isinstance(quote, str):
                    continue
                # Если цитата есть — проверим, что она встречается в исходном тексте (простая проверка)
                if isinstance(quote, str) and quote.strip():
                    if quote.strip() not in text_by_sid[sid]:
                        # Мягкая проверка: проигнорируем quote, но оставим источник
                        valid_sources.append({"source_id": sid})
                        ok = True
                        continue
                valid_sources.append({"source_id": sid} if not quote else {"source_id": sid, "quote": quote})
                ok = True

            if not ok or not valid_sources:
                continue

            # Дедуп по стабильному хэшу
            h = hash_fact(statement, valid_sources, category)
            if self.cp and h in self.cp.seen_hashes:
                continue

            # Соберём дополнительные метаданные: список id источников, даты, топики
            sid_nums = []
            dates = []
            topic_ids = []
            topic_titles = []
            for s in valid_sources:
                sid = s.get("source_id", "")
                try:
                    num = int(sid.split(":")[1])
                    sid_nums.append(num)
                except Exception:
                    pass
                it = item_by_sid.get(sid) or {}
                if "date" in it and it["date"]:
                    dates.append(str(it["date"]))
                if "topic_id" in it and it["topic_id"] is not None:
                    try:
                        topic_ids.append(int(it["topic_id"]))
                    except Exception:
                        pass
                if "topic_title" in it and it["topic_title"]:
                    topic_titles.append(str(it["topic_title"]))
            # Уникализация и сортировка
            sid_nums_uniq = sorted(set(sid_nums))
            dates_uniq = sorted(set(dates))
            topic_ids_uniq = sorted(set(topic_ids))
            topic_titles_uniq = sorted(set(topic_titles))
            meta_obj = {
                "ts": now_ts(),
                "source_ids_csv": ",".join(str(x) for x in sid_nums_uniq) if sid_nums_uniq else "",
            }
            if dates_uniq:
                meta_obj["dates"] = dates_uniq
                meta_obj["date_min"] = dates_uniq[0]
                meta_obj["date_max"] = dates_uniq[-1]
            if len(topic_ids_uniq) == 1:
                meta_obj["topic_id"] = topic_ids_uniq[0]
            elif topic_ids_uniq:
                meta_obj["topic_ids"] = topic_ids_uniq
            if len(topic_titles_uniq) == 1:
                meta_obj["topic_title"] = topic_titles_uniq[0]
            elif topic_titles_uniq:
                meta_obj["topic_titles"] = topic_titles_uniq

            accepted_items.append(
                {
                    "statement": statement.strip(),
                    "sources": valid_sources,
                    **({"category": category} if category else {}),
                    "hash": h,
                    "meta": meta_obj,
                }
            )

        if not accepted_items:
            return 0

        # Запишем
        append_jsonl(self.knowledge_path, accepted_items)

        # Обновим seen_hashes
        if self.cp:
            for it in accepted_items:
                h = it["hash"]
                if h not in self.cp.seen_hashes:
                    self.cp.seen_hashes.append(h)
            self.cp_mgr.save(self.cp)

        return len(accepted_items)

    def _log(self, msg: str) -> None:
        try:
            sys.stderr.write(str(msg) + "\n")
            sys.stderr.flush()
        except Exception:
            pass

    def _update_progress(self, first_id: int, last_id: int) -> None:
        if not self.start_time or self.total_to_process <= 0:
            return
        done = min(self.processed_count, self.total_to_process)
        total = self.total_to_process
        remaining = max(total - done, 0)
        elapsed = max(time.time() - self.start_time, 0.0)
        eta_min = (elapsed / done * remaining / 60.0) if done > 0 else 0.0
        pct = (done / total * 100.0) if total > 0 else 0.0
        bar_w = 30
        filled = int(bar_w * (done / total)) if total > 0 else 0
        if filled > bar_w:
            filled = bar_w
        bar = "[" + "#" * filled + "." * (bar_w - filled) + "]"
        self._log(f"[progress] {bar} {done}/{total} ({pct:.1f}%) | ост.: {remaining} | диапазон: {first_id}-{last_id} | ETA: {eta_min:.1f} мин")

    def run(
        self,
        db_path: str,
        table: str = "messages",
        limit_rows: Optional[int] = None,
        resume: bool = True,
        filter_topic_id: Optional[int] = None,
        filter_topic_title_contains: Optional[str] = None,
        min_text_len: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Читаем/инициализируем чекпоинт
        self.cp = self.cp_mgr.load()
        start_after = self.cp.last_id if (resume and self.cp) else 0
        if not resume and self.cp:
            self.cp.last_id = 0
            self.cp.seen_hashes = []
            self.cp_mgr.save(self.cp)
            start_after = 0

        # Инициализация прогресса
        self.total_to_process = count_messages(
            db_path=db_path,
            table=table,
            start_after_id=start_after,
            limit_rows=limit_rows,
            filter_topic_id=filter_topic_id,
            filter_topic_title_contains=filter_topic_title_contains,
            min_text_len=min_text_len,
        )
        self.processed_count = 0
        self.start_time = time.time()
        self._log(f"[progress] total={self.total_to_process} | start_after={start_after} | limit={limit_rows or '∞'}")

        batch: List[Dict[str, Any]] = []
        tokens_used = 0

        def flush_batch():
            nonlocal batch, tokens_used
            if not batch:
                return
            try:
                self._finalize_and_run_batch(batch)
            except Exception:
                self.errors += 1
            finally:
                batch = []
                tokens_used = 0

        for item in collect_items_from_db(
            db_path=db_path,
            table=table,
            start_after_id=start_after,
            limit_rows=limit_rows,
            filter_topic_id=filter_topic_id,
            filter_topic_title_contains=filter_topic_title_contains,
            min_text_len=min_text_len,
        ):
            t = item.get("text", "")
            add_tokens = self.cfg.per_item_overhead + token_estimate(t)
            if tokens_used > 0 and (tokens_used + add_tokens) > self.cfg.context_budget:
                flush_batch()
            batch.append(item)
            tokens_used += add_tokens

        # финальный батч
        flush_batch()

        # Итоговая статистика
        result = {
            "batches": self.batches,
            "accepted_facts": self.accepted_facts,
            "empty_batches": self.empty_batches,
            "errors": self.errors,
            "last_id": self.cp.last_id if self.cp else 0,
            "output_dir": self.art_out_dir,
            "raw_dir": self.raw_dir,
            "knowledge_path": self.knowledge_path,
            "state_path": self.cp_mgr.state_path,
        }
        # Сохраним сводку
        save_json_atomic(os.path.join(self.art_out_dir, f"summary_{now_ts()}.json"), result)
        return result
