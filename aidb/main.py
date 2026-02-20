import argparse
import json
import sys

from .knowledge_builder import BuilderConfig, KnowledgeBuilder


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aidb-knowledge-builder",
        description="Итеративная сборка базы знаний из SQLite при помощи универсального промпта и батчинга.",
    )
    # Источник данных
    p.add_argument("--db-path", required=True, help="Путь к SQLite базе с таблицей сообщений")
    p.add_argument("--table", default="messages", help="Имя таблицы с сообщениями (по умолчанию: messages)")
    p.add_argument("--limit-rows", type=int, default=None, help="Ограничение количества строк для выборки")

    # Поведение и артефакты
    p.add_argument("--output-dir", default="out", help="Каталог для артефактов (knowledge.jsonl, raw/)")
    p.add_argument("--state-dir", default=".state", help="Каталог состояния (чекпоинты)")
    p.add_argument("--state-file", default="checkpoint.json", help="Имя файла состояния чекпоинта")
    p.add_argument("--no-resume", action="store_true", help="Игнорировать чекпоинт и начать заново")

    # Генерация (для mock/LLM)
    p.add_argument("--backend", default="mock", choices=["mock", "cloudru"], help="Бэкенд генерации (mock|cloudru)")
    p.add_argument("--gen-max-tokens", type=int, default=1024, help="Ограничение токенов вывода")
    p.add_argument("--temperature", type=float, default=0.2, help="Температура выборки")
    p.add_argument("--top-p", type=float, default=0.95, help="Top-p сэмплирование")
    p.add_argument("--primary-model", default="openai/gpt-oss-120b", help="Основная модель Cloud.ru")
    p.add_argument("--fallback-model", default="Qwen/Qwen3-235B-A22B-Instruct-2507", help="Резервная модель Cloud.ru (fallback)")
    p.add_argument("--dotenv-path", default=None, help="Путь к .env с CLOUDRU_API_KEY/CLOUDRU_API_BASE")

    # Бюджеты контекста
    p.add_argument("--context-budget", type=int, default=180_000, help="Бюджет токенов контекста (~200k)")
    p.add_argument("--per-item-overhead", type=int, default=400, help="Оверхед токенов на элемент")

    # Подсказки
    p.add_argument("--hints", default="", help="Дополнительные подсказки (доменно-нейтральные)")

    # Фильтры выборки
    p.add_argument("--filter-topic-id", type=int, default=None, help="Обрабатывать только указанный topic_id")
    p.add_argument("--filter-topic-title-contains", default=None, help="Фильтр по подстроке в topic_title (LIKE %%...%%)")
    p.add_argument("--min-text-len", type=int, default=None, help="Минимальная длина текста сообщения (исключить пустые/медиа)")

    return p


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = BuilderConfig(
        gen_max_tokens=args.gen_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        context_budget=args.context_budget,
        per_item_overhead=args.per_item_overhead,
        output_dir=args.output_dir,
        state_dir=args.state_dir,
        state_file=args.state_file,
        hints=(args.hints or "").strip(),
        backend=args.backend,
        primary_model=args.primary_model,
        fallback_model=args.fallback_model,
        dotenv_path=args.dotenv_path,
    )

    builder = KnowledgeBuilder(cfg)

    result = builder.run(
        db_path=args.db_path,
        table=args.table,
        limit_rows=args.limit_rows,
        resume=not args.no_resume,
        filter_topic_id=args.filter_topic_id,
        filter_topic_title_contains=args.filter_topic_title_contains,
        min_text_len=args.min_text_len,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
