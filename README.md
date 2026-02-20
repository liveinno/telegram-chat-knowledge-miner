# Telegram Chat Knowledge Miner

Итеративный универсальный экстрактор уникальных знаний (факты, определения, методики, ограничения и т.п.) из SQLite базы Telegram-чатов с поддержкой:
- OpenAI-совместимого Cloud.ru FM API (`/v1/chat/completions`)
- Большого контекста (~200k токенов, автоматический батчинг по бюджету)
- Строгого JSON-вывода по схеме + валидация и repair-циклы
- Дедупликации по стабильному хэшу и чекпоинтов (resume)
- Fallback-модели при нулевой приёмке фактов
- Инлайнинга текста из документов (txt/pdf/docx) с опциональным `--media-root`
- Доменно-нейтрального промпта (подходит для любых тем)

## Структура проекта

```
aidb/
├── main.py              # CLI интерфейс
├── knowledge_builder.py # Основной цикл, батчинг, вызовы модели
├── prompting.py         # Промпты и JSON Schema
├── extract.py           # Извлечение данных из SQLite
├── cloudru.py           # Клиент Cloud.ru FM API
├── utils.py             # Утилиты: JSONL, чекпоинты, хэширование
└── __init__.py
```

## Артефакты

- `out/knowledge.jsonl` — принятые факты (по одному JSON-объекту на строку)
- `out/raw/*.json` — сырые попытки генерации / ремонта / фоллбэка (для аудита)
- `.state/checkpoint.json` — состояние: `last_id`, `seen_hashes`
- В консоли печатается сводка JSON с метриками

## Установка

1) Python 3.10+
2) Установить зависимости:
```bash
pip install -r requirements.txt
```

Рекомендуется изолированное окружение (venv/conda/poetry).

## Настройка

Создайте файл `.env` на основе `.env.example`:

```bash
cp .env.example .env
# Отредактируйте .env, добавив ваш API ключ
```

Поддерживаются обе формы переменных:
```
CLOUDRU_API_BASE=https://api.cloud.ru/v1
CLOUDRU_API_KEY=your_api_key_here

# или альтернативная форма:
CLOUD_RU_API_BASE=https://api.cloud.ru/v1
CLOUD_RU_API_KEY=your_api_key_here
```

## Ожидаемый формат данных

SQLite база с таблицей сообщений. Поддерживаемые колонки:
- `id` — идентификатор сообщения
- `text` | `message` | `content` | `body` — текст сообщения
- `date` — дата сообщения (опционально)
- `topic_id` — ID топика форума (опционально)
- `topic_title` — название топика (опционально)

## Запуск (CLI)

```bash
python -m aidb.main --help
```

### Основные параметры

- `--db-path` (обяз.) — путь к SQLite базе
- `--table` — имя таблицы (по умолчанию: `messages`)
- `--limit-rows` — ограничение выборки
- `--output-dir` — каталог для артефактов (по умолчанию: `out`)
- `--state-dir` — каталог чекпоинтов (по умолчанию: `.state`)
- `--no-resume` — начать заново
- `--backend` — `mock` | `cloudru` (по умолчанию: `mock`)
- `--primary-model` — основная модель (по умолчанию: `openai/gpt-oss-120b`)
- `--fallback-model` — резервная модель
- `--dotenv-path` — путь к `.env` файлу
- `--filter-topic-id` — фильтр по ID топика
- `--filter-topic-title-contains` — фильтр по названию топика
- `--min-text-len` — минимальная длина текста

### Примеры запуска

**Тестовый прогон (25 сообщений):**
```bash
python -m aidb.main \
  --backend cloudru \
  --db-path /path/to/your/telegram.db \
  --table messages \
  --limit-rows 25 \
  --output-dir out_test \
  --state-dir .state_test \
  --no-resume
```

**Полный прогон с фильтром по топику:**
```bash
python -m aidb.main \
  --backend cloudru \
  --db-path /path/to/your/telegram.db \
  --table messages \
  --filter-topic-title-contains "Важное" \
  --min-text-len 10 \
  --output-dir out_filtered \
  --state-dir .state_filtered \
  --hints "Извлекай определения систем, ограничения, условия, версии"
```

**Продолжение с чекпоинта:**
```bash
python -m aidb.main \
  --backend cloudru \
  --db-path /path/to/your/telegram.db \
  --table messages \
  --output-dir out_run \
  --state-dir .state_run
```

## Формат результата

Каждый факт в `knowledge.jsonl`:
```json
{
  "statement": "Краткое утверждение (до ~60 слов)",
  "sources": [
    {"source_id": "msg:123", "quote": "Цитата из текста"}
  ],
  "category": "Опциональная категория",
  "hash": "sha256_hash",
  "meta": {
    "ts": "20240115-123000",
    "source_ids_csv": "123,124",
    "date_min": "2024-01-01",
    "date_max": "2024-01-15",
    "topic_id": 5
  }
}
```

## Метрики

После каждого запуска в stdout печатается JSON-резюме:
```json
{
  "batches": 3,
  "accepted_facts": 42,
  "empty_batches": 0,
  "errors": 0,
  "last_id": 2125
}
```

## Рекомендации

- Снижайте `--temperature` (0.1) для более стабильного JSON
- Используйте `--hints` для тонкой настройки экстракции
- Анализируйте `out/raw/*.json` для отладки
- Для чистого перезапуска используйте новые `--output-dir`/`--state-dir` и `--no-resume`

## Типичные проблемы

- 401/403: неверный API ключ (проверьте `.env`)
- 404/405: проверьте корректность `API_BASE`
- Валидация JSON падает: смотрите `out/raw/*.json`

## Лицензия

CC BY-NC 4.0
