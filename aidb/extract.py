import sqlite3
from typing import Dict, Generator, Iterable, Optional


def iter_messages(
    db_path: str,
    table: str = "messages",
    start_after_id: int = 0,
    limit_rows: Optional[int] = None,
    filter_topic_id: Optional[int] = None,
    filter_topic_title_contains: Optional[str] = None,
    min_text_len: Optional[int] = None,
) -> Generator[Dict, None, None]:
    """
    Универсальный итератор по строкам SQLite.
    Ожидается как минимум схема: (id INTEGER PRIMARY KEY, text TEXT NOT NULL)

    Возвращает элементы вида:
      {"id": int, "text": str, "source_id": f"msg:{id}"}
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()

        # Проверим наличие колонок и выберем подходящее текстовое поле (универсально)
        cur.execute(f"PRAGMA table_info({table})")
        cols = [r["name"] for r in cur.fetchall()]
        if "id" not in cols:
            raise RuntimeError(f"Таблица {table} должна содержать колонку id. Найдены: {cols}")

        # Универсальный выбор текстовой колонки: text | message | content | body
        text_col = None
        for cand in ("text", "message", "content", "body"):
            if cand in cols:
                text_col = cand
                break
        if not text_col:
            raise RuntimeError(
                f"Таблица {table} должна содержать одну из текстовых колонок (text|message|content|body). Найдены: {cols}"
            )

        # Дополнительно детектируем служебные поля если присутствуют
        date_col = "date" if "date" in cols else None
        topic_id_col = "topic_id" if "topic_id" in cols else None
        topic_title_col = "topic_title" if "topic_title" in cols else None

        # Сформируем SELECT с опциональными колонками
        select_cols = [f"id", f"{text_col} AS text"]
        if date_col:
            select_cols.append(f"{date_col} AS date")
        if topic_id_col:
            select_cols.append(f"{topic_id_col} AS topic_id")
        if topic_title_col:
            select_cols.append(f"{topic_title_col} AS topic_title")

        # Порядок обхода от старых к новым: ORDER BY id ASC
        where = ["id > ?"]
        params = [start_after_id]
        if min_text_len is not None:
            where.append(f"length({text_col}) >= ?")
            params.append(int(min_text_len))
        if filter_topic_id is not None and topic_id_col:
            where.append(f"{topic_id_col} = ?")
            params.append(int(filter_topic_id))
        if filter_topic_title_contains and topic_title_col:
            where.append(f"{topic_title_col} LIKE ?")
            params.append(f"%{filter_topic_title_contains}%")
        sql = f"SELECT {', '.join(select_cols)} FROM {table} WHERE " + " AND ".join(where) + " ORDER BY id ASC"
        if limit_rows is not None:
            sql += " LIMIT ?"
            params.append(int(limit_rows))

        for row in cur.execute(sql, params):
            rid = int(row["id"])
            text = str(row["text"] or "")
            # Опциональные поля
            date_val = str(row["date"]) if ("date" in row.keys()) and (row["date"] is not None) else None
            topic_id_val = int(row["topic_id"]) if ("topic_id" in row.keys()) and (row["topic_id"] is not None) else None
            topic_title_val = str(row["topic_title"]) if ("topic_title" in row.keys()) and (row["topic_title"] is not None) else None

            item = {
                "id": rid,
                "text": text,
                "source_id": f"msg:{rid}",
            }
            if date_val is not None:
                item["date"] = date_val
            if topic_id_val is not None:
                item["topic_id"] = topic_id_val
            if topic_title_val is not None:
                item["topic_title"] = topic_title_val

            yield item
    finally:
        conn.close()


def collect_items_from_db(
    db_path: str,
    table: str = "messages",
    start_after_id: int = 0,
    limit_rows: Optional[int] = None,
    filter_topic_id: Optional[int] = None,
    filter_topic_title_contains: Optional[str] = None,
    min_text_len: Optional[int] = None,
) -> Iterable[Dict]:
    """
    Обёртка — на будущее можно расширять (например, слияние с документами),
    пока возвращает только сообщения из SQLite.
    """
    return iter_messages(
        db_path=db_path,
        table=table,
        start_after_id=start_after_id,
        limit_rows=limit_rows,
        filter_topic_id=filter_topic_id,
        filter_topic_title_contains=filter_topic_title_contains,
        min_text_len=min_text_len,
    )


def count_messages(
    db_path: str,
    table: str = "messages",
    start_after_id: int = 0,
    limit_rows: Optional[int] = None,
    filter_topic_id: Optional[int] = None,
    filter_topic_title_contains: Optional[str] = None,
    min_text_len: Optional[int] = None,
) -> int:
    """
    Подсчёт количества сообщений для обработки: id > start_after_id, с учётом фильтров.
    Если задан limit_rows, возвращается min(реального количества, limit_rows).
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Детектируем колонки, чтобы корректно сформировать условия
        cur.execute(f"PRAGMA table_info({table})")
        cols = [r["name"] for r in cur.fetchall()]
        # Универсальный выбор текстовой колонки
        text_col = None
        for cand in ("text", "message", "content", "body"):
            if cand in cols:
                text_col = cand
                break
        topic_id_col = "topic_id" if "topic_id" in cols else None
        topic_title_col = "topic_title" if "topic_title" in cols else None

        where = ["id > ?"]
        params = [start_after_id]
        if min_text_len is not None and text_col:
            where.append(f"length({text_col}) >= ?")
            params.append(int(min_text_len))
        if filter_topic_id is not None and topic_id_col:
            where.append(f"{topic_id_col} = ?")
            params.append(int(filter_topic_id))
        if filter_topic_title_contains and topic_title_col:
            where.append(f"{topic_title_col} LIKE ?")
            params.append(f"%{filter_topic_title_contains}%")

        sql = f"SELECT COUNT(*) FROM {table} WHERE " + " AND ".join(where)
        cur.execute(sql, params)
        total = int(cur.fetchone()[0] or 0)
        if limit_rows is not None:
            total = min(total, int(limit_rows))
        return total
    finally:
        conn.close()
