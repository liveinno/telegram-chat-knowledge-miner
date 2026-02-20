import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from .prompting import build_repair_messages


class CloudRuBackend:
    """
    Клиент Cloud.ru FM (OpenAI-совместимый /v1/chat/completions).
    Читает настройки из .env и/или переменных окружения:

      - CLOUDRU_API_BASE (например: https://api.cloud.ru/v1)
      - CLOUDRU_API_KEY  (Bearer токен)

    Параметры моделей задаются через primary_model / fallback_model.
    """

    def __init__(
        self,
        primary_model: str,
        fallback_model: Optional[str] = None,
        dotenv_path: Optional[str] = None,
        timeout: float = 60.0,
        retries: int = 2,
        backoff: float = 1.5,
    ) -> None:
        # Подгрузим .env (если указан) и по умолчанию из рабочей директории проекта
        if dotenv_path and os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
        # Также попробуем AIdb/.env относительно CWD
        default_env = os.path.join(os.getcwd(), "AIdb", ".env")
        if os.path.exists(default_env):
            load_dotenv(default_env)

        api_base = os.environ.get("CLOUDRU_API_BASE") or os.environ.get("CLOUD_RU_API_BASE") or ""
        self.api_base = api_base.rstrip("/")
        self.api_key = os.environ.get("CLOUDRU_API_KEY") or os.environ.get("CLOUD_RU_API_KEY") or ""
        if not self.api_base:
            # Попробуем дефолт (если в .env не задан)
            self.api_base = "https://api.cloud.ru/v1"
        if not self.api_key:
            raise RuntimeError("Не задан CLOUDRU_API_KEY в окружении/.env")

        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _chat_once(
        self,
        model: str,
        messages: List[Dict[str, str]],
        gen_max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Возврат: (text, error_message)
        """
        url = f"{self.api_base}/chat/completions" if not self.api_base.endswith("/v1") else f"{self.api_base}/chat/completions"
        # Если api_base уже оканчивается на /v1 — оставим как есть.
        # Дополнительно поддержим случай, когда api_base указывает на корень без /v1.
        if self.api_base.endswith("/v1"):
            url = f"{self.api_base}/chat/completions"
        else:
            # Попробуем добавить /v1
            url = f"{self.api_base}/v1/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(gen_max_tokens),
            # Запросим строгий JSON-объект на стороне модели, если поддерживается
            "response_format": {"type": "json_object"},
        }

        try:
            resp = self.session.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        except Exception as e:
            return None, f"request_error: {e!r}"

        if resp.status_code != 200:
            return None, f"http_{resp.status_code}: {resp.text[:500]}"

        try:
            data = resp.json()
        except Exception as e:
            return None, f"bad_json: {e!r} body={resp.text[:500]}"

        # Попробуем извлечь OpenAI-компатибельный контент
        try:
            text = data["choices"][0]["message"]["content"]
            return str(text), None
        except Exception:
            # В некоторых реализациях может быть иной формат
            # Попробуем generic поля
            for k in ("output", "text", "content"):
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    return v, None
            return None, f"no_text_in_response: keys={list(data.keys())}"

    def _chat_with_retries(
        self,
        model: str,
        messages: List[Dict[str, str]],
        gen_max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Tuple[Optional[str], Optional[str]]:
        delay = 0.5
        for attempt in range(1, self.retries + 1):
            text, err = self._chat_once(model, messages, gen_max_tokens, temperature, top_p)
            if err is None and text is not None:
                return text, None
            if attempt < self.retries:
                time.sleep(delay)
                delay *= self.backoff
        return None, err or "unknown_error"

    def _parse_json(self, s: str) -> Optional[Dict[str, Any]]:
        s = (s or "").strip()
        if not s:
            return None
        # На случай, если модель обернёт в кодблоки
        if s.startswith("```"):
            s = s.strip("`").strip()
            # Уберём возможный префикс типа json
            if s.lower().startswith("json"):
                s = s[4:].strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
        return None

    def generate(
        self,
        sources: List[Dict[str, Any]],
        messages: List[Dict[str, str]],
        gen_max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Dict[str, Any]:
        """
        Основной вызов: отправляем messages, парсим JSON. При неудаче —
        пытаемся сделать «ремонт» через build_repair_messages(bad_output).
        Также предусмотрен fallback-модель.
        """
        # 1) primary
        text, err = self._chat_with_retries(
            self.primary_model, messages, gen_max_tokens, temperature, top_p
        )
        parsed = self._parse_json(text or "")
        if parsed is not None and isinstance(parsed.get("facts"), list):
            # Если primary вернул корректный JSON, но без фактов — попробуем fallback-модель
            if self.fallback_model and not parsed.get("facts"):
                text_fb, err_fb = self._chat_with_retries(
                    self.fallback_model, messages, gen_max_tokens, temperature, top_p
                )
                parsed_fb = self._parse_json(text_fb or "")
                if parsed_fb is not None and isinstance(parsed_fb.get("facts"), list) and parsed_fb.get("facts"):
                    return parsed_fb
            return parsed

        # 2) repair на primary
        if text:
            repair_msgs = build_repair_messages(text)
            text2, err2 = self._chat_with_retries(
                self.primary_model, repair_msgs, gen_max_tokens, max(0.0, temperature - 0.1), top_p
            )
            parsed2 = self._parse_json(text2 or "")
            if parsed2 is not None and isinstance(parsed2.get("facts"), list):
                if self.fallback_model and not parsed2.get("facts"):
                    text_fb2, err_fb2 = self._chat_with_retries(
                        self.fallback_model, messages, gen_max_tokens, temperature, top_p
                    )
                    parsed_fb2 = self._parse_json(text_fb2 or "")
                    if parsed_fb2 is not None and isinstance(parsed_fb2.get("facts"), list) and parsed_fb2.get("facts"):
                        return parsed_fb2
                return parsed2

        # 3) fallback (если задан)
        if self.fallback_model:
            text3, err3 = self._chat_with_retries(
                self.fallback_model, messages, gen_max_tokens, temperature, top_p
            )
            parsed3 = self._parse_json(text3 or "")
            if parsed3 is not None and isinstance(parsed3.get("facts"), list):
                return parsed3

            # 4) repair на fallback
            if text3:
                repair_msgs2 = build_repair_messages(text3)
                text4, err4 = self._chat_with_retries(
                    self.fallback_model, repair_msgs2, gen_max_tokens, max(0.0, temperature - 0.1), top_p
                )
                parsed4 = self._parse_json(text4 or "")
                if parsed4 is not None and isinstance(parsed4.get("facts"), list):
                    return parsed4

        # Если не удалось — вернём пустую структуру
        return {"facts": []}
