"""Клиент GigaChat API.

Реализует официальную схему работы:
1. POST /api/v2/oauth — получает access token в обмен на ключ авторизации.
2. POST /v1/chat/completions — отправляет сообщения и получает ответ модели.

Документация: https://developers.sber.ru/docs/ru/gigachat/api/overview
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any

import requests
import urllib3

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.giga.chat/v1"
DEFAULT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
DEFAULT_SCOPE = "GIGACHAT_API_PERS"
DEFAULT_MODEL = "GigaChat-2"


class GigaChatError(Exception):
    """Базовая ошибка при работе с GigaChat API."""


class GigaChatAuthError(GigaChatError):
    """Ошибка авторизации: не удалось получить access token."""


class GigaChatClient:
    def __init__(
        self,
        auth_key: str,
        *,
        scope: str = DEFAULT_SCOPE,
        base_url: str = DEFAULT_BASE_URL,
        oauth_url: str = DEFAULT_OAUTH_URL,
        model: str = DEFAULT_MODEL,
        verify_ssl: bool = True,
        ca_bundle: str | None = None,
        timeout: int = 60,
    ) -> None:
        if not auth_key:
            raise ValueError("Не передан ключ авторизации GigaChat")

        # На случай, если пользователь скопировал ключ вместе с префиксом.
        auth_key = auth_key.strip()
        if auth_key.lower().startswith("basic "):
            auth_key = auth_key[6:].strip()
        self._auth_key = auth_key
        self.scope = scope or DEFAULT_SCOPE
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.oauth_url = oauth_url or DEFAULT_OAUTH_URL
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout

        if ca_bundle:
            self._verify: bool | str = ca_bundle
        else:
            self._verify = verify_ssl
            if not verify_ssl:
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "ZerocoderVKBot/1.0",
                "Accept": "application/json",
            }
        )

        self._token: str | None = None
        self._expires_at_ms: int = 0
        self._token_lock = threading.Lock()

    @property
    def verify_ssl(self) -> bool | str:
        """Текущий режим проверки SSL (для диагностических логов)."""
        return self._verify

    # ------------------------------------------------------------------
    # Авторизация
    # ------------------------------------------------------------------
    def _get_access_token(self, force_refresh: bool = False) -> str:
        with self._token_lock:
            now_ms = int(time.time() * 1000)
            # Обновляем токен заранее, за минуту до истечения.
            if (
                not force_refresh
                and self._token
                and now_ms < self._expires_at_ms - 60_000
            ):
                return self._token

            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "RqUID": str(uuid.uuid4()),
                "Authorization": f"Basic {self._auth_key}",
            }
            try:
                response = self._session.post(
                    self.oauth_url,
                    headers=headers,
                    data={"scope": self.scope},
                    timeout=self.timeout,
                    verify=self._verify,
                )
            except (requests.RequestException, OSError) as exc:
                raise GigaChatAuthError(
                    f"Не удалось подключиться к серверу авторизации: {exc}"
                ) from exc

            if response.status_code != 200:
                raise GigaChatAuthError(
                    "GigaChat отклонил ключ авторизации "
                    f"(HTTP {response.status_code}): {_short(response.text)}"
                )

            try:
                data = response.json()
                token = data["access_token"]
            except (ValueError, KeyError) as exc:
                raise GigaChatAuthError(
                    "Сервер авторизации вернул неожиданный ответ"
                ) from exc

            expires_at = data.get("expires_at")
            if isinstance(expires_at, (int, float)) and expires_at:
                expires_at_ms = int(expires_at)
                # Разные версии API возвращают секунды или миллисекунды.
                if expires_at_ms < 100_000_000_000:
                    expires_at_ms *= 1000
            else:
                expires_at_ms = now_ms + 25 * 60 * 1000

            self._token = token
            self._expires_at_ms = expires_at_ms
            logger.debug("Получен новый access token GigaChat")
            return token

    def _invalidate_token(self) -> None:
        with self._token_lock:
            self._token = None
            self._expires_at_ms = 0

    # ------------------------------------------------------------------
    # Генерация ответа
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 900,
    ) -> str:
        """Отправляет историю сообщений и возвращает текст ответа."""
        if not messages:
            raise ValueError("Список сообщений пуст")

        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        last_error: Exception | None = None
        force_refresh = False
        for attempt in range(1, 4):
            token = self._get_access_token(force_refresh=force_refresh)
            force_refresh = False
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {token}",
            }

            try:
                response = self._session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    verify=self._verify,
                )
            except (requests.RequestException, OSError) as exc:
                last_error = GigaChatError(f"Сетевая ошибка GigaChat: {exc}")
                logger.warning("GigaChat: %s (попытка %s/3)", exc, attempt)
                if attempt < 3:
                    time.sleep(2 * attempt)
                    continue
                raise last_error from exc

            if response.status_code == 401 and attempt == 1:
                # Токен мог истечь или быть отозван — пробуем обновить.
                logger.info("GigaChat вернул 401, обновляю access token")
                self._invalidate_token()
                force_refresh = True
                continue

            if response.status_code == 429 or response.status_code >= 500:
                retry_after = _retry_after_seconds(response)
                last_error = GigaChatError(
                    f"GigaChat временно недоступен (HTTP {response.status_code})"
                )
                logger.warning(
                    "GigaChat HTTP %s (попытка %s/3), пауза %.1f с",
                    response.status_code,
                    attempt,
                    retry_after,
                )
                if attempt < 3:
                    time.sleep(retry_after)
                    continue
                raise last_error

            if response.status_code != 200:
                raise GigaChatError(
                    f"GigaChat вернул ошибку HTTP {response.status_code}: "
                    f"{_short(response.text)}"
                )

            try:
                data = response.json()
            except ValueError as exc:
                raise GigaChatError("GigaChat вернул ответ не в формате JSON") from exc

            text = _extract_content(data)
            if not text.strip():
                raise GigaChatError("GigaChat вернул пустой ответ")
            return text.strip()

        raise last_error or GigaChatError("Не удалось получить ответ GigaChat")

    def get_models(self) -> list[dict[str, Any]]:
        """Возвращает список доступных моделей (для диагностики)."""
        token = self._get_access_token()
        try:
            response = self._session.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {token}"},
                timeout=self.timeout,
                verify=self._verify,
            )
        except (requests.RequestException, OSError) as exc:
            raise GigaChatError(f"Сетевая ошибка GigaChat: {exc}") from exc

        if response.status_code != 200:
            raise GigaChatError(
                f"Не удалось получить список моделей: HTTP {response.status_code}"
            )
        try:
            return response.json().get("data", [])
        except ValueError as exc:
            raise GigaChatError("GigaChat вернул некорректный JSON") from exc

    def check_auth(self) -> None:
        """Проверяет ключ и доступность API без расходования токенов."""
        self.get_models()


def _extract_content(data: dict[str, Any]) -> str:
    """Достаёт текст ответа из разных вариантов структуры ответа API."""
    choices = data.get("choices")
    if not choices and isinstance(data.get("value"), dict):
        choices = data["value"].get("choices")
    if not choices:
        return ""

    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def _retry_after_seconds(response: requests.Response) -> float:
    value = response.headers.get("Retry-After")
    if value:
        try:
            return max(1.0, min(float(value), 30.0))
        except ValueError:
            pass
    return 3.0


def _short(text: str, limit: int = 300) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:limit] + ("…" if len(text) > limit else "")
