"""Чат-бот сообщества ВКонтакте для онлайн-университета Zerocoder.

Бот слушает сообщения сообщества через VK Long Poll API, передаёт диалог
в GigaChat API вместе с системным промптом Алины и базой знаний о курсах,
а затем отправляет ответ пользователю.

Запуск из корня проекта: python -m src
Все секреты берутся из .env (см. .env.example и docs/SETUP.md).
"""

from __future__ import annotations

import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
import vk_api
from dotenv import load_dotenv
from vk_api.bot_longpoll import VkBotEventType, VkBotLongPoll
from vk_api.keyboard import VkKeyboard
from vk_api.exceptions import ApiError

from .gigachat_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_OAUTH_URL,
    DEFAULT_SCOPE,
    GigaChatClient,
    GigaChatError,
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MANAGER_CONTACT = "care@zerocoder.ru, +7 (939) 328-38-12"
VK_MESSAGE_LIMIT = 3500
SESSION_TTL_SECONDS = 24 * 60 * 60


def env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def setup_logging() -> None:
    level_name = env("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        stream=sys.stdout,
        force=True,
    )
    # Отдельный файл лога в UTF-8: перенаправление stdout на Windows
    # ломает кодировку, а файл читается корректно на любой системе.
    file_handler = logging.FileHandler(
        PROJECT_ROOT / "bot.log", encoding="utf-8", delay=False
    )
    file_handler.setFormatter(logging.Formatter(fmt))
    file_handler.setLevel(level)
    logging.getLogger().addHandler(file_handler)
    # Не включаем DEBUG для HTTP-библиотек: иначе в консоль могут попасть
    # заголовки Authorization с токенами.
    for noisy_logger in (
        "urllib3",
        "requests",
        "http.client",
        "vk_api",
        "gigachat",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def read_text(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Не удалось прочитать {label}: {path} ({exc})") from exc


def build_system_prompt(knowledge_base: str, sales_form_url: str, manager: str) -> str:
    template = read_text(
        PROJECT_ROOT / "prompts" / "alina_system_prompt.md", "системный промпт"
    )

    if sales_form_url:
        form_value = sales_form_url
    else:
        form_value = (
            "(ссылка на форму временно недоступна — предложи связаться "
            f"с менеджером: {manager})"
        )

    prompt = (
        template.replace("{{KNOWLEDGE_BASE}}", knowledge_base)
        .replace("{{SALES_FORM_URL}}", form_value)
        .replace("{{MANAGER_CONTACT}}", manager)
    )

    if not sales_form_url:
        prompt += (
            "\n\nВАЖНО: ссылка на форму заявки не настроена. Не придумывай URL. "
            f"Если пользователь готов связаться — дай контакт: {manager}."
        )
    return prompt


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


GREETINGS = {
    "/start",
    "start",
    "старт",
    "начать",
    "привет",
    "привет!",
    "здравствуйте",
    "здравствуйте!",
    "здравствуй",
    "добрый день",
    "добрый день!",
    "добрый вечер",
    "добрый вечер!",
    "доброе утро",
    "доброе утро!",
}

HELP_COMMANDS = {"/help", "help", "помощь", "что ты умеешь", "что умеешь"}
RESET_COMMANDS = {"/reset", "reset", "сброс", "сбросить", "сбросить диалог"}

GREETING_TEXT = (
    "Здравствуйте! Я Алина, консультант онлайн-университета Zerocoder. "
    "Помогу подобрать программу по нейросетям, вайб-кодингу или новой "
    "ИИ-профессии.\n\n"
    "Подскажите, вы хотите освоить ИИ-инструменты для своей работы или "
    "сменить направление и начать зарабатывать?"
)

HELP_TEXT = (
    "Я консультирую по обучению в Zerocoder:\n"
    "• программы по нейросетям и ИИ-инструментам;\n"
    "• профессии: AI-инженер, вайб-кодер, промпт-инженер, автоматизатор;\n"
    "• курсы по Claude Code, n8n, GigaChat, китайским нейросетям;\n"
    "• абонементы, детское и корпоративное обучение;\n"
    "• форматы, сроки, стоимость и запись.\n\n"
    "Просто напишите свой вопрос. Команда /reset сбрасывает диалог."
)

RESET_TEXT = "Диалог сброшен. Чем могу помочь с выбором программы?"

FALLBACK_TEXT = (
    "Извините, у меня возникли технические сложности. "
    "Попробуйте, пожалуйста, написать ещё раз через минуту. "
    "Если вопрос срочный — свяжитесь с нами: {contact}."
)


class VkBot:
    def __init__(self) -> None:
        self.vk_token = env("VK_TOKEN")
        self.group_id_raw = env("VK_GROUP_ID")
        self.gigachat_key = env("GIGACHAT_AUTH_KEY")
        self.sales_form_url = env("SALES_FORM_URL")
        self.manager_contact = env("MANAGER_CONTACT", DEFAULT_MANAGER_CONTACT)
        self.max_history = env_int("MAX_HISTORY_MESSAGES", 12, minimum=2)
        self.fallback_text = FALLBACK_TEXT.format(contact=self.manager_contact)

        if self.sales_form_url and not self.sales_form_url.startswith(
            ("http://", "https://")
        ):
            logging.warning(
                "SALES_FORM_URL должен начинаться с http:// или https://. "
                "Кнопка заявки отключена, бот предложит контакт менеджера."
            )
            self.sales_form_url = ""

        missing = [
            name
            for name, value in (
                ("VK_TOKEN", self.vk_token),
                ("VK_GROUP_ID", self.group_id_raw),
                ("GIGACHAT_AUTH_KEY", self.gigachat_key),
            )
            if not value
        ]
        if missing:
            raise SystemExit(
                "Не заданы обязательные переменные окружения: "
                + ", ".join(missing)
                + ". Заполните файл .env по образцу .env.example."
            )

        try:
            self.group_id = int(self.group_id_raw.lstrip("-"))
        except ValueError as exc:
            raise SystemExit("VK_GROUP_ID должен быть числом без минуса.") from exc

        ca_bundle = env("GIGACHAT_CA_BUNDLE") or None
        if ca_bundle:
            # Относительный путь трактуем относительно папки проекта.
            ca_path = Path(ca_bundle).expanduser()
            if not ca_path.is_absolute():
                ca_path = PROJECT_ROOT / ca_path
            ca_bundle = str(ca_path)

        knowledge_base = read_text(
            PROJECT_ROOT / "data" / "knowledge_base.md", "база знаний"
        )
        self.system_prompt = build_system_prompt(
            knowledge_base, self.sales_form_url, self.manager_contact
        )

        self.gigachat = GigaChatClient(
            self.gigachat_key,
            scope=env("GIGACHAT_SCOPE", DEFAULT_SCOPE),
            base_url=env("GIGACHAT_BASE_URL", DEFAULT_BASE_URL),
            oauth_url=env("GIGACHAT_OAUTH_URL", DEFAULT_OAUTH_URL),
            model=env("GIGACHAT_MODEL", DEFAULT_MODEL),
            verify_ssl=env_bool("GIGACHAT_VERIFY_SSL", True),
            ca_bundle=ca_bundle,
        )

        self.vk_session = vk_api.VkApi(token=self.vk_token, api_version="5.199")
        self.vk = self.vk_session.get_api()
        self.sessions: dict[int, dict[str, Any]] = {}
        self.keyboard = self._build_keyboard()

    # ------------------------------------------------------------------
    # VK
    # ------------------------------------------------------------------
    def _build_keyboard(self) -> str | None:
        if not self.sales_form_url:
            return None
        keyboard = VkKeyboard(one_time=False)
        keyboard.add_openlink_button("Оставить заявку", link=self.sales_form_url)
        return keyboard.get_keyboard()

    def _check_vk(self) -> None:
        try:
            groups = self.vk.groups.getById(group_id=self.group_id)
            if isinstance(groups, dict):
                groups = groups.get("groups", [])
            if groups:
                name = groups[0].get("name", "без названия")
                logging.info("Подключение к ВК успешно. Сообщество: %s", name)
            else:
                logging.info("Подключение к ВК успешно.")
        except Exception as exc:
            raise SystemExit(
                "Не удалось подключиться к ВКонтакте. Проверьте VK_TOKEN, "
                "VK_GROUP_ID и права ключа доступа (сообщения сообщества). "
                f"Ответ ВК: {exc}"
            ) from exc

        # Диагностика: дошли ли сообщения до сообщества.
        try:
            count = self.vk.messages.getConversations(count=1).get("count", 0)
            logging.info("Диалогов в сообществе: %s", count)
            if count == 0:
                logging.info(
                    "Подсказка: сообществу ещё никто не писал. Проверьте, что "
                    "включены «Сообщения сообщества» и «Возможности бота» "
                    "(Управление → Сообщения)."
                )
        except Exception as exc:
            logging.warning("Не удалось получить список диалогов: %s", exc)

    def _send(self, peer_id: int, text: str, *, keyboard: str | None = None) -> None:
        for chunk in split_message(text, VK_MESSAGE_LIMIT):
            params: dict[str, Any] = {
                "peer_id": peer_id,
                "message": chunk,
                "random_id": random.randint(1, 2**31 - 1),
            }
            if keyboard:
                params["keyboard"] = keyboard
            try:
                self.vk.messages.send(**params)
            except ApiError as exc:
                logging.error("Не удалось отправить сообщение в ВК: %s", exc)
                if keyboard:
                    # Кнопка могла оказаться некорректной — пробуем без неё.
                    params.pop("keyboard", None)
                    try:
                        self.vk.messages.send(**params)
                        continue
                    except ApiError as retry_exc:
                        logging.error(
                            "Повторная отправка тоже не удалась: %s", retry_exc
                        )
                return

    def _set_typing(self, peer_id: int) -> None:
        try:
            self.vk.messages.setActivity(
                type="typing", peer_id=peer_id, group_id=self.group_id
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Диалог
    # ------------------------------------------------------------------
    def _history(self, user_id: int) -> list[dict[str, str]]:
        session = self.sessions.get(user_id)
        if not session:
            session = {"history": [], "last_seen": time.time()}
            self.sessions[user_id] = session
        session["last_seen"] = time.time()
        return session["history"]

    def _trim(self, history: list[dict[str, str]]) -> None:
        if len(history) > self.max_history:
            del history[: len(history) - self.max_history]

    def _cleanup_sessions(self) -> None:
        now = time.time()
        stale = [
            user_id
            for user_id, session in self.sessions.items()
            if now - session.get("last_seen", now) > SESSION_TTL_SECONDS
        ]
        for user_id in stale:
            self.sessions.pop(user_id, None)

    def _answer(self, user_id: int, text: str) -> str:
        """Возвращает текст ответа на сообщение пользователя."""
        command = normalize(text)
        history = self._history(user_id)

        if command in GREETINGS:
            history.clear()
            history.append({"role": "assistant", "content": GREETING_TEXT})
            return GREETING_TEXT

        if command in HELP_COMMANDS:
            return HELP_TEXT

        if command in RESET_COMMANDS:
            history.clear()
            return RESET_TEXT

        if command.startswith("/"):
            return HELP_TEXT

        if not text.strip():
            return (
                "Я пока умею работать только с текстом. Напишите, пожалуйста, "
                "вопрос словами — подскажу по программам Zerocoder."
            )

        history.append({"role": "user", "content": text[:4000]})
        self._trim(history)

        messages = [{"role": "system", "content": self.system_prompt}] + history
        try:
            reply = self.gigachat.chat(messages)
        except GigaChatError as exc:
            logging.error("Ошибка GigaChat: %s", exc)
            history.pop()
            return self.fallback_text

        history.append({"role": "assistant", "content": reply})
        self._trim(history)
        return reply

    def handle_message(self, event: Any) -> None:
        message = event.message
        text = (message.get("text") or "").strip()
        user_id = message.get("from_id")
        peer_id = message.get("peer_id") or user_id

        logging.info(
            "Событие message_new: from=%s peer=%s out=%s символов=%s",
            user_id,
            peer_id,
            message.get("out"),
            len(text),
        )

        if message.get("out"):
            return

        if not user_id or user_id <= 0:
            logging.info("Пропущено: некорректный from_id=%s", user_id)
            return

        self._set_typing(peer_id)
        reply = self._answer(user_id, text)
        self._send(peer_id, reply, keyboard=self.keyboard)

    def run(self) -> None:
        logging.info("=== Бот Алина запускается ===")
        logging.info("Модель GigaChat: %s", self.gigachat.model)
        logging.info("Scope: %s", self.gigachat.scope)
        logging.info("Проверка SSL: %s", self.gigachat.verify_ssl)
        logging.info(
            "Форма заявки: %s",
            "настроена" if self.sales_form_url else "не настроена",
        )

        self._check_vk()

        try:
            self.gigachat.check_auth()
            logging.info("GigaChat: авторизация успешна")
        except GigaChatError as exc:
            logging.error(
                "GigaChat: проверка авторизации не удалась — %s. "
                "Бот продолжит работу, но ответы могут быть недоступны.",
                exc,
            )

        try:
            longpoll = VkBotLongPoll(self.vk_session, self.group_id, wait=25)
        except Exception as exc:
            raise SystemExit(
                "Не удалось запустить Long Poll API ВКонтакте. Проверьте, что "
                "Long Poll включён в настройках сообщества и ключ имеет право "
                f"«Сообщения сообщества». Ответ ВК: {exc}"
            ) from exc
        logging.info("Слушаю сообщения сообщества...")

        counter = 0
        while True:
            try:
                for event in longpoll.listen():
                    if event.type != VkBotEventType.MESSAGE_NEW:
                        continue
                    try:
                        self.handle_message(event)
                    except Exception:
                        logging.exception("Ошибка обработки сообщения")

                    counter += 1
                    if counter % 100 == 0:
                        self._cleanup_sessions()
            except requests.exceptions.ReadTimeout:
                continue
            except KeyboardInterrupt:
                logging.info("Остановка бота")
                break
            except Exception as exc:
                logging.error("Ошибка Long Poll: %s. Переподключение через 5 с", exc)
                time.sleep(5)


def split_message(text: str, limit: int = VK_MESSAGE_LIMIT) -> list[str]:
    """Разбивает длинный ответ на части, не разрывая слова и абзацы."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else [""]

    chunks: list[str] = []
    current = ""
    for paragraph in text.split("\n"):
        candidate = f"{current}\n{paragraph}" if current else paragraph
        if len(candidate) <= limit:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        # Абзац сам длиннее лимита — режем по словам.
        words = paragraph.split(" ")
        for word in words:
            candidate = f"{current} {word}".strip() if current else word
            if len(candidate) <= limit:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                while len(word) > limit:
                    chunks.append(word[:limit])
                    word = word[limit:]
                current = word

    if current:
        chunks.append(current)
    return chunks


def main() -> None:
    setup_logging()
    try:
        VkBot().run()
    except SystemExit as exc:
        if exc.code:
            logging.error("%s", exc)
        raise


if __name__ == "__main__":
    main()
