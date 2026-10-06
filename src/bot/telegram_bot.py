import asyncio
import html
import os
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv

from src.common.model_utils import analyze_full, analyze_text
from src.common.voice_utils import transcribe_audio

load_dotenv()
BOT_TOKEN = os.getenv("telegram_bot_token") or os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
SCAM_THRESHOLD = float(os.getenv("SCAM_THRESHOLD", "0.6"))

if not BOT_TOKEN:
    raise ValueError("telegram_bot_token not found in .env")

API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"
FILE_BASE = f"https://api.telegram.org/file/bot{BOT_TOKEN}"
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCAM_TEMPLATES_PATH = BASE_DIR / "data" / "scam_calls" / "scam.txt"
SAFE_TEMPLATES_PATH = BASE_DIR / "data" / "scam_calls" / "not_scam.txt"

MENU_TEXT = (
    "🛡 ScamShield готов к проверке.\n"
    "Выберите действие кнопками ниже:\n"
    "• Проверить текст\n"
    "• Проверить голосовое\n"
    "• Включить/выключить проверку сообщений в группе"
)
BTN_CHECK_TEXT = "🔎 Проверить текст"
BTN_CHECK_VOICE = "🎙 Проверить голосовое"
BTN_GROUP_ON = "✅ Включить проверку в группе"
BTN_GROUP_OFF = "⛔ Выключить проверку в группе"
BTN_HELP_CONNECT = "ℹ️ Как подключить"
BTN_GROUP_ON_HERE = "✅ Включить проверку здесь"
BTN_GROUP_OFF_HERE = "⛔ Выключить проверку здесь"

GROUP_SCAN_ENABLED_USERS: set[int] = set()
ENABLED_GROUP_CHATS: set[int] = set()
WAITING_FOR_TEXT_FROM_CHAT: set[int] = set()
BUSINESS_ALERT_CHAT_CACHE: dict[str, int] = {}


def load_templates(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


SCAM_TEMPLATES = load_templates(SCAM_TEMPLATES_PATH)
SAFE_TEMPLATES = load_templates(SAFE_TEMPLATES_PATH)


def boost_risk_with_rules(text: str, base_prob: float) -> float:
    text_lower = text.lower().strip()
    boosted = base_prob
    scam_similarity = max((SequenceMatcher(None, text_lower, t).ratio() for t in SCAM_TEMPLATES), default=0.0)
    safe_similarity = max((SequenceMatcher(None, text_lower, t).ratio() for t in SAFE_TEMPLATES), default=0.0)
    boosted += 0.35 * scam_similarity
    boosted -= 0.20 * safe_similarity
    return max(0.0, min(boosted, 1.0))


def is_dangerous(prob: float) -> bool:
    return prob >= SCAM_THRESHOLD


def format_alert_message(source_label: str, original_text: str, risk: float) -> str:
    quoted = html.escape((original_text or "").strip()) or "[пустое сообщение]"
    return (
        "⚠️ <b>Обнаружен риск мошенничества</b>\n"
        f"Источник: <b>{html.escape(source_label)}</b>\n"
        f"Риск: <b>{risk * 100:.1f}%</b>\n\n"
        "🧾 <b>Цитата сообщения:</b>\n"
        f"<i>{quoted}</i>"
    )


def build_main_keyboard() -> dict[str, Any]:
    return {
        "keyboard": [
            [{"text": BTN_CHECK_TEXT}, {"text": BTN_CHECK_VOICE}],
            [{"text": BTN_HELP_CONNECT}],
        ],
        "resize_keyboard": True,
    }


async def tg_api(session: aiohttp.ClientSession, method: str, payload: dict[str, Any]) -> dict[str, Any]:
    async with session.post(f"{API_BASE}/{method}", json=payload) as response:
        data = await response.json(content_type=None)
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API {method} failed: {data}")
    return data["result"]


async def send_message(
    session: aiohttp.ClientSession,
    chat_id: int,
    text: str,
    business_connection_id: str | None = None,
    parse_mode: str | None = "HTML",
    reply_markup: dict[str, Any] | None = None,
    reply_to_message_id: int | None = None,
) -> None:
    payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    if business_connection_id:
        payload["business_connection_id"] = business_connection_id
    if reply_markup:
        payload["reply_markup"] = reply_markup
    if reply_to_message_id:
        payload["reply_to_message_id"] = reply_to_message_id
    await tg_api(session, "sendMessage", payload)


async def download_voice(session: aiohttp.ClientSession, file_id: str, prefix: str) -> str:
    file_info = await tg_api(session, "getFile", {"file_id": file_id})
    file_path = file_info["file_path"]
    extension = Path(file_path).suffix or ".ogg"
    output_path = f"{prefix}{extension}"
    async with session.get(f"{FILE_BASE}/{file_path}") as response:
        response.raise_for_status()
        content = await response.read()
    with open(output_path, "wb") as file:
        file.write(content)
    return output_path


async def resolve_alert_chat_id(session: aiohttp.ClientSession, business_connection_id: str) -> int | None:
    cached = BUSINESS_ALERT_CHAT_CACHE.get(business_connection_id)
    if cached:
        return cached
    result = await tg_api(session, "getBusinessConnection", {"business_connection_id": business_connection_id})
    owner_user_id = result.get("user", {}).get("id")
    if owner_user_id:
        BUSINESS_ALERT_CHAT_CACHE[business_connection_id] = owner_user_id
        return owner_user_id
    return None


async def process_text(text: str) -> float:
    prob = analyze_text(text)
    return boost_risk_with_rules(text, prob)


async def process_voice(session: aiohttp.ClientSession, file_id: str, local_prefix: str) -> tuple[str, float, float, float]:
    local_path = await download_voice(session, file_id, local_prefix)
    try:
        text = transcribe_audio(local_path)
        try:
            final_prob, text_prob, audio_prob = analyze_full(text, local_path)
        except Exception as exc:
            print(f"Audio model failed, fallback to text-only: {exc}")
            text_prob = analyze_text(text)
            audio_prob = text_prob
            final_prob = text_prob
        text_prob = boost_risk_with_rules(text, text_prob)
        final_prob = (text_prob + audio_prob) / 2
        return text, final_prob, text_prob, audio_prob
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


async def analyze_and_respond_text(session: aiohttp.ClientSession, chat_id: int, source_label: str, text: str, only_dangerous: bool = False, reply_to_message_id: int | None = None) -> None:
    risk = await process_text(text)
    dangerous = is_dangerous(risk)
    if only_dangerous and not dangerous:
        return
    status = "⚠️ Высокий риск" if dangerous else "✅ Явных признаков скама мало"
    await send_message(
        session,
        chat_id,
        f"{status}\nИсточник: <b>{html.escape(source_label)}</b>\nРиск: <b>{risk * 100:.1f}%</b>",
        reply_to_message_id=reply_to_message_id,
    )


async def analyze_and_respond_voice(session: aiohttp.ClientSession, chat_id: int, source_label: str, file_id: str, prefix: str, only_dangerous: bool = False, reply_to_message_id: int | None = None) -> None:
    transcript, final_prob, text_prob, audio_prob = await process_voice(session, file_id, prefix)
    dangerous = is_dangerous(final_prob)
    if only_dangerous and not dangerous:
        return
    status = "⚠️ Высокий риск" if dangerous else "✅ Явных признаков скама мало"
    await send_message(
        session,
        chat_id,
        (
            f"{status}\nИсточник: <b>{html.escape(source_label)}</b>\n"
            f"Риск: <b>{final_prob * 100:.1f}%</b>\n"
            f"📊 Текст: {text_prob * 100:.1f}%\n"
            f"🎙 Аудио: {audio_prob * 100:.1f}%\n\n"
            f"🧾 Расшифровка:\n<i>{html.escape(transcript or '[пусто]')}</i>"
        ),
        reply_to_message_id=reply_to_message_id,
    )




def connection_help_text() -> str:
    return (
        "<b>Как подключить бота</b>\n\n"
        "<b>1) Личный чат:</b> откройте диалог с ботом и используйте кнопки проверки текста/голоса.\n\n"
        "<b>2) Проверка в группе:</b>\n"
        "• Добавьте бота в нужную группу.\n"
        "• Дайте право читать сообщения.\n"
        "• В группе отправьте кнопку/текст: <i>✅ Включить проверку здесь</i>.\n"
        "• Для остановки: <i>⛔ Выключить проверку здесь</i>.\n\n"
        "<b>3) Business-режим:</b> подключите Telegram Business к этому боту. "
        "Тогда опасные сообщения из личных чатов будут приходить в ваш личный чат с ботом."
    )


def group_keyboard() -> dict[str, Any]:
    return {
        "keyboard": [
            [{"text": BTN_GROUP_ON_HERE}, {"text": BTN_GROUP_OFF_HERE}],
        ],
        "resize_keyboard": True,
    }
async def handle_private_bot_chat(session: aiohttp.ClientSession, message: dict[str, Any]) -> None:
    chat_id = message["chat"]["id"]
    text = (message.get("text") or "").strip()
    voice = message.get("voice") or message.get("audio")

    if text in {"/start", BTN_CHECK_TEXT, BTN_CHECK_VOICE, BTN_HELP_CONNECT}:
        if text == "/start":
            WAITING_FOR_TEXT_FROM_CHAT.discard(chat_id)
            await send_message(session, chat_id, MENU_TEXT, reply_markup=build_main_keyboard())
            return
        if text == BTN_CHECK_TEXT:
            WAITING_FOR_TEXT_FROM_CHAT.add(chat_id)
            await send_message(session, chat_id, "Отправьте текст для проверки.", reply_markup=build_main_keyboard())
            return
        if text == BTN_CHECK_VOICE:
            await send_message(session, chat_id, "Отправьте голосовое сообщение для проверки.", reply_markup=build_main_keyboard())
            return
        if text == BTN_HELP_CONNECT:
            await send_message(session, chat_id, connection_help_text(), reply_markup=build_main_keyboard())
            return

    if text and chat_id in WAITING_FOR_TEXT_FROM_CHAT:
        WAITING_FOR_TEXT_FROM_CHAT.discard(chat_id)
        await analyze_and_respond_text(session, chat_id, "Личный чат", text)
        return

    if voice and voice.get("file_id"):
        try:
            await analyze_and_respond_voice(session, chat_id, "Личный чат", voice["file_id"], "temp_private_voice")
        except Exception as exc:
            print(f"Private voice processing failed: {exc}")
            await send_message(session, chat_id, "❌ Не удалось обработать голосовое сообщение.")
        return

    await send_message(session, chat_id, MENU_TEXT, reply_markup=build_main_keyboard())




async def is_group_admin(session: aiohttp.ClientSession, chat_id: int, user_id: int | None) -> bool:
    if not user_id:
        return False
    try:
        member = await tg_api(session, "getChatMember", {"chat_id": chat_id, "user_id": user_id})
    except Exception as exc:
        print(f"Cannot check admin status: {exc}")
        return False
    return member.get("status") in {"administrator", "creator"}
async def handle_group_message(session: aiohttp.ClientSession, message: dict[str, Any]) -> None:
    chat = message.get("chat", {})
    chat_id = chat.get("id")
    if not chat_id:
        return

    text = (message.get("text") or "").strip()
    user_id = message.get("from", {}).get("id")
    if text in {BTN_GROUP_ON_HERE, BTN_GROUP_ON, BTN_GROUP_OFF_HERE, BTN_GROUP_OFF}:
        if not await is_group_admin(session, chat_id, user_id):
            await send_message(session, chat_id, "Только администратор группы может включать или выключать проверку.")
            return
    if text in {BTN_GROUP_ON_HERE, BTN_GROUP_ON}:
        ENABLED_GROUP_CHATS.add(chat_id)
        await send_message(session, chat_id, "Проверка скама включена для этой группы.", reply_markup=group_keyboard())
        return
    if text in {BTN_GROUP_OFF_HERE, BTN_GROUP_OFF}:
        ENABLED_GROUP_CHATS.discard(chat_id)
        await send_message(session, chat_id, "Проверка скама выключена для этой группы.", reply_markup=group_keyboard())
        return

    if chat_id not in ENABLED_GROUP_CHATS:
        entities = message.get("entities") or []
        is_mention = any(e.get("type") == "mention" for e in entities)
        if is_mention or message.get("reply_to_message"):
            await send_message(
                session,
                chat_id,
                "Чтобы включить проверку в этой группе, отправьте: <b>✅ Включить проверку здесь</b>",
                reply_markup=group_keyboard(),
            )
        return

    source_label = chat.get("title") or f"group_id={chat_id}"
    if text:
        await analyze_and_respond_text(
            session,
            chat_id,
            source_label,
            text,
            only_dangerous=True,
            reply_to_message_id=message.get("message_id"),
        )
        return

    voice = message.get("voice") or message.get("audio")
    if voice and voice.get("file_id"):
        try:
            await analyze_and_respond_voice(
                session,
                chat_id,
                source_label,
                voice["file_id"],
                "temp_group_voice",
                only_dangerous=True,
                reply_to_message_id=message.get("message_id"),
            )
        except Exception as exc:
            print(f"Group voice processing failed: {exc}")
            await send_message(session, chat_id, "❌ Не удалось обработать голосовое сообщение в группе.")


async def handle_business_update(session: aiohttp.ClientSession, business_message: dict[str, Any]) -> None:
    source_chat_id = business_message.get("chat", {}).get("id")
    source_title = business_message.get("chat", {}).get("title")
    business_connection_id = business_message.get("business_connection_id")
    if not source_chat_id or not business_connection_id:
        return
    alert_chat_id = await resolve_alert_chat_id(session, business_connection_id)
    if not alert_chat_id:
        print("Cannot resolve alert chat id for business connection")
        return
    source_label = source_title or f"chat_id={source_chat_id}"
    text = business_message.get("text")
    if text:
        risk = await process_text(text)
        if is_dangerous(risk):
            alert = format_alert_message(source_label, text, risk)
            await send_message(session, alert_chat_id, alert)
        return
    voice = business_message.get("voice") or business_message.get("audio")
    if voice and voice.get("file_id"):
        try:
            transcript, final_prob, text_prob, audio_prob = await process_voice(session, voice["file_id"], "temp_business_voice")
        except Exception as exc:
            print(f"Voice processing failed: {exc}")
            await send_message(session, alert_chat_id, "❌ Не удалось обработать голосовое сообщение. Проверьте установку ffmpeg и доступность аудиокодеков.")
            return
        if is_dangerous(final_prob):
            alert = (
                format_alert_message(source_label, transcript, final_prob)
                + "\n\n"
                + f"📊 Текст: {text_prob * 100:.1f}%\n"
                + f"🎙 Аудио: {audio_prob * 100:.1f}%"
            )
            await send_message(session, alert_chat_id, alert)


async def handle_update(session: aiohttp.ClientSession, update: dict[str, Any]) -> None:
    message = update.get("message")
    if message:
        chat_type = message.get("chat", {}).get("type")
        if chat_type == "private":
            await handle_private_bot_chat(session, message)
        elif chat_type in {"group", "supergroup"}:
            await handle_group_message(session, message)
    business_message = update.get("business_message") or update.get("edited_business_message")
    if business_message:
        await handle_business_update(session, business_message)


async def polling_loop() -> None:
    offset = 0
    allowed_updates = ["message", "business_message", "edited_business_message"]
    print("Bot started...")
    print(f"Scam threshold: {SCAM_THRESHOLD}")
    print(f"Scam templates loaded: {len(SCAM_TEMPLATES)}")
    print(f"Safe templates loaded: {len(SAFE_TEMPLATES)}")
    timeout = aiohttp.ClientTimeout(total=90)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            try:
                updates = await tg_api(session, "getUpdates", {"offset": offset, "timeout": 60, "allowed_updates": allowed_updates})
                for item in updates:
                    offset = item["update_id"] + 1
                    await handle_update(session, item)
            except Exception as exc:
                print(f"Polling error: {exc}")
                await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(polling_loop())
