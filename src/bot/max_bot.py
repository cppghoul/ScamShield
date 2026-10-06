import asyncio
import html
import os
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from maxapi import Bot, Dispatcher
from maxapi.enums import ChatType, MessageLinkType, ParseMode
from maxapi.types import (
    BotStarted,
    CallbackButton,
    Command,
    MessageCallback,
    MessageCreated,
)
from maxapi.types.attachments import Attachment
from maxapi.types.message import Message, MessageBody
from maxapi.utils.inline_keyboard import InlineKeyboardBuilder

from src.common.model_utils import analyze_text

load_dotenv()
MAX_BOT_TOKEN = os.getenv("max_bot_token") or os.getenv("MAX_BOT_TOKEN")
SCAM_THRESHOLD = float(os.getenv("SCAM_THRESHOLD", "0.6"))

if not MAX_BOT_TOKEN:
    raise ValueError("max_bot_token not found in .env")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCAM_TEMPLATES_PATH = BASE_DIR / "data" / "scam_calls" / "scam.txt"
SAFE_TEMPLATES_PATH = BASE_DIR / "data" / "scam_calls" / "not_scam.txt"

BTN_CHECK_TEXT = "🔎 Проверить текст"
BTN_HELP_CONNECT = "ℹ️ Как подключить"
BTN_GROUP_ON_HERE = "✅ Включить проверку здесь"
BTN_GROUP_OFF_HERE = "⛔ Выключить проверку здесь"

MENU_TEXT = (
    "🛡 ScamShield готов к проверке.\n"
    "Отправьте текстовое сообщение на анализ "
    "либо выберите действие кнопками ниже."
)

ENABLED_GROUP_CHATS: set[int] = set()
BOT_USERNAME: str | None = None
BOT_USER_ID: int | None = None
BOT: Bot | None = None


def get_bot(message: Message | None = None) -> Bot:
    if message is not None:
        bot = getattr(message, "bot", None)
        if bot is not None:
            return bot
    if BOT is None:
        raise RuntimeError("Bot is not initialized")
    return BOT


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


def extract_content(message: Message) -> tuple[str, bool]:
    bodies: list[MessageBody] = []
    is_forward = False
    if message.body is not None and (message.body.text or "").strip():
        bodies.append(message.body)
    if (
        message.link is not None
        and message.link.type == MessageLinkType.FORWARD
        and message.link.message is not None
    ):
        is_forward = True
        bodies.append(message.link.message)

    text_parts = [(body.text or "").strip() for body in bodies]
    text_parts = [part for part in text_parts if part]
    return "\n".join(text_parts), is_forward


def build_main_keyboard() -> Attachment:
    keyboard = InlineKeyboardBuilder()
    keyboard.row(
        CallbackButton(text=BTN_CHECK_TEXT, payload="check_text"),
        CallbackButton(text=BTN_HELP_CONNECT, payload="help"),
    )
    return keyboard.as_markup()


def build_group_keyboard() -> Attachment:
    keyboard = InlineKeyboardBuilder()
    keyboard.row(
        CallbackButton(text=BTN_GROUP_ON_HERE, payload="group_on"),
        CallbackButton(text=BTN_GROUP_OFF_HERE, payload="group_off"),
    )
    return keyboard.as_markup()


async def process_text(text: str) -> float:
    prob = analyze_text(text)
    return boost_risk_with_rules(text, prob)


async def safe_reply(message: Message, text: str, **kwargs: Any) -> None:
    if message.body is not None:
        await message.reply(text, **kwargs)
    else:
        await message.answer(text, **kwargs)


async def analyze_and_respond_text(
    bot: Bot,
    message: Message,
    source_label: str,
    text: str,
    only_dangerous: bool = False,
) -> None:
    risk = await process_text(text)
    dangerous = is_dangerous(risk)
    if only_dangerous and not dangerous:
        return
    status = "⚠️ Высокий риск" if dangerous else "✅ Явных признаков скама мало"
    await safe_reply(
        message,
        f"{status}\nИсточник: <b>{html.escape(source_label)}</b>\nРиск: <b>{risk * 100:.1f}%</b>",
        format=ParseMode.HTML,
    )


def connection_help_text() -> str:
    return (
        "<b>Как подключить бота</b>\n\n"
        "<b>1) Личный чат:</b> откройте диалог с ботом и отправьте текст "
        "(можно переслать из другого чата) - "
        "бот проверит его на признаки мошенничества.\n\n"
        "<b>2) Проверка в группе:</b>\n"
        "• Добавьте бота в нужную группу.\n"
        "• Дайте боту права администратора (MAX требует их для чтения сообщений).\n"
        "• Нажмите кнопку <i>✅ Включить проверку здесь</i> (или отправьте /scan_on).\n"
        "• Для остановки: <i>⛔ Выключить проверку здесь</i> (или /scan_off)."
    )


async def send_menu(message: Message) -> None:
    await message.answer(MENU_TEXT, attachments=[build_main_keyboard()])


async def is_group_admin(bot: Bot, chat_id: int, user_id: int | None) -> bool:
    if not user_id:
        return False
    try:
        member = await bot.get_chat_member(chat_id, user_id)
    except Exception as exc:
        print(f"Cannot check admin status: {exc}")
        return False
    return bool(member and (member.is_owner or member.is_admin))


async def handle_private_message(bot: Bot, message: Message) -> None:
    chat_id = message.recipient.chat_id
    if chat_id is None:
        return

    text, is_forward = extract_content(message)
    source_label = "Личный чат (пересылка)" if is_forward else "Личный чат"

    if text:
        await analyze_and_respond_text(bot, message, source_label, text)
        return

    await send_menu(message)


async def handle_group_message(bot: Bot, message: Message) -> None:
    chat_id = message.recipient.chat_id
    if chat_id is None:
        return

    text, is_forward = extract_content(message)
    sender_id = message.sender.user_id if message.sender else None

    if chat_id not in ENABLED_GROUP_CHATS:
        lowered = text.lower()
        mention = bool(BOT_USERNAME) and BOT_USERNAME.lower() in lowered
        is_reply_to_bot = False
        if message.link and message.link.type == MessageLinkType.REPLY:
            origin_sender = message.link.sender
            if origin_sender is None and message.link.message:
                try:
                    origin = await bot.get_message(message.link.message.mid)
                    origin_sender = origin.sender if origin else None
                except Exception:
                    origin_sender = None
            is_reply_to_bot = bool(
                origin_sender
                and (
                    origin_sender.is_bot
                    or (BOT_USER_ID is not None and origin_sender.user_id == BOT_USER_ID)
                )
            )
        if mention or is_reply_to_bot:
            await message.answer(
                "Чтобы включить проверку в этой группе, нажмите кнопку "
                f"<b>{BTN_GROUP_ON_HERE}</b> или отправьте /scan_on (только для админов).",
                format=ParseMode.HTML,
            )
        return

    source_label = f"группа {chat_id}" + (" (пересылка)" if is_forward else "")

    if text:
        await analyze_and_respond_text(
            bot,
            message,
            source_label,
            text,
            only_dangerous=True,
        )
        return


async def on_bot_started(event: BotStarted) -> None:
    print(f"Bot started in chat {event.chat_id}")
    try:
        await event.bot.send_message(
            chat_id=event.chat_id, text=MENU_TEXT, attachments=[build_main_keyboard()]
        )
    except Exception as exc:
        print(f"Failed to send menu on BotStarted: {exc}")


async def on_start_command(event: MessageCreated) -> None:
    await send_menu(event.message)


async def on_help_command(event: MessageCreated) -> None:
    await event.message.answer(connection_help_text(), format=ParseMode.HTML)


async def on_any_message(event: MessageCreated) -> None:
    message = event.message

    if message.sender and message.sender.is_bot:
        return

    try:
        bot = get_bot(message)
    except RuntimeError as exc:
        print(f"No bot context in message, skipping: {exc}")
        return

    chat_type = message.recipient.chat_type
    if chat_type == ChatType.DIALOG:
        await handle_private_message(bot, message)
    elif chat_type == ChatType.CHAT:
        await handle_group_message(bot, message)


async def ack_callback(event: MessageCallback, notification: str | None = None) -> None:
    try:
        await event.ack(notification=notification)
    except Exception as exc:
        print(f"Callback ack failed: {exc}")


async def on_message_callback(event: MessageCallback) -> None:
    callback = event.callback
    payload = (callback.payload or "").strip()
    user_id = callback.user.user_id if callback.user else None

    chat_id = None
    if event.message is not None:
        chat_id = event.message.recipient.chat_id
    if chat_id is None:
        await ack_callback(event)
        return

    bot = get_bot(event.message if event.message is not None else None)

    if payload == "check_text":
        await ack_callback(event, "Отправьте текст в чат бота")
        await event.message.answer("Отправьте текст для проверки - бот проанализирует его на признаки мошенничества.")
        return

    if payload == "help":
        await ack_callback(event)
        await event.message.answer(
            connection_help_text(), format=ParseMode.HTML, attachments=[build_main_keyboard()]
        )
        return

    if payload in {"group_on", "group_off"}:
        if not await is_group_admin(bot, chat_id, user_id):
            await ack_callback(event, "Нужны права администратора")
            await event.message.answer("Только администратор группы может включать или выключать проверку.")
            return
        if payload == "group_on":
            ENABLED_GROUP_CHATS.add(chat_id)
            await ack_callback(event, "Проверка включена")
            await event.message.answer(
                "Проверка скама включена для этой группы.",
                attachments=[build_group_keyboard()],
            )
        else:
            ENABLED_GROUP_CHATS.discard(chat_id)
            await ack_callback(event, "Проверка выключена")
            await event.message.answer(
                "Проверка скама выключена для этой группы.",
                attachments=[build_group_keyboard()],
            )
        return

    await ack_callback(event, "Неизвестное действие")


async def on_error(event) -> None:
    print(f"Dispatcher error: {event.exception}")


def register_handlers(dp: Dispatcher) -> None:
    dp.bot_started()(on_bot_started)
    dp.message_created(Command("start"))(on_start_command)
    dp.message_created(Command("help"))(on_help_command)
    dp.message_created(Command("scan_on"))(on_scan_on_command)
    dp.message_created(Command("scan_off"))(on_scan_off_command)
    dp.message_created()(on_any_message)
    dp.message_callback()(on_message_callback)
    dp.error()(on_error)


async def on_scan_on_command(event: MessageCreated) -> None:
    message = event.message
    chat_id = message.recipient.chat_id
    if chat_id is None or message.recipient.chat_type != ChatType.CHAT:
        await message.answer("Эта команда работает только в группах.")
        return
    sender_id = message.sender.user_id if message.sender else None
    bot = get_bot(message)
    if not await is_group_admin(bot, chat_id, sender_id):
        await message.answer("Только администратор группы может включать или выключать проверку.")
        return
    ENABLED_GROUP_CHATS.add(chat_id)
    await message.answer(
        "Проверка скама включена для этой группы.",
        attachments=[build_group_keyboard()],
    )


async def on_scan_off_command(event: MessageCreated) -> None:
    message = event.message
    chat_id = message.recipient.chat_id
    if chat_id is None or message.recipient.chat_type != ChatType.CHAT:
        await message.answer("Эта команда работает только в группах.")
        return
    sender_id = message.sender.user_id if message.sender else None
    bot = get_bot(message)
    if not await is_group_admin(bot, chat_id, sender_id):
        await message.answer("Только администратор группы может включать или выключать проверку.")
        return
    ENABLED_GROUP_CHATS.discard(chat_id)
    await message.answer(
        "Проверка скама выключена для этой группы.",
        attachments=[build_group_keyboard()],
    )


async def main() -> None:
    global BOT_USERNAME, BOT_USER_ID, BOT
    bot = Bot(MAX_BOT_TOKEN)
    BOT = bot
    dp = Dispatcher()

    register_handlers(dp)

    try:
        me = await bot.get_me()
        BOT_USER_ID = me.user_id
        BOT_USERNAME = f"@{me.username}" if me.username else None
        print(f"Authorized as {me.first_name} (id: {BOT_USER_ID}, username: {BOT_USERNAME})")
    except Exception as exc:
        print(f"get_me failed: {exc}")

    print("Bot started...")
    print(f"Scam threshold: {SCAM_THRESHOLD}")
    print(f"Scam templates loaded: {len(SCAM_TEMPLATES)}")
    print(f"Safe templates loaded: {len(SAFE_TEMPLATES)}")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
