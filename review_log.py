"""
Review log: a small record of the moderation decisions worth a human look,
kept on the Railway Volume (DATA_DIR/review_log.jsonl) so the admin can
download it, label it in Mod Trainer, and the model can be improved.

The bot NEVER retrains or changes its own rules from this file. It only
collects; every change is tested against the regression suite first.

What gets logged (one JSON line each):
  - every action the bot takes (blocklist warn/ban, DM mute, spam mute)
  - every message with a DM-type word that the bot let through, with the
    reason (borderline score, rescued by reply/group/conversation, etc.)
Never logged: ordinary messages without a DM word, usernames, phone numbers.

Size: when the file passes REVIEW_LOG_MAX_MB it is renamed to
review_log.old.jsonl (replacing any older one), so at most about twice
that size is ever kept. Logging can never crash the bot: every error is
caught and printed.
"""

import json
import os
from datetime import datetime, timezone, timedelta

import config

IST = timezone(timedelta(hours=5, minutes=30))


def _rotate_if_needed(path):
    try:
        if os.path.getsize(path) > config.REVIEW_LOG_MAX_MB * 1024 * 1024:
            old = path.replace(".jsonl", ".old.jsonl")
            os.replace(path, old)
    except FileNotFoundError:
        pass


def log(*, message, text, action, reason, score=None, categories=None,
        reply_to_text=None, prev_messages=None, is_edit=False):
    if not config.REVIEW_LOG_ENABLED:
        return
    try:
        path = config.REVIEW_LOG_PATH
        _rotate_if_needed(path)
        user = getattr(message, "from_user", None)
        record = {
            "ts": datetime.now(IST).isoformat(timespec="seconds"),
            "chat_id": message.chat.id,
            "msg_id": message.id,
            "user_id": user.id if user else None,
            "is_edit": is_edit,
            "text": text,
            "reply_to_msg_id": message.reply_to_message_id,
            "reply_to_text": reply_to_text,
            "prev_messages": prev_messages or [],
            "action": action,          # mute | warn | ban | spam | none
            "reason": reason,          # why the bot decided this
            "score": None if score is None else round(float(score), 3),
            "categories": sorted(categories) if categories else [],
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[ERROR] review_log failed: {type(e).__name__}: {e}")
