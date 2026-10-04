"""
Keeps short-term memory of recent messages per chat, for three purposes:

1. find_instigator — reconstruct "who started it" when a message gets
   flagged as abusive, so a victim replying to an earlier abusive message
   isn't the one who gets actioned.
2. get_recent_texts — a sender's own recent messages, used ONLY to soften
   a borderline score toward safe (never to escalate), e.g. someone
   clearly mid-conversation about something unrelated.
3. check_and_record_duplicate — flood/spam detection: the same exact
   message text from the same user within SPAM_WINDOW_HOURS.

This layer is deterministic (reply chains, timestamps, and exact-text
matches aren't guesses) — but (1) and (2) are only as good as the upstream
classifier's is_abusive decision for earlier messages in the window.
"""

from collections import deque
import re
import time

import config

# chat_id -> deque of message records (small, fixed-size, short-term)
history = {}

# (chat_id, user_id) -> list of (normalized_text, timestamp), pruned by
# SPAM_WINDOW_HOURS. Separate from `history` because spam detection needs
# hours-long per-user memory, not just the last N messages in the chat.
_spam_history = {}


def record_message(chat_id, msg_id, user_id, reply_to_id, text, is_abusive):
    buf = history.setdefault(chat_id, deque(maxlen=config.MAX_HISTORY_PER_CHAT))
    buf.append({
        "msg_id": msg_id,
        "user_id": user_id,
        "reply_to_id": reply_to_id,
        "text": text,
        "is_abusive": is_abusive,
        "ts": time.time(),
    })


def find_instigator(chat_id, target_user_id, target_msg_id, reply_to_id):
    """
    Given a flagged message from target_user_id, check whether it's a
    reply/reaction to an earlier abusive message directed at them.
    Returns (instigator_user_id, instigator_msg_id), or (None, None) if
    target_user_id appears to be the first abusive message in this
    thread, as far as we can tell from the recent window.
    """
    buf = history.get(chat_id)
    if not buf:
        return None, None

    now = time.time()

    # 1) direct reply chain: does this message reply to an earlier
    #    abusive message from someone else?
    if reply_to_id:
        for m in buf:
            if m["msg_id"] == reply_to_id and m["is_abusive"] and m["user_id"] != target_user_id:
                return m["user_id"], m["msg_id"]

    # 2) fallback: no explicit reply_to — look at the recent window for
    #    an earlier abusive message from someone else aimed at this user
    participants_recent = [
        m for m in buf
        if now - m["ts"] <= config.CONTEXT_WINDOW_SECONDS and m["msg_id"] != target_msg_id
    ]
    for m in participants_recent:
        if m["is_abusive"] and m["user_id"] != target_user_id:
            return m["user_id"], m["msg_id"]

    return None, None


def get_recent_texts(chat_id, user_id, exclude_msg_id=None, n=3):
    """
    Returns up to n of this sender's own most recent message texts in
    this chat (most recent first), excluding exclude_msg_id. Used only to
    RESCUE a borderline score toward safe — never to push toward warn.
    """
    buf = history.get(chat_id)
    if not buf:
        return []
    out = []
    for m in reversed(buf):
        if m["user_id"] != user_id:
            continue
        if exclude_msg_id is not None and m["msg_id"] == exclude_msg_id:
            continue
        out.append(m["text"])
        if len(out) >= n:
            break
    return out


_WS_RE = re.compile(r"\s+")


def _normalize_for_spam_check(text):
    return _WS_RE.sub(" ", (text or "").strip().lower())


def check_and_record_duplicate(chat_id, user_id, text):
    """
    Returns True if this exact (normalized) text was already sent by this
    same user in this chat within SPAM_WINDOW_HOURS. Always records the
    current message afterward (whether or not it was a duplicate), so the
    window keeps sliding forward.

    Exact-match only, deliberately no fuzzy matching, to avoid false
    positives on messages that are merely similar in topic.
    """
    if user_id is None:
        return False

    normalized = _normalize_for_spam_check(text)
    if not normalized:
        return False

    key = (chat_id, user_id)
    now = time.time()
    window_seconds = config.SPAM_WINDOW_HOURS * 3600

    entries = _spam_history.setdefault(key, [])
    # prune anything outside the window
    entries[:] = [(t, ts) for (t, ts) in entries if now - ts <= window_seconds]

    is_duplicate = any(t == normalized for (t, ts) in entries)
    entries.append((normalized, now))

    return is_duplicate
