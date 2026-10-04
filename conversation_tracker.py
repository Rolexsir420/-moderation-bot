"""
Keeps a short rolling window of recent messages per chat so we can
reconstruct 'who started it' when a message gets flagged as abusive.

This layer is deterministic (reply chains and timestamps aren't guesses) —
but it is only as good as the upstream classifier deciding is_abusive.
If the instigator's opening message wasn't flagged, this can't reach back
to protect the person who was baited. Keep victim-side actions soft
(delete + note, no strike) and always log both sides to the admin chat.
"""

from collections import deque
import time

import config

# chat_id -> deque of message records
history = {}


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
    Returns (instigator_user_id, instigator_msg_id) if found, else
    (None, None) (meaning target_user_id is the first abusive message
    in this thread, as far as we can tell from the recent window).
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


def get_message_text(chat_id, msg_id):
    """
    Returns the text of a specific message by id, if it's still in this
    chat's rolling history, else None (e.g. it aged out of the small
    fixed-size buffer, or was never recorded). Used to pull in what a
    reply is actually replying TO -- e.g. Person A asks "anybody want
    to play a game?", Person B replies "add me" -- read alone "add me"
    is ambiguous, but combined with what it's replying to, it's clearly
    about joining a game, not DM solicitation.
    """
    buf = history.get(chat_id)
    if not buf:
        return None
    for m in buf:
        if m["msg_id"] == msg_id:
            return m["text"]
    return None


def get_recent_texts(chat_id, user_id, exclude_msg_id, n=3):
    """
    Returns the text of this user's last n messages in this chat
    (most recent first), excluding exclude_msg_id (the message
    currently being evaluated). Used only to RESCUE borderline
    classifier calls toward safe -- e.g. a bare "pm" that's actually
    part of an ongoing political/news conversation, not solicitation.
    Never used to escalate a score, only to soften it.
    """
    buf = history.get(chat_id)
    if not buf:
        return []
    recent = [
        m["text"] for m in reversed(buf)
        if m["user_id"] == user_id and m["msg_id"] != exclude_msg_id
    ]
    return recent[:n]


# --- Duplicate-message spam detection ------------------------------------
# Separate from `history` above on purpose: `history` is a small
# fixed-size deque (MAX_HISTORY_PER_CHAT, e.g. 50 messages) meant for the
# short instigator-lookup window. Spam detection needs a much longer
# (hours-long) per-user memory that a busy group would blow straight
# through if it shared that same small buffer -- so it gets its own
# time-pruned store instead of a fixed message count.
#
# (chat_id, user_id) -> list of {"text": normalized_text, "ts": ts}
_spam_history = {}


def _normalize_for_spam_check(text):
    return " ".join((text or "").lower().split())


def check_and_record_duplicate(chat_id, user_id, text):
    """
    Returns True if this user has already sent this EXACT same message
    (case/whitespace-insensitive) in this chat within the last
    config.SPAM_WINDOW_HOURS. Always records the current message for
    future checks, regardless of the result.

    Only meant for content that's otherwise clean (no blocklist hit, no
    DM-solicitation signal) but becomes a problem purely through
    repetition -- e.g. the same self-promo line posted every so often
    for hours. Exact-match only, deliberately: fuzzy/near-duplicate
    matching risks flagging people who just talk about similar topics.
    """
    now = time.time()
    window = config.SPAM_WINDOW_HOURS * 3600
    key = (chat_id, user_id)
    entries = _spam_history.setdefault(key, [])

    # prune anything older than the window
    entries[:] = [e for e in entries if now - e["ts"] <= window]

    normalized = _normalize_for_spam_check(text)
    is_duplicate = any(e["text"] == normalized for e in entries)

    entries.append({"text": normalized, "ts": now})

    return is_duplicate
