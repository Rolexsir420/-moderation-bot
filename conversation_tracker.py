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
    """
    Adds a message to the chat's short-term history. If this msg_id is
    already recorded (an EDITED message), the existing record is updated
    in place instead of appending a duplicate, so history rescue never
    sees both the old and the edited text as two separate messages.
    """
    buf = history.setdefault(chat_id, deque(maxlen=config.MAX_HISTORY_PER_CHAT))
    for m in buf:
        if m["msg_id"] == msg_id:
            m["text"] = text
            m["is_abusive"] = m["is_abusive"] or is_abusive
            return
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


def get_record(chat_id, msg_id):
    """The stored record for msg_id in this chat, or None if not in the window."""
    buf = history.get(chat_id)
    if not buf:
        return None
    for m in buf:
        if m["msg_id"] == msg_id:
            return m
    return None


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


def get_last_messages(chat_id, n=5, exclude_msg_id=None):
    """The last n messages in the chat (any sender, oldest first), as
    {"user_id", "text"} — context saved with each review-log entry."""
    buf = history.get(chat_id)
    if not buf:
        return []
    out = [{"user_id": m["user_id"], "text": m["text"]}
           for m in buf if m["msg_id"] != exclude_msg_id]
    return out[-n:]


def get_recent_chat_texts(chat_id, exclude_user_id, within_seconds=120, n=5):
    """
    Up to n recent message texts from OTHER users in this chat, sent in the
    last within_seconds (most recent first), skipping ones already flagged
    abusive. Used ONLY to rescue a borderline score toward safe when
    someone answers a question without tagging it — e.g. Prem asks "pvt
    or govt?" and Tharun just types "nanba pvt la".
    """
    buf = history.get(chat_id)
    if not buf:
        return []
    now = time.time()
    out = []
    for m in reversed(buf):
        if now - m["ts"] > within_seconds:
            break
        if m["user_id"] == exclude_user_id or m["is_abusive"]:
            continue
        out.append(m["text"])
        if len(out) >= n:
            break
    return out


_WS_RE = re.compile(r"\s+")


def _normalize_for_spam_check(text):
    return _WS_RE.sub(" ", (text or "").strip().lower())


# Ordinary chit-chat that naturally repeats many times a day in any
# friend group and should never be treated as flood/spam, however many
# times it's sent. Checked after normalization (lowercased, whitespace
# collapsed), so "Hi", "HI", "hi  " all match "hi".
COMMON_FILLERS = {
    "hi", "hii", "hiii", "hello", "hey", "heyy",
    "saaptiya", "saaptingala", "saapten", "saapten da",
    "nalla iruken", "nalla irukken", "nalla irukka",
    "mm", "mmm", "mmmm", "hmm", "hmmm",
    "ok", "okay", "k", "kk",
    "vanakkam", "bye", "byee", "gm", "gn",
    "good morning", "good night", "good afternoon",
    "thanks", "thank you", "tq", "thx",
    "yes", "no", "ya", "yeah", "nope",
    "super", "nice", "lol", "haha", "hahaha",
}


def check_and_record_duplicate(chat_id, user_id, text):
    """
    Returns True once this exact (normalized) text has been sent by this
    same user in this chat at least SPAM_REPEAT_THRESHOLD times within
    SPAM_WINDOW_HOURS (counting the current message). Always records the
    current message afterward (whether or not the threshold was hit yet),
    so the window keeps sliding forward.

    Two deliberate limits on what counts as spam:
      - exact-match only, no fuzzy matching, to avoid false positives on
        messages that are merely similar in topic.
      - a single word, or a common greeting/filler phrase (COMMON_FILLERS
        above), never counts as spam no matter how many times it repeats
        — ordinary chat is full of people saying "hi"/"ok"/"mm" a dozen
        times a day, that's not flooding.
    """
    if user_id is None:
        return False

    normalized = _normalize_for_spam_check(text)
    if not normalized:
        return False

    if len(normalized.split()) <= 1 or normalized in COMMON_FILLERS:
        return False

    key = (chat_id, user_id)
    now = time.time()
    window_seconds = config.SPAM_WINDOW_HOURS * 3600

    entries = _spam_history.setdefault(key, [])
    # prune anything outside the window
    entries[:] = [(t, ts) for (t, ts) in entries if now - ts <= window_seconds]

    matching_count = sum(1 for (t, ts) in entries if t == normalized)
    entries.append((normalized, now))

    # +1 to count the current message itself, not just prior occurrences
    return (matching_count + 1) >= config.SPAM_REPEAT_THRESHOLD
