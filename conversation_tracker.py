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
