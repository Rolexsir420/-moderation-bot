"""
Userbot (phone/OTP or SESSION_STRING login — NOT a bot-token bot) that
moderates a Tamil-language Telegram group.

Architecture: this bot NEVER deletes/mutes/bans anyone itself. It only
REPLIES to the offending message with short command text in the exact
convention a separate, pre-existing admin command bot in the group
understands (that bot infers the target from the reply-to relationship,
no @mention needed):

    .delwarn {reason}            - warn + delete, no mute
    .delmute {hours}h {reason}   - delete + mute for {hours}
    .delban {reason}             - delete + ban   (NOT currently used —
                                     blocklist hits are always .delwarn,
                                     and DM-solicitation is always a fixed
                                     .delmute, never escalating to ban)

Pipeline per message:
  1. Hardcoded blocklist (blocklist.py) — deterministic, checked first.
     Any hit -> always ".delwarn {category reason}", never escalates.
  2. Duplicate-message spam detection (conversation_tracker.py) — same
     exact text from the same user within SPAM_WINDOW_HOURS -> immediate
     ".delmute {SPAM_MUTE_HOURS}h no spam".
  3. Learned classifier (trainer.py / predict.py) for DM-solicitation
     intent ("pm me", "dm pannuga", "add me", ...) with several
     context-aware rescue mechanisms before committing to a mute:
       a. reply-to-parent-message rescue (fetches the message being
          replied to; can rescue even a confident "warn" label)
       b. bias-only + long-message gate (a bare anchor word in an
          otherwise long, unrelated sentence doesn't by itself mute)
       c. sender's-own-recent-message rescue (only softens borderline
          scores toward safe, never escalates)
"""

import asyncio
from datetime import datetime, timezone, timedelta

from pyrogram import Client, filters
from pyrogram.types import Message

import config
import blocklist
import conversation_tracker
import strikes
from predict import classify, load_weights
from trainer import is_direct_ask, parent_explains

weights = load_weights()

# Priority-ordered: first matching category in the hit set wins.
# "severe_exploitation" is handled separately in apply_blocklist_action
# (escalated to .delban + a loud admin alert, never just .delwarn) and is
# listed here only so reason_for_categories() has text for it too.
CATEGORY_REASONS = [
    ("severe_exploitation", "severe - exploitation content"),
    ("tamil_abusive", "no badwords"),
    ("tamil_sexual_slang", "no badwords"),
    ("sexual", "no 18+ talk in groupchat"),
    ("clothing_body", "no 18+ talk in groupchat"),
    ("relationship_dating", "no dating txt"),
    ("gender_orientation", "no 18+ talk in groupchat"),
    ("service_paid", "no scam/paid content"),
    ("scam", "no scam/paid content"),
    ("caste_hate", "no hate speech"),
]

# Categories that are an immediate .delban (not .delwarn) plus a loud
# admin alert, because bot action alone is not an adequate response to
# apparent child-sexual-abuse-solicitation content — the human admin
# needs to see this immediately and report the account to Telegram (and,
# depending on content, to local cybercrime authorities) on top of
# whatever the in-group bot does.
SEVERE_CATEGORIES = {"severe_exploitation"}


def reason_for_categories(categories):
    for category, reason in CATEGORY_REASONS:
        if category in categories:
            return reason
    return "no badwords"


def format_duration(n):
    """3.0 -> '3' so the command text reads '.delmute 3h ...' not '3.0h'."""
    if n == int(n):
        return str(int(n))
    return str(n)


if config.SESSION_STRING:
    app = Client(
        "moderation_userbot",
        api_id=config.API_ID,
        api_hash=config.API_HASH,
        session_string=config.SESSION_STRING,
    )
else:
    app = Client(
        "moderation_userbot",
        api_id=config.API_ID,
        api_hash=config.API_HASH,
        phone_number=config.PHONE_NUMBER,
    )


async def log_to_admin(client, text):
    if not config.ADMIN_CHAT_ID:
        return
    try:
        await client.send_message(config.ADMIN_CHAT_ID, text)
    except Exception as e:
        print(f"[ERROR] log_to_admin failed: {type(e).__name__}: {e}")


async def apply_blocklist_action(client, message, categories, user_id):
    reason = reason_for_categories(categories)
    is_severe = bool(categories & SEVERE_CATEGORIES)
    cmd_text = f".delban {reason}" if is_severe else f".delwarn {reason}"
    await message.reply_text(cmd_text)
    # strike count kept for the admin log/audit trail only — for ordinary
    # blocklist hits it never changes which command gets sent (always
    # .delwarn). Severe hits escalate to .delban regardless of strikes.
    new_count = strikes.add_strike(user_id, category=next(iter(categories)))

    if is_severe:
        await log_to_admin(
            client,
            "⚠️⚠️ URGENT — possible child sexual abuse "
            "solicitation/exploitation content detected and banned "
            f"automatically. user={user_id} chat={message.chat.id} "
            f"msg_id={message.id} categories={sorted(categories)}.\n"
            "Bot action (.delban) is NOT sufficient on its own for this: "
            "please also report this account to Telegram directly "
            "(message > Report > Child Abuse, or t.me/notoscam / the in-app "
            "Report button) and, if you believe this reflects a real "
            "situation rather than just chat talk, consider reporting to "
            "local cybercrime authorities (in India: cybercrime.gov.in or "
            "dial 1930). Please review the original message yourself before "
            "it scrolls away.",
        )
    else:
        await log_to_admin(
            client,
            f"[blocklist] msg_id={message.id} user={user_id} categories={sorted(categories)} "
            f"strike_count={new_count} -> {cmd_text}",
        )


async def try_reply_context_rescue(client, message, text, original_total):
    """
    If this message is a reply to another message, fetch that parent
    message and re-classify the COMBINED text (parent + this message).

    Rescue to safe if the parent text genuinely helps — the combined
    score has to drop from the standalone score (not stay flat or rise),
    AND land clearly below a confident-warn level. Requiring a strict
    drop below 0 was too strict in practice: "Game vaanga" + "Add me"
    (a reply to a game invite) dropped from 0.28 to 0.05 — real context
    that should rescue it, but 0.05 isn't "< 0". A relative threshold
    (dropped, and now well under WARN_THRESHOLD) catches that case
    without needing a combined score that goes fully negative.

    Can rescue even a confident "warn" label, not just borderline ones,
    because the parent message is ground truth context, not a guess.
    Returns (rescued: bool, parent_text: str | None).
    """
    if not message.reply_to_message_id:
        return False, None
    if is_direct_ask(text):
        # "Pvt vaa" / "pm vaanga" ask directly; a reply to someone's
        # pvt-company message doesn't make that innocent.
        return False, None
    try:
        parent = await client.get_messages(message.chat.id, message.reply_to_message_id)
        parent_text = (getattr(parent, "text", None) or getattr(parent, "caption", None) or "")
    except Exception as e:
        print(f"[ERROR] try_reply_context_rescue get_messages failed: {type(e).__name__}: {e}")
        return False, None
    if not parent_text:
        return False, None
    if parent_explains(parent_text, text):
        # "Game vaanga" <- "Add me": the parent gives "add" its ordinary
        # meaning (a game/team/list/ticket), so the reply is answering it.
        return True, parent_text

    combined_text = parent_text + " " + text
    _combined_label, combined_total, _combined_hits = classify(combined_text, weights)
    rescued = (
        combined_total < original_total
        and combined_total < (config.WARN_THRESHOLD * 0.5)
    )
    return rescued, parent_text


# Text messages AND media (photo/video/document/...) that carry a caption.
# filters.text alone skipped captions, so "dm me" written on a photo was
# never checked.
MODERATED = filters.group & (filters.text | filters.caption) & ~filters.bot


@app.on_message(MODERATED)
async def handle_group_message(client, message: Message):
    await moderate(client, message, is_edit=False)


@app.on_edited_message(MODERATED)
async def handle_edited_group_message(client, message: Message):
    """
    Closes the "send something clean, then edit it into 'add me'"
    evasion. Guards against false triggers:
      - Telegram also sends edit updates for reaction changes; those
        carry no edit_date (or unchanged text), so they're skipped.
      - a message we already actioned is never actioned a second time.
      - spam detection is skipped for edits (an edit isn't a new send).
    """
    if not getattr(message, "edit_date", None):
        return
    prev = conversation_tracker.get_record(message.chat.id, message.id)
    new_text = message.text or message.caption or ""
    if prev is not None:
        if prev["is_abusive"] or prev["text"] == new_text:
            return
    await moderate(client, message, is_edit=True)


async def moderate(client, message: Message, is_edit=False):
    user = message.from_user

    # never act on our own messages (this is a userbot, not a separate
    # bot account, so it WILL see its own .delwarn/.delmute replies)
    #
    # the `~filters.bot` on the decorator already drops messages sent by
    # any bot ACCOUNT (e.g. the separate group-management bot — "Oreo
    # Queen" — that actually performs the mute/ban and posts its own
    # confirmation/notice messages like "Unga message remove pannitten...
    # Please report to admins instead"). Those are never something for
    # OUR classifier/blocklist to act on, whatever words happen to be in
    # them, so we must not re-moderate that bot's own messages. The
    # is_bot check below is kept as a second, explicit guard in case a
    # management "bot" in a given group is actually a disguised regular
    # account rather than a real bot-API account (so ~filters.bot alone
    # wouldn't catch it) — if that ever happens here, add its user id to
    # an allow-skip list the same way.
    if user and (getattr(user, "is_self", False) or getattr(user, "is_bot", False)):
        return

    if config.TARGET_CHAT_IDS and message.chat.id not in config.TARGET_CHAT_IDS:
        return
    if not config.TARGET_CHAT_IDS:
        # discovery mode: no TARGET_CHAT_IDS set yet, just log chat ids
        print(f"[DEBUG] message from chat id={message.chat.id} title={message.chat.title!r}")

    text = message.text or message.caption or ""
    if not text.strip():
        return
    user_id = user.id if user else None
    is_abusive = False

    # --- 1. hardcoded blocklist, checked first and deterministically ---
    is_hit, categories = blocklist.check_text(text)
    if is_hit:
        await apply_blocklist_action(client, message, categories, user_id)
        conversation_tracker.record_message(
            message.chat.id, message.id, user_id, message.reply_to_message_id, text, True,
        )
        return

    # --- 2. learned classifier for DM-solicitation intent ---
    label, total, hits = classify(text, weights)
    anchor_present = len(hits) > 0
    word_count = len(text.split())
    has_real_context = any(feat != "__BIAS__" for _, feat, _ in hits)

    should_mute = False

    if label == "warn":
        rescued, _parent_text = await try_reply_context_rescue(client, message, text, total)
        if not rescued:
            should_mute = True
    elif label == "uncertain" and anchor_present and total >= 0:
        # a bare anchor word (message/inbox/private/personal/add/etc.)
        # in an otherwise long, unrelated sentence should not, by
        # itself, trigger a mute.
        if word_count > 3 and not has_real_context:
            should_mute = False
        else:
            rescued, _parent_text = await try_reply_context_rescue(client, message, text, total)
            if rescued:
                should_mute = False
            else:
                recent_texts = conversation_tracker.get_recent_texts(
                    message.chat.id, user_id, exclude_msg_id=message.id,
                )
                # sender-history rescue only ever softens toward safe,
                # never escalates — so just check if any of their own
                # recent messages combined with this one reads as safe.
                rescued_by_history = False
                for prior_text in recent_texts:
                    combined = prior_text + " " + text
                    _l, combined_total, _h = classify(combined, weights)
                    if combined_total < 0:
                        rescued_by_history = True
                        break
                # untagged answer to someone else's recent question ("pvt
                # or govt?" -> "nanba pvt la"): same test as the reply
                # rescue, against what OTHER people said in the last 2 min.
                # Only reachable for borderline ("uncertain") scores, so a
                # confident ask like "Pvt vaa" is never rescued this way.
                if not rescued_by_history and not is_direct_ask(text):
                    for other_text in conversation_tracker.get_recent_chat_texts(
                        message.chat.id, user_id,
                    ):
                        _l, combined_total, _h = classify(other_text + " " + text, weights)
                        if combined_total < total and combined_total < config.WARN_THRESHOLD * 0.5:
                            rescued_by_history = True
                            break
                should_mute = not rescued_by_history
    elif (
        anchor_present
        and not has_real_context
        and word_count <= 2
        and message.reply_to_message_id
    ):
        # A bare anchor word used only as a reply/tag ("add", "pm") with
        # no other context of its own, AND the base score wasn't even
        # borderline-positive (the branch above only fires for total >=
        # 0) — e.g. replying to a plain "hi" with just "add". There's no
        # legitimate reason visible anywhere for this, so default to
        # mute UNLESS the parent message it's replying to supplies real,
        # unrelated context (a grocery list, a game invite, an admin
        # emergency, etc.) that genuinely explains it.
        rescued, _parent_text = await try_reply_context_rescue(client, message, text, total)
        should_mute = not rescued

    if should_mute:
        cmd_text = f".delmute {format_duration(config.DM_MUTE_HOURS)}h no dm"
        await message.reply_text(cmd_text)
        is_abusive = True
        await log_to_admin(
            client,
            f"[dm-solicitation{' (edited msg)' if is_edit else ''}] "
            f"user={user_id} score={total:.2f} -> {cmd_text}",
        )
        conversation_tracker.record_message(
            message.chat.id, message.id, user_id, message.reply_to_message_id, text, True,
        )
        return

    # --- 3. duplicate-message spam detection (only reached for messages
    #     that are otherwise clean; never for edits — an edit isn't a
    #     new send, so it must not count toward the repeat threshold) ---
    is_duplicate = (not is_edit) and conversation_tracker.check_and_record_duplicate(
        chat_id=message.chat.id, user_id=user_id, text=text,
    )
    if is_duplicate:
        spam_cmd_text = f".delmute {format_duration(config.SPAM_MUTE_HOURS)}h no spam"
        await message.reply_text(spam_cmd_text)
        is_abusive = True
        await log_to_admin(
            client,
            f"[spam-duplicate] user={user_id} -> {spam_cmd_text}",
        )
        conversation_tracker.record_message(
            message.chat.id, message.id, user_id, message.reply_to_message_id, text, True,
        )
        return

    conversation_tracker.record_message(
        message.chat.id, message.id, user_id, message.reply_to_message_id, text, is_abusive,
    )


async def warm_peer_cache(client):
    """
    Fetches the full dialog list on startup so later get_messages/
    send_message calls against chats the account hasn't actively
    opened yet don't fail with "Peer id invalid".
    """
    try:
        async for _ in client.get_dialogs():
            pass
    except Exception as e:
        print(f"[ERROR] warm_peer_cache failed: {type(e).__name__}: {e}")


async def main():
    async with app:
        await warm_peer_cache(app)
        print("Userbot moderation online.")
        await asyncio.Event().wait()


if __name__ == "__main__":
    app.run(main())
