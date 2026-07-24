"""
Main bot entrypoint.

Pipeline per group message:
  0. blocklist.check_text(text) -> list of (category, term) hits.
       Hardcoded, deterministic, independent of the ML classifier.
       ANY hit -> delete the message immediately and escalate a strike
       via strikes.py (warn -> mute -> ban). This exists for content
       (explicit sexual terms, slurs, caste terms, scam terms, certain
       emojis) where a bare match is essentially never innocent, so we
       don't want to wait on training data or risk a miss during the
       classifier's "uncertain" review window.
  1. classify(text) -> "warn" / "safe" / "uncertain"
       "warn" here means DM-solicitation intent (pm/dm/msg/akka/aunty/
       fwb/ex anchors) — NOT general abuse or threats, which this
       classifier doesn't cover. Only runs if step 0 found no hit.
  2. record it in conversation_tracker's rolling window (kept for any
     future abuse/argument feature; not used by the DM-intention action).
  3. "warn"      -> delete the message and mute the sender directly for
                    config.DM_MUTE_HOURS (default 3h), tagged "no dm".
                    Fixed action, no escalation, no other bot needed.
  4. "uncertain" -> forward to the admin chat with Confirm/Reject buttons.
                    Whatever the admin picks becomes new labeled training
                    data and the model retrains immediately.
  5. "safe"      -> no action.
"""

import time
import uuid
from datetime import datetime, timedelta

from pyrogram import Client, filters
from pyrogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    CallbackQuery,
    Message,
)
from pyrogram.errors import RPCError

# NOTE: this account no longer calls restrict_chat_member/ban_chat_member
# itself. It only replies with a short admin-command-style text
# (".delwarn ...", ".delmute Xh ...", ".delban ...") to the offending
# message; the group's separate admin command bot infers the target
# from that reply and performs the actual delete + mute/warn/ban.

# Short, human-admin-style reason text per blocklist category, used in
# the visible ".delwarn"/".delmute"/".delban" style reply so anyone
# checking recent group actions immediately understands why -- same
# convention as a human admin typing ".delmute 3h no dm" manually.
# Priority order matters when a message hits multiple categories at
# once: the first matching entry here wins, keeping the visible reason
# short and singular rather than listing every category.
CATEGORY_REASONS = [
    ("caste_hate", "no hate speech"),
    ("tamil_abusive", "no badwords"),
    ("sexual", "no 18+ talk in groupchat"),
    ("tamil_sexual_slang", "no 18+ talk in groupchat"),
    ("clothing_body", "no 18+ talk in groupchat"),
    ("emoji", "no 18+ talk in groupchat"),
    ("scam", "no scam/paid content"),
    ("service_paid", "no scam/paid content"),
    ("relationship_dating", "no dating txt"),
    ("gender_orientation", "restricted topic"),
]


def reason_for_categories(categories):
    cat_set = set(categories)
    for cat, reason in CATEGORY_REASONS:
        if cat in cat_set:
            return reason
    return "group rules violation"


def format_duration(n):
    """3.0 -> '3', 1.5 -> '1.5' -- matches how a human admin would type it."""
    return str(int(n)) if float(n).is_integer() else str(n)

import config
import conversation_tracker
import strikes
import blocklist
from predict import classify, load_weights
from feedback_loop import add_feedback
from trainer import extract_features as detect_solicitation_wording

if config.SESSION_STRING:
    # Headless login (Railway) — no OTP prompt needed.
    app = Client(
        "moderation_userbot",
        api_id=config.API_ID,
        api_hash=config.API_HASH,
        session_string=config.SESSION_STRING,
    )
elif config.PHONE_NUMBER:
    # Interactive login (local only) — will prompt for OTP on first run.
    app = Client(
        "moderation_userbot",
        api_id=config.API_ID,
        api_hash=config.API_HASH,
        phone_number=config.PHONE_NUMBER,
    )
else:
    raise SystemExit(
        "Set either SESSION_STRING (headless/Railway) or PHONE_NUMBER "
        "(local interactive login) in your environment. Run "
        "generate_session.py locally first to get a SESSION_STRING."
    )

# reloaded whenever feedback_loop retrains
weights = load_weights()

# review_id -> original text, waiting on an admin decision
pending_reviews = {}


def reload_weights():
    global weights
    weights = load_weights()


async def log_to_admin(text):
    if config.ADMIN_CHAT_ID:
        try:
            await app.send_message(config.ADMIN_CHAT_ID, text)
        except RPCError as e:
            print(f"[ERROR] log_to_admin send_message failed: {type(e).__name__}: {e}")


async def apply_dm_mute(client, chat_id, user_id, mention, reason):
    """
    Logs a DM-intention detection to the admin chat. The actual
    ".delmute Xh no dm" reply that triggers the real mute is sent by
    the caller (handle_group_message), since only the caller has the
    original Message object (or the instigator's stored msg_id) to
    reply to. This account does NOT restrict the member itself -- the
    group's separate admin command bot performs the real action based
    on that reply.
    """
    await log_to_admin(
        f"🔇 Requested mute for {mention} ({config.DM_MUTE_HOURS}h, no dm) — {reason}. "
        f"(Real mute performed by the group's admin command bot, not this account.)"
    )


async def apply_blocklist_action(client, message: Message, user, mention, hits):
    """
    Deterministic path: reply to the offending message with a short,
    human-admin-style command (".delwarn no badwords",
    ".delmute 60m no 18+ talk in groupchat", ".delban ..."). This
    account does NOT delete/mute/ban directly -- the group's separate
    admin command bot picks up the reply (it infers the target from
    the reply-to relationship, no @mention needed) and performs the
    actual delete + action. We only decide WHICH command to send,
    based on our own strike escalation (warn -> mute -> ban).
    """
    categories = sorted({cat for cat, term in hits})
    reason = reason_for_categories(categories)
    user_id = user.id if user else None

    if user_id is None:
        await log_to_admin(
            f"🚫 Blocklist hit ({', '.join(categories)}) but no user id — no command sent."
        )
        return

    count, action = strikes.add_strike(user_id)

    if action == "ban":
        cmd_text = f".delban {reason}"
    elif action == "mute":
        cmd_text = f".delmute {config.MUTE_MINUTES}m {reason}"
    else:
        cmd_text = f".delwarn {reason}"

    try:
        await message.reply_text(cmd_text)
    except RPCError as e:
        print(f"[ERROR] blocklist reply_text failed: {type(e).__name__}: {e}")

    # Full detail (strike count, all matched categories) still goes to
    # the admin chat, even though the group only sees the short command.
    await log_to_admin(
        f"{cmd_text} — {mention}, strike {count} (categories: {', '.join(categories)})"
    )


@app.on_message(filters.group & filters.text)
async def handle_group_message(client, message: Message):
    if config.ADMIN_CHAT_ID and message.chat.id == config.ADMIN_CHAT_ID:
        return

    if config.TARGET_CHAT_IDS:
        if message.chat.id not in config.TARGET_CHAT_IDS:
            # Restricted to specific group(s); this chat isn't one of
            # them. No debug print here either, once you've locked in a
            # target this stays quiet for every other chat.
            return
    else:
        # Discovery mode: no target set yet, log every chat so you can
        # find the id you want. Once TARGET_CHAT_IDS is set, this stops.
        print(f"[DEBUG] message from chat id={message.chat.id} title={message.chat.title!r}")

    user = message.from_user
    if user and getattr(user, "is_self", False):
        # This is a userbot, not a bot-token bot -- its own log/warning
        # messages come through this same handler like anyone else's.
        # Never moderate its own account.
        return

    text = message.text
    mention = user.mention if user else "someone"

    # Step 0: hardcoded blocklist, deterministic, checked first.
    blocklist_hits = blocklist.check_text(text)
    if blocklist_hits:
        await apply_blocklist_action(client, message, user, mention, blocklist_hits)
        conversation_tracker.record_message(
            chat_id=message.chat.id,
            msg_id=message.id,
            user_id=user.id if user else None,
            reply_to_id=message.reply_to_message_id,
            text=text,
            is_abusive=True,
        )
        return

    label, total, hits = classify(text, weights)
    anchor_present = len(detect_solicitation_wording(text)) > 0

    if label == "warn":
        is_abusive = True
    elif label == "uncertain" and anchor_present and total >= 0:
        # Score didn't clear WARN_THRESHOLD, but it's still net-positive
        # (leaning toward warn) AND DM-solicitation wording (pm/dm/msg/
        # akka/aunty/fwb/ex) is present. Per admin preference: auto-mute
        # directly rather than queue for admin review — e.g. "any akka
        # dm" (score 0.87, below 1.5 but clearly leaning warn).
        #
        # total < 0 is excluded here on purpose: a negative score means
        # the model itself leans toward safe (e.g. "enaku dm
        # pannathinga" = "please don't dm me" scores -0.93 -- correctly
        # safe-leaning, just not quite past the strict -1.5 cutoff).
        # Muting those was a real bug -- context words the model
        # recognized as negation/refusal were being overridden just
        # because an anchor word was technically present.
        is_abusive = True
        await log_to_admin(
            f"🔇 Auto-muted {mention} (score={total:.2f}, below warn threshold "
            f"but leaning warn with DM-solicitation wording present) in "
            f"{message.chat.title}:\n“{text}”"
        )
    else:
        # label == "safe", or "uncertain" with no anchor, or "uncertain"
        # with an anchor but a safe-leaning (negative) score -- no action.
        is_abusive = False

    # record into the rolling window regardless of label, so future
    # instigator lookups can see this message
    conversation_tracker.record_message(
        chat_id=message.chat.id,
        msg_id=message.id,
        user_id=user.id if user else None,
        reply_to_id=message.reply_to_message_id,
        text=text,
        is_abusive=is_abusive,
    )

    if not is_abusive:
        return

    instigator_id, instigator_msg_id = conversation_tracker.find_instigator(
        chat_id=message.chat.id,
        target_user_id=user.id if user else None,
        target_msg_id=message.id,
        reply_to_id=message.reply_to_message_id,
    )

    dm_cmd_text = f".delmute {format_duration(config.DM_MUTE_HOURS)}h no dm"

    if instigator_id:
        # this sender was reacting to someone else's DM-soliciting message
        # (per conversation_tracker) — soft note for them, no command;
        # the mute request goes to the instigator's original message
        # instead, since the admin command bot infers the target from
        # the reply-to relationship.
        try:
            await message.reply_text(
                f"Unga message remove pannitten {mention}. Please report to admins "
                f"instead of replying in the group. 🙏"
            )
        except RPCError as e:
            print(f"[ERROR] instigator soft-note reply_text failed: {type(e).__name__}: {e}")
        if instigator_msg_id:
            try:
                await client.send_message(
                    message.chat.id, dm_cmd_text, reply_to_message_id=instigator_msg_id
                )
            except RPCError as e:
                print(f"[ERROR] instigator dm_cmd reply failed: {type(e).__name__}: {e}")
        else:
            # Shouldn't happen (find_instigator only returns an id when
            # it also has the msg_id), but fall back to a plain tag so
            # nothing is silently dropped.
            try:
                await client.send_message(
                    message.chat.id, f"{dm_cmd_text} (user {instigator_id})"
                )
            except RPCError as e:
                print(f"[ERROR] instigator dm_cmd fallback failed: {type(e).__name__}: {e}")
        await apply_dm_mute(
            client, message.chat.id, instigator_id, f"user {instigator_id}",
            reason=f"DM-intention message that {mention} was reacting to",
        )
    else:
        try:
            await message.reply_text(dm_cmd_text)
        except RPCError as e:
            print(f"[ERROR] dm_mute reply_text failed: {type(e).__name__}: {e}")
        await apply_dm_mute(
            client, message.chat.id, user.id if user else None, mention,
            reason=f"“{text}” (score={total:.2f})",
        )


@app.on_callback_query(filters.regex(r"^fb_(warn|safe):"))
async def handle_admin_feedback(client, cq: CallbackQuery):
    decision, review_id = cq.data.split(":", 1)
    review = pending_reviews.pop(review_id, None)
    if not review:
        await cq.answer("Already handled or expired.", show_alert=True)
        return

    confirmed_warn = decision == "fb_warn"
    n, _ = add_feedback(
        review["text"], confirmed_warn,
        reason="admin_review_confirm" if confirmed_warn else "admin_review_false_positive",
    )
    reload_weights()

    label = "WARN" if confirmed_warn else "SAFE"
    await cq.message.edit_text(
        f"{cq.message.text}\n\n✅ Admin marked: {label} — retrained on {n} examples."
    )
    await cq.answer("Recorded and retrained.")


async def warm_peer_cache():
    """
    Fetch the full dialog list once at startup so Pyrogram caches the
    access_hash for every chat/group/channel this account is in.
    Userbots on a fresh session don't have this cached yet, which
    causes intermittent "Peer id invalid" errors on incoming updates
    from chats it hasn't explicitly fetched yet. This is a background
    warning (not fatal), but warming the cache avoids it.
    """
    count = 0
    async for _ in app.get_dialogs():
        count += 1
    print(f"Peer cache warmed — {count} chats loaded.")


async def main():
    await app.start()
    print("Starting moderation bot...")
    await warm_peer_cache()
    from pyrogram import idle
    await idle()
    await app.stop()


if __name__ == "__main__":
    app.run(main())
