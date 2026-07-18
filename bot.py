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
    ChatPermissions,
)
from pyrogram.errors import RPCError

# A fully-muted member: every permission explicitly off. Passing
# permissions=None to restrict_chat_member crashes Pyrogram (it expects
# a real ChatPermissions object to read attributes off of).
MUTED_PERMISSIONS = ChatPermissions(
    can_send_messages=False,
    can_send_media_messages=False,
    can_send_other_messages=False,
    can_send_polls=False,
    can_add_web_page_previews=False,
    can_change_info=False,
    can_invite_users=False,
    can_pin_messages=False,
)

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
        except RPCError:
            pass


async def apply_dm_mute(client, chat_id, user_id, mention, reason):
    """
    Fixed action for a confirmed DM-intention message: mute the user for
    config.DM_MUTE_HOURS. No escalation, no strike counting — every
    confirmed DM-intention message gets this same response.
    """
    until = datetime.utcnow() + timedelta(hours=config.DM_MUTE_HOURS)
    try:
        await client.restrict_chat_member(chat_id, user_id, permissions=MUTED_PERMISSIONS, until_date=until)
    except RPCError:
        pass
    await log_to_admin(
        f"🔇 Muted {mention} for {config.DM_MUTE_HOURS}h (no dm) — {reason}"
    )


async def apply_blocklist_action(client, message: Message, user, mention, hits):
    """
    Deterministic path: delete the message, escalate a strike, and act
    according to strikes.add_strike (warn -> mute -> ban), based on the
    hardcoded blocklist in blocklist.py. Always logs the hit categories
    (not the raw matched word) to the admin chat.

    A blocklist hit always follows the plain strike ladder regardless of
    whether DM-solicitation wording is also present in the same message —
    e.g. "masturbation...dm" is a warn on strike 1, same as any other
    first-time blocklist hit. (DM-solicitation on its own, with no
    blocklist word, is handled separately in handle_group_message and
    can escalate straight to mute — see the auto-mute-on-uncertain rule.)
    """
    categories = sorted({cat for cat, term in hits})
    user_id = user.id if user else None

    try:
        await message.delete()
    except RPCError:
        pass

    if user_id is None:
        await log_to_admin(
            f"🚫 Blocklist hit ({', '.join(categories)}) but no user id — message deleted, no strike applied."
        )
        return

    count, action = strikes.add_strike(user_id)

    try:
        await client.send_message(
            message.chat.id,
            f"⚠️ {mention}, your message was removed for violating group rules "
            f"({', '.join(categories)}). Strike {count}.",
        )
    except RPCError:
        pass

    if action == "ban":
        try:
            await client.ban_chat_member(message.chat.id, user_id)
        except RPCError:
            pass
        await log_to_admin(
            f"⛔ Banned {mention} — strike {count} ({', '.join(categories)})"
        )
    elif action == "mute":
        until = datetime.utcnow() + timedelta(minutes=config.MUTE_MINUTES)
        try:
            await client.restrict_chat_member(
                message.chat.id, user_id, permissions=MUTED_PERMISSIONS, until_date=until
            )
        except RPCError:
            pass
        await log_to_admin(
            f"🔇 Muted {mention} for {config.MUTE_MINUTES}m — strike {count} "
            f"({', '.join(categories)})"
        )
    else:
        await log_to_admin(
            f"🚩 Warned {mention} — strike {count} ({', '.join(categories)})"
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
    elif label == "uncertain" and anchor_present:
        # Score didn't clear WARN_THRESHOLD, but DM-solicitation wording
        # (pm/dm/msg/akka/aunty/fwb/ex) is present in the message. Per
        # admin preference: auto-mute directly rather than queue for
        # admin review — e.g. "any akka dm" (score 0.87, below 1.5).
        is_abusive = True
        await log_to_admin(
            f"🔇 Auto-muted {mention} (score={total:.2f}, below warn threshold "
            f"but DM-solicitation wording present) in {message.chat.title}:\n"
            f"“{text}”"
        )
    else:
        # label == "safe", or "uncertain" with no anchor at all (nothing
        # DM-related in the message to begin with) -- no action.
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

    instigator_id = conversation_tracker.find_instigator(
        chat_id=message.chat.id,
        target_user_id=user.id if user else None,
        target_msg_id=message.id,
        reply_to_id=message.reply_to_message_id,
    )

    # Always remove the flagged message itself.
    try:
        await message.delete()
    except RPCError:
        pass

    if instigator_id:
        # this sender was reacting to someone else's DM-soliciting message
        # (per conversation_tracker) — soft note for them, no mute; the
        # mute goes to the instigator instead.
        try:
            await client.send_message(
                message.chat.id,
                f"Unga message remove pannitten {mention}. Please report to admins "
                f"instead of replying in the group. 🙏",
            )
        except RPCError:
            pass
        await apply_dm_mute(
            client, message.chat.id, instigator_id, f"user {instigator_id}",
            reason=f"DM-intention message that {mention} was reacting to",
        )
    else:
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
