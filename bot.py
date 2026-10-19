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

from pyrogram.raw.functions.updates import GetState
from pyrogram import Client, filters
from pyrogram.types import Message

import config
import blocklist
import conversation_tracker
import strikes
import review_log
from predict import classify, load_weights
from trainer import (is_direct_ask, parent_explains, is_cant_complaint,
                     context_is_about_messages, mentions_someones_message, is_refusal,
                     is_strong_ask, is_asking_around)

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

# Anchors that are a DM ask even as a bare one-word reply ("add", "dm",
# "pm", "pvt" replying to "hi"). Deliberately NOT here: "message", "personal",
# "contact", "reach", "akka", "aunty", "ex", "mes" — as a bare
# reply those are almost always ordinary chat ("Message pathiya", "No
# personal", "Personal" replying to "Enga?"), and the group's own history
# showed them being muted wrongly by the bare-reply rule.
BARE_REPLY_ANCHORS = {"add", "dm", "pm", "ib", "inbox", "msg", "txt", "text", "ping", "private"}


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
    await send_cmd(message, cmd_text)
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


_parent_cache = {}   # (chat_id, msg_id) -> parent text, last 500 lookups


def _msg_text(m):
    return (getattr(m, "text", None) or getattr(m, "caption", None) or "") if m else ""


async def get_parent_text(client, message):
    """
    Text of the message this one replies to, or None if it isn't a reply
    or the parent can't be read. Avoids Telegram calls wherever possible:
    Pyrogram usually attaches the parent already (message.reply_to_message),
    and our own chat history has recent messages. Only when neither has it
    do we ask Telegram, once per message, so a busy group no longer hits
    "Waiting for N seconds (required by channels.GetMessages)".
    """
    rid = message.reply_to_message_id
    if not rid:
        return None
    key = (message.chat.id, message.id)
    if key in _parent_cache:
        return _parent_cache[key]
    text = None
    if getattr(message, "reply_to_message", None) is not None:
        text = _msg_text(message.reply_to_message)
    if text is None:
        rec = conversation_tracker.get_record(message.chat.id, rid)
        if rec is not None:
            text = rec["text"] or ""
    if text is None:
        try:
            text = _msg_text(await client.get_messages(message.chat.id, rid))
        except Exception as e:
            print(f"[ERROR] get_parent_text failed: {type(e).__name__}: {e}")
            text = None
    if len(_parent_cache) >= 500:
        _parent_cache.clear()
    _parent_cache[key] = text
    return text


async def review(client, message, text, action, reason, *, score=None,
                 categories=None, parent_text=None, is_edit=False):
    """Save this decision to the review log (never affects the decision)."""
    try:
        if parent_text is None:
            parent_text = await get_parent_text(client, message)
        review_log.log(
            message=message, text=text, action=action, reason=reason, score=score,
            categories=categories, reply_to_text=parent_text, is_edit=is_edit,
            prev_messages=conversation_tracker.get_last_messages(
                message.chat.id, n=5, exclude_msg_id=message.id),
        )
    except Exception as e:
        print(f"[ERROR] review failed: {type(e).__name__}: {e}")


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
    parent_text = await get_parent_text(client, message)
    if parent_text is None:
        return False, None
    if not parent_text:
        return False, None
    if parent_explains(parent_text, text):
        # "Game vaanga" <- "Add me": the parent gives "add" its ordinary
        # meaning (a game/team/list/ticket), so the reply is answering it.
        return True, parent_text

    combined_text = parent_text + " " + text
    _combined_label, combined_total, _combined_hits = classify(combined_text, weights)
    if not _combined_hits:
        # every DM word got switched off by a meaning rule because of the
        # parent's words ("group exams" <- "Addd panuga"): not a real
        # rescue; meaning-based rescues go through parent_explains above
        return False, parent_text
    rescued = (
        combined_total < original_total
        and combined_total < (config.WARN_THRESHOLD * 0.5)
    )
    return rescued, parent_text


# Text messages AND media (photo/video/document/...) that carry a caption.
# filters.text alone skipped captions, so "dm me" written on a photo was
# never checked.
MODERATED = filters.group & (filters.text | filters.caption) & ~filters.bot


# --- heartbeat: proves messages are actually arriving -----------------
# Every 10 minutes the log shows how many group messages Telegram delivered
# (per chat id) and how many the bot acted on. "seen 0" while the group is
# busy means Telegram isn't sending updates; a chat id that isn't in
# TARGET_CHAT_IDS means the bot is ignoring that group on purpose.
_seen = {}
_acted = 0


@app.on_message(filters.group, group=-1)
async def _count_seen(client, message: Message):
    _seen[message.chat.id] = _seen.get(message.chat.id, 0) + 1


async def send_cmd(message, cmd_text):
    """Reply with the Oreo Queen command; a failure is printed, not hidden."""
    global _acted
    try:
        await message.reply_text(cmd_text)
        _acted += 1
    except Exception as e:
        print(f"[ERROR] could not send '{cmd_text}' in chat {message.chat.id}: "
              f"{type(e).__name__}: {e}", flush=True)


async def heartbeat():
    global _acted
    while True:
        await asyncio.sleep(600)
        chats = ", ".join(f"{cid}: {n}" for cid, n in sorted(_seen.items(), key=lambda x: -x[1])[:5]) or "none"
        print(f"[heartbeat] last 10 min: messages seen {sum(_seen.values())} ({chats}) | actions {_acted} "
              f"| watching {sorted(config.TARGET_CHAT_IDS) or 'ALL (discovery mode)'}", flush=True)
        if not _seen:
            # Telegram sends a session's updates only to the connection that
            # last asked for them; asking again pulls them back here
            try:
                await app.invoke(GetState())
                print("[heartbeat] no messages arrived - asked Telegram to resume updates", flush=True)
            except Exception as e:
                print(f"[ERROR] resume updates failed: {type(e).__name__}: {e}", flush=True)
        _seen.clear()
        _acted = 0


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
        await review(client, message, text,
                     "ban" if categories & SEVERE_CATEGORIES else "warn", "blocklist",
                     categories=categories, is_edit=is_edit)
        conversation_tracker.record_message(
            message.chat.id, message.id, user_id, message.reply_to_message_id, text, True,
        )
        return

    # --- 1a. refusals and rule talk: "Don't pm me", "DM not allowed",
    #     "dm pannathinga", "text panatha" — never a request ---
    if is_refusal(text):
        await review(client, message, text, "none", "refusal_or_rule", is_edit=is_edit)
        conversation_tracker.record_message(
            message.chat.id, message.id, user_id, message.reply_to_message_id, text, False,
        )
        return

    # --- 1b. "can't / didn't" replies inside a conversation about messages ---
    # "Msg panna mudiyala" answering "enna pa inaiku un msg ye kaanom" is
    # part of that conversation: no action. The same words out of the blue
    # are checked by the classifier like anything else.
    if is_cant_complaint(text):
        parent_text = (await get_parent_text(client, message)) or ""
        # the replied-to message: any DM word or topic word makes it related;
        # recent chat from others (busy group): only clear topic words, 2 min
        recent = conversation_tracker.get_recent_chat_texts(
            message.chat.id, user_id, within_seconds=120, n=5,
        )
        if (context_is_about_messages([parent_text])
                or context_is_about_messages(recent, include_dm_words=False)
                or mentions_someones_message(text)):
            await review(client, message, text, "none", "cant_reply_in_conversation",
                         parent_text=parent_text, is_edit=is_edit)
            conversation_tracker.record_message(
                message.chat.id, message.id, user_id, message.reply_to_message_id, text, False,
            )
            return

    # --- 2. learned classifier for DM-solicitation intent ---
    label, total, hits = classify(text, weights)
    anchor_present = len(hits) > 0
    word_count = len(text.split())
    has_real_context = any(feat != "__BIAS__" for _, feat, _ in hits)

    should_mute = False
    reason = "below_threshold"
    _parent_text = None

    if is_strong_ask(text):
        # "dm pannunga", "pm me", "pvt vaa": always a DM ask, even when
        # other words in the sentence pull the learned score down
        should_mute = True
        reason = "strong_dm_ask"

    if should_mute:
        pass
    elif label == "warn":
        rescued, _parent_text = await try_reply_context_rescue(client, message, text, total)
        if not rescued:
            should_mute = True
            reason = "confident_dm_ask"
        else:
            reason = "confident_but_reply_rescued"
    elif label == "uncertain" and anchor_present and total >= 0:
        # a bare anchor word (message/inbox/private/personal/add/etc.)
        # in an otherwise long, unrelated sentence should not, by
        # itself, trigger a mute.
        if word_count > 3 and not has_real_context:
            should_mute = False
            reason = "lone_dm_word_in_long_sentence"
        else:
            rescued, _parent_text = await try_reply_context_rescue(client, message, text, total)
            if rescued:
                should_mute = False
                reason = "borderline_reply_rescued"
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
                    # only a real explanation counts: in a busy group a random
                    # earlier line ("U from", "nanba pvt la") lowering the
                    # combined score is not a reason to let "Dm" go
                    if parent_explains(prior_text, text):
                        rescued_by_history = True
                        reason = "borderline_own_history_rescued"
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
                        # busy group: someone ELSE's random line ("U from")
                        # must not excuse a bare "Dm"; only a message that
                        # gives the word its ordinary meaning ("pvt or govt?")
                        if parent_explains(other_text, text):
                            rescued_by_history = True
                            reason = "borderline_chat_rescued"
                            break
                should_mute = not rescued_by_history
                if should_mute:
                    reason = "borderline_dm_ask"
    elif (
        anchor_present
        and not has_real_context
        and word_count <= 2
        and message.reply_to_message_id
        and {a for a, _f, _w in hits} & BARE_REPLY_ANCHORS
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
        reason = "bare_dm_word_reply" if should_mute else "bare_dm_word_reply_rescued"

    if should_mute:
        cmd_text = f".delmute {format_duration(config.DM_MUTE_HOURS)}h no dm"
        await send_cmd(message, cmd_text)
        is_abusive = True
        await log_to_admin(
            client,
            f"[dm-solicitation{' (edited msg)' if is_edit else ''}] "
            f"user={user_id} score={total:.2f} -> {cmd_text}",
        )
        await review(client, message, text, "mute", reason, score=total,
                     parent_text=_parent_text, is_edit=is_edit)
        conversation_tracker.record_message(
            message.chat.id, message.id, user_id, message.reply_to_message_id, text, True,
        )
        return

    # --- 2b. asking around for women, without a DM ask: warn ---
    # "Any girls here", "Any akka irukingala", "Any ladies for friendship"
    if is_asking_around(text):
        cmd_text = ".delwarn no dating txt"
        await send_cmd(message, cmd_text)
        await log_to_admin(client, f"[asking-around] user={user_id} -> {cmd_text}")
        await review(client, message, text, "warn", "asking_around_for_women", is_edit=is_edit)
        conversation_tracker.record_message(
            message.chat.id, message.id, user_id, message.reply_to_message_id, text, True,
        )
        return

    if anchor_present:
        # a DM-type word the bot let through: worth a human look later
        await review(client, message, text, "none", reason, score=total,
                     parent_text=_parent_text, is_edit=is_edit)

    # --- 3. duplicate-message spam detection (only reached for messages
    #     that are otherwise clean; never for edits — an edit isn't a
    #     new send, so it must not count toward the repeat threshold) ---
    is_duplicate = (not is_edit) and conversation_tracker.check_and_record_duplicate(
        chat_id=message.chat.id, user_id=user_id, text=text,
    )
    if is_duplicate:
        spam_cmd_text = f".delmute {format_duration(config.SPAM_MUTE_HOURS)}h no spam"
        await send_cmd(message, spam_cmd_text)
        is_abusive = True
        await log_to_admin(
            client,
            f"[spam-duplicate] user={user_id} -> {spam_cmd_text}",
        )
        await review(client, message, text, "spam", "repeated_message", is_edit=is_edit)
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
        print(f"Userbot moderation online. Watching chats: "
              f"{sorted(config.TARGET_CHAT_IDS) or 'ALL (discovery mode)'}", flush=True)
        asyncio.create_task(heartbeat())
        await asyncio.Event().wait()


if __name__ == "__main__":
    app.run(main())
