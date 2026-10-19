"""
Central configuration. Everything is driven by environment variables so the
same code runs locally and on Railway without edits. Copy .env.example to
.env for local testing, or set these as Railway service variables.
"""

import os

# --- Telegram credentials (USERBOT login, not a bot token) ---
API_ID = int(os.environ.get("API_ID", "0"))
API_HASH = os.environ.get("API_HASH", "")

# Either SESSION_STRING (preferred for Railway) OR PHONE_NUMBER (for a local
# interactive login that creates the session the first time) must be set.
SESSION_STRING = os.environ.get("SESSION_STRING", "")
PHONE_NUMBER = os.environ.get("PHONE_NUMBER", "")

# Chat where review/strike/debug logs get posted. Must be a chat the
# userbot account is a member of.
ADMIN_CHAT_ID = int(os.environ.get("ADMIN_CHAT_ID", "0"))

# Comma-separated list of chat ids the bot should actually moderate.
# Leave empty to run in "discovery mode": the bot logs every chat id it
# sees (via [DEBUG] lines) but takes no moderation action anywhere, so you
# can find the right id and then lock it down here.
_raw_target_ids = os.environ.get("TARGET_CHAT_IDS", "")
TARGET_CHAT_IDS = {
    int(x.strip()) for x in _raw_target_ids.split(",") if x.strip()
}

# --- Persistent storage ---
# On Railway: create a Volume, mount it at e.g. /data, and set DATA_DIR=/data.
# Without a mounted volume, files here are wiped on every redeploy.
DATA_DIR = os.environ.get("DATA_DIR", ".")
os.makedirs(DATA_DIR, exist_ok=True)

WEIGHTS_PATH = os.path.join(DATA_DIR, "weights.json")
LABELED_DATA_PATH = os.path.join(DATA_DIR, "labeled_data.jsonl")
STRIKES_PATH = os.path.join(DATA_DIR, "strikes.json")

# --- Classifier thresholds ---
WARN_THRESHOLD = float(os.environ.get("WARN_THRESHOLD", "1.5"))
SAFE_THRESHOLD = float(os.environ.get("SAFE_THRESHOLD", "-1.5"))

# --- Strike bookkeeping (admin-log only; blocklist hits never escalate the
# command sent — they are always .delwarn, see bot.py) ---
STRIKE_LIMIT_MUTE = int(os.environ.get("STRIKE_LIMIT_MUTE", "3"))
STRIKE_LIMIT_BAN = int(os.environ.get("STRIKE_LIMIT_BAN", "5"))
MUTE_MINUTES = int(os.environ.get("MUTE_MINUTES", "60"))

# --- DM-solicitation action ---
# When the classifier flags DM-solicitation intent, the bot REPLIES with
# ".delmute {DM_MUTE_HOURS}h no dm" for the other admin bot to act on.
# The bot itself never calls restrict/ban/delete directly.
DM_MUTE_HOURS = float(os.environ.get("DM_MUTE_HOURS", "3"))

# --- Duplicate-message spam detection ---
# Same exact message text from the same user, repeated at least
# SPAM_REPEAT_THRESHOLD times within SPAM_WINDOW_HOURS hours ->
# ".delmute {SPAM_MUTE_HOURS}h no spam" reply, no warn tier. Raised from
# "any 2nd repeat" to "8-10 repeats" per explicit request — ordinary
# chit-chat (greetings, short replies) naturally repeats throughout the
# day and shouldn't be treated as flood/spam on the 2nd occurrence.
#
# Window widened from 8h to a full day (24h): this is specifically for
# the SLOW-drip spam pattern (same recruitment/ad message reposted with
# ~1 hour gaps across a whole day) that a short rapid-fire flood filter
# wouldn't catch. Fast repeat-within-minutes flooding is already handled
# by the group's separate admin/help bot — this is a second, independent
# net for the slower pattern, not a replacement for that one.
SPAM_WINDOW_HOURS = float(os.environ.get("SPAM_WINDOW_HOURS", "24"))
SPAM_MUTE_HOURS = float(os.environ.get("SPAM_MUTE_HOURS", "3"))
SPAM_REPEAT_THRESHOLD = int(os.environ.get("SPAM_REPEAT_THRESHOLD", "8"))
# spam counts are saved to the Volume at most this often (seconds)
SPAM_SAVE_SECONDS = float(os.environ.get("SPAM_SAVE_SECONDS", "30"))

# --- Review log (decisions saved for the admin to label; never auto-trains) ---
REVIEW_LOG_PATH = os.path.join(DATA_DIR, "review_log.jsonl")
REVIEW_LOG_ENABLED = os.environ.get("REVIEW_LOG_ENABLED", "1") not in ("0", "false", "False", "")
REVIEW_LOG_MAX_MB = float(os.environ.get("REVIEW_LOG_MAX_MB", "5"))

# --- Conversation / instigator / reply-context tracking ---
CONTEXT_WINDOW_SECONDS = int(os.environ.get("CONTEXT_WINDOW_SECONDS", "300"))
# 300 messages is roughly the last 30 minutes in a 10-15k messages/day group
MAX_HISTORY_PER_CHAT = int(os.environ.get("MAX_HISTORY_PER_CHAT", "300"))
