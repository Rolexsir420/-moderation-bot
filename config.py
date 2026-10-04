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
# Same exact message text from the same user within this many hours ->
# immediate ".delmute {SPAM_MUTE_HOURS}h no spam" reply, no warn tier.
SPAM_WINDOW_HOURS = float(os.environ.get("SPAM_WINDOW_HOURS", "8"))
SPAM_MUTE_HOURS = float(os.environ.get("SPAM_MUTE_HOURS", "3"))

# --- Conversation / instigator / reply-context tracking ---
CONTEXT_WINDOW_SECONDS = int(os.environ.get("CONTEXT_WINDOW_SECONDS", "300"))
MAX_HISTORY_PER_CHAT = int(os.environ.get("MAX_HISTORY_PER_CHAT", "50"))
