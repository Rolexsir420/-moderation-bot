"""
Central configuration. Everything is driven by environment variables so the
same code runs locally and on Railway without edits. Copy .env.example to
.env for local testing, or set these as Railway service variables.
"""

import os

# --- Telegram credentials (userbot: logs in as your real account, not a
#     bot-token bot) ---
API_ID = int(os.environ.get("API_ID", "0"))
API_HASH = os.environ.get("API_HASH", "")

# One of these two is required:
#  - PHONE_NUMBER: used for the ONE-TIME interactive login (run
#    generate_session.py locally, enter the OTP Telegram texts you).
#  - SESSION_STRING: the string generate_session.py prints out after that
#    login. Set this as a Railway env var so the bot can start headless
#    on every redeploy without asking for an OTP again.
PHONE_NUMBER = os.environ.get("PHONE_NUMBER", "")
SESSION_STRING = os.environ.get("SESSION_STRING", "")

# Chat where the review queue, retraining confirmations, and instigator/
# victim action logs get posted. Must be a chat/channel the bot is in.
ADMIN_CHAT_ID = int(os.environ.get("ADMIN_CHAT_ID", "0"))

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

# --- Strike escalation (used for the instigator/abuse-argument path) ---
STRIKE_LIMIT_MUTE = int(os.environ.get("STRIKE_LIMIT_MUTE", "3"))
STRIKE_LIMIT_BAN = int(os.environ.get("STRIKE_LIMIT_BAN", "5"))
MUTE_MINUTES = int(os.environ.get("MUTE_MINUTES", "60"))

# --- DM-intention action ---
# When the classifier flags "warn" (DM solicitation intent), delete the
# message and mute the sender directly for this many hours. Fixed action,
# not escalating — no other bot needed.
DM_MUTE_HOURS = float(os.environ.get("DM_MUTE_HOURS", "3"))

# --- Conversation / instigator tracking ---
CONTEXT_WINDOW_SECONDS = int(os.environ.get("CONTEXT_WINDOW_SECONDS", "300"))
MAX_HISTORY_PER_CHAT = int(os.environ.get("MAX_HISTORY_PER_CHAT", "50"))
