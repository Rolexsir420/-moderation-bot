"""
Run this ONCE, locally, on your own machine — not on Railway.

Railway can't show you a prompt to type in the OTP Telegram texts you,
so the login handshake has to happen interactively first. This script
logs you in with your real phone number, then prints a SESSION_STRING.
Copy that string into Railway's env vars (SESSION_STRING=...) and
bot.py will start headless from then on, no OTP needed again — until
the session is revoked (e.g. you log out of it from Telegram's
Settings -> Devices, or don't use it for a long time).

Usage:
    python generate_session.py

You'll be asked for:
  - your phone number (international format, e.g. +91XXXXXXXXXX)
  - the OTP Telegram sends you
  - your 2FA password, ONLY if you have Two-Step Verification enabled

SECURITY NOTE: the printed session string is equivalent to your login
session — anyone who has it can act as your account without needing
your phone or password again. Treat it like a password:
  - never commit it to git / paste it in a public place
  - only put it in Railway's private env vars
  - if it ever leaks, immediately revoke it from Telegram Settings ->
    Devices -> the matching active session -> Terminate.
"""

from pyrogram import Client

import config

if __name__ == "__main__":
    if not config.API_ID or not config.API_HASH:
        raise SystemExit("Set API_ID and API_HASH in your environment first.")

    app = Client(
        "moderation_userbot_login",
        api_id=config.API_ID,
        api_hash=config.API_HASH,
        phone_number=config.PHONE_NUMBER or None,  # prompts if not set
    )

    with app:
        session_string = app.export_session_string()

    print("\nLogin successful. Copy the line below into Railway's env vars:\n")
    print(f"SESSION_STRING={session_string}\n")
    print("Keep this secret — it grants full access to your Telegram account.")
