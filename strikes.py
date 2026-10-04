"""
Persistent per-user strike counts, kept purely for the admin log / audit
trail. Since the architecture pivot to reply-only moderation, strike count
no longer changes which command the bot sends for a blocklist hit — those
are ALWAYS ".delwarn {reason}", never escalating to delmute/delban no
matter how many strikes a user has. Only DM-solicitation gets a real mute
action, and that is a fixed DM_MUTE_HOURS, not strike-based either.

Kept anyway because it's useful for admins to see "this person has been
warned N times" in the admin chat log.
"""

import json
import os

import config


def _load():
    if not os.path.exists(config.STRIKES_PATH):
        return {}
    try:
        with open(config.STRIKES_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data):
    with open(config.STRIKES_PATH, "w") as f:
        json.dump(data, f, indent=2)


def add_strike(user_id, category=None):
    """
    Increments and persists the strike count for user_id. Returns the new
    total. category is recorded for the admin log only.
    """
    data = _load()
    key = str(user_id)
    record = data.get(key, {"count": 0, "categories": {}})
    record["count"] += 1
    if category:
        record["categories"][category] = record["categories"].get(category, 0) + 1
    data[key] = record
    _save(data)
    return record["count"]


def get_strikes(user_id):
    data = _load()
    return data.get(str(user_id), {"count": 0, "categories": {}})
