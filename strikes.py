"""
Simple persistent strike counter with escalation. One JSON file, no DB
needed for this scale. Mount config.DATA_DIR on a Railway volume so this
survives redeploys.
"""

import json
import os

import config


def _load():
    if not os.path.exists(config.STRIKES_PATH):
        return {}
    with open(config.STRIKES_PATH) as f:
        return json.load(f)


def _save(data):
    with open(config.STRIKES_PATH, "w") as f:
        json.dump(data, f, indent=2)


def add_strike(user_id):
    """Returns (new_count, action) where action is 'warn', 'mute', or 'ban'."""
    data = _load()
    key = str(user_id)
    data[key] = data.get(key, 0) + 1
    _save(data)
    count = data[key]
    if count >= config.STRIKE_LIMIT_BAN:
        return count, "ban"
    if count >= config.STRIKE_LIMIT_MUTE:
        return count, "mute"
    return count, "warn"


def get_strikes(user_id):
    return _load().get(str(user_id), 0)


def reset_strikes(user_id):
    data = _load()
    data.pop(str(user_id), None)
    _save(data)
