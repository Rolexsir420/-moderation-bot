"""
Wire this into the admin review-queue button handler in bot.py. Every admin
correction becomes a new labeled example and the model retrains immediately
— no manual word-listing needed.
"""

import json

import config
from trainer import train, load_examples


def add_feedback(text: str, admin_confirmed_warn: bool, reason: str = "admin_feedback"):
    """
    Call this from the admin review-queue handler:
      admin confirms "warn" on an uncertain/flagged message -> add_feedback(text, True, reason)
      admin confirms "safe" (false positive)                -> add_feedback(text, False, reason)
    """
    record = {"text": text, "warn": admin_confirmed_warn, "reason": reason}
    with open(config.LABELED_DATA_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return retrain()


def retrain():
    examples = load_examples(config.LABELED_DATA_PATH)
    weights = train(examples)
    with open(config.WEIGHTS_PATH, "w") as f:
        json.dump(weights, f, indent=2)
    return len(examples), weights


if __name__ == "__main__":
    n, _ = add_feedback("dm pannunga bro tell timing", False, "false_positive: asking about a call slot")
    print("retrained on", n, "examples")
