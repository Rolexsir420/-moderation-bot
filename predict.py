import json
import os

import config
from trainer import extract_features


def load_weights(path=None):
    path = path or config.WEIGHTS_PATH
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def score(text, weights):
    feats = extract_features(text)
    total = 0.0
    hits = []
    for anchor, feat in feats:
        w = weights.get(anchor, {}).get(feat)
        if w is not None:
            total += w
            hits.append((anchor, feat, round(w, 2)))
    return total, hits


def classify(text, weights):
    """
    Returns (label, total, hits) where label is one of:
      "warn"      -> confident this is unwanted solicitation/abuse
      "safe"      -> confident this is fine
      "uncertain" -> send to admin review queue, don't auto-act
    """
    total, hits = score(text, weights)
    if total >= config.WARN_THRESHOLD:
        return "warn", total, hits
    if total <= config.SAFE_THRESHOLD:
        return "safe", total, hits
    return "uncertain", total, hits


if __name__ == "__main__":
    weights = load_weights()

    tests = [
        ("pm pannuga", "warn"),
        ("dm pannunga", "warn"),
        ("please dm me", "warn"),
        ("pm pannatha", "safe"),
        ("dm pannathinga please", "safe"),
        ("no enaku dm pannathinga", "safe"),
        ("9pm pannuga", "safe"),
        ("any akka msg me", "warn"),
        ("any akka irukingala", "safe"),
        ("akka mela msg paaru", "safe"),
        ("any aunty irukingala", "warn"),
        ("nice group akka thanks", "safe"),
        ("fwb only interested", "warn"),
    ]

    passed = 0
    for text, expected in tests:
        label, total, hits = classify(text, weights)
        ok = "PASS" if label == expected else "FAIL"
        if label == expected:
            passed += 1
        print(f"[{ok}] '{text}' -> {label} (score={total:.2f}, expected={expected})")

    print(f"\n{passed}/{len(tests)} passed")
