"""
Scores a message against the trained per-anchor weights.

classify(text, weights) -> (label, total, hits)
  label: "warn" | "safe" | "uncertain"
  total: summed log-odds score across all anchors found in the text
  hits:  list of (anchor, feature, weight) tuples that contributed,
         so callers can inspect *why* (e.g. was it bias-only, or did
         real context words fire).
"""

import config
from trainer import extract_features


def classify(text, weights):
    feats = extract_features(text)
    hits = []
    total = 0.0

    for anchor, feat in feats:
        anchor_weights = weights.get(anchor)
        if not anchor_weights:
            continue
        w = anchor_weights.get(feat)
        if w is None:
            continue
        total += w
        hits.append((anchor, feat, w))

    if total >= config.WARN_THRESHOLD:
        label = "warn"
    elif total <= config.SAFE_THRESHOLD:
        label = "safe"
    else:
        label = "uncertain"

    return label, total, hits


def anchor_present(hits):
    return len(hits) > 0


def load_weights(path=None):
    import json
    import os
    path = path or config.WEIGHTS_PATH
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
