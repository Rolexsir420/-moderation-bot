"""
Learns moderation decisions from labeled (text, warn) examples.
No hardcoded word lists for akka/aunty/mela/irukingala/negation —
the model figures out which context words push toward warn/no-warn
around each anchor keyword, purely from the labeled data.

Approach: per-anchor Naive-Bayes-style log-odds over context words
found near that anchor. Each anchor (pm, dm, msg, akka, aunty, fwb...)
gets its own vocabulary, so "irukingala" can mean something different
next to "akka" than next to "aunty" — the data decides, not us.
"""

import json
import re
import math
from collections import defaultdict

import config

ANCHORS = ["pm", "dm", "msg", "message", "inbox", "akka", "aunty", "fwb", "ex", "text", "txt"]
WINDOW = 4          # words to look at on either side of an anchor
ALPHA = 1.0         # Laplace smoothing
BIAS = "__BIAS__"   # feature representing "anchor present, regardless of context"


def tokenize(text):
    # keep digits glued to letters ("9pm" stays "9pm", not stripped to "pm")
    # so the model can actually learn "digit+pm/am = time" from data
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def merge_split_anchors(tokens):
    """
    Catches anchors typed with punctuation/spaces between each letter
    ("d.m", "d m", "f.w.b") which the regex tokenizer otherwise splits
    into single-character tokens that never match an anchor.
    Returns (merged_tokens, flags) where flags[i] is True if that
    position was reconstructed from split letters.
    """
    result, flags = [], []
    i, n = 0, len(tokens)
    while i < n:
        merged = False
        for anchor in ANCHORS:
            L = len(anchor)
            if L < 2 or i + L > n:
                continue
            candidate = tokens[i:i + L]
            if all(len(t) == 1 for t in candidate) and "".join(candidate) == anchor:
                result.append(anchor)
                flags.append(True)
                i += L
                merged = True
                break
        if not merged:
            result.append(tokens[i])
            flags.append(False)
            i += 1
    return result, flags


def collapse_runs(s):
    """
    'pmm' -> 'pm', 'inboxx' -> 'inbox'. Collapses any run of 2+ identical
    consecutive letters down to one, so doubled-letter evasion ("Pmm",
    "Inboxx vadi") still resolves to the real anchor.
    """
    return re.sub(r"(.)\1+", r"\1", s)


def extract_features(text):
    """
    Returns list of (anchor, feature) pairs found in this text.
    feature is either a nearby context word or BIAS.
    """
    raw_tokens = tokenize(text)
    tokens, split_flags = merge_split_anchors(raw_tokens)
    found = []
    time_pattern = re.compile(r"^\d{1,2}(am|pm)$")

    for i, tok in enumerate(tokens):
        base = tok
        is_time_glued = False
        is_letter_doubled = False

        m = time_pattern.match(tok)
        if m:
            base = m.group(1)
            is_time_glued = True
        elif tok not in ANCHORS:
            collapsed = collapse_runs(tok)
            if collapsed in ANCHORS:
                base = collapsed
                is_letter_doubled = True

        if base in ANCHORS:
            found.append((base, BIAS))
            if is_time_glued:
                found.append((base, "DIGIT_GLUED"))
            if is_letter_doubled:
                # so the model can learn whether this evasion pattern
                # itself correlates with warn/safe, same idea as
                # PUNCT_SPLIT_ANCHOR below
                found.append((base, "LETTER_DOUBLED_ANCHOR"))
            if split_flags[i]:
                # so the model can learn from data whether this evasion
                # pattern itself correlates with warn/safe
                found.append((base, "PUNCT_SPLIT_ANCHOR"))
            lo, hi = max(0, i - WINDOW), min(len(tokens), i + WINDOW + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                found.append((base, tokens[j]))
    return found


def load_examples(path=None):
    path = path or config.LABELED_DATA_PATH
    examples = []
    if not __import__("os").path.exists(path):
        return examples
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def train(examples):
    counts = defaultdict(lambda: defaultdict(lambda: {"warn": 0, "safe": 0}))
    vocab_size = defaultdict(set)

    for ex in examples:
        label = "warn" if ex["warn"] else "safe"
        feats = extract_features(ex["text"])
        for anchor, feat in feats:
            counts[anchor][feat][label] += 1
            vocab_size[anchor].add(feat)

    weights = {}
    for anchor, feat_counts in counts.items():
        V = len(vocab_size[anchor])
        total_warn = sum(c["warn"] for c in feat_counts.values())
        total_safe = sum(c["safe"] for c in feat_counts.values())
        weights[anchor] = {}
        for feat, c in feat_counts.items():
            p_warn = (c["warn"] + ALPHA) / (total_warn + ALPHA * V)
            p_safe = (c["safe"] + ALPHA) / (total_safe + ALPHA * V)
            weights[anchor][feat] = math.log(p_warn / p_safe)

    return weights


def main():
    examples = load_examples(config.LABELED_DATA_PATH)
    weights = train(examples)
    with open(config.WEIGHTS_PATH, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"Trained on {len(examples)} examples.")
    print(f"Learned anchors: {list(weights.keys())}")


if __name__ == "__main__":
    main()
