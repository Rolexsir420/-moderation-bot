"""
Learns moderation decisions from labeled (text, warn) examples.
No hardcoded word lists for context words (akka/aunty/mela/irukingala/
negation etc.) — the model figures out which context words push toward
warn/no-warn around each anchor keyword, purely from the labeled data.
(Explicit abusive/sexual/caste words are instead handled deterministically
by blocklist.py, which runs before this classifier ever sees the text.)

Approach: per-anchor Naive-Bayes-style log-odds over context words found
near that anchor. Each anchor (pm, dm, msg, akka, aunty, fwb...) gets its
own vocabulary, so "irukingala" can mean something different next to
"akka" than next to "aunty" — the data decides, not us.

Evasion handling built into extract_features():
  - digit-glued time patterns ("9pm")
  - spaced time patterns ("9 pm")
  - letter-doubling / repetition collapse ("ammme" -> "me", "aaadddd" ->
    "add" via _COLLAPSED_ANCHORS, for anchors that themselves contain a
    natural double letter)
  - glued compounds ("dmme" -> "dm" + "me")
  - split-letter anchors ("d.m", "d m" -> "dm")
"""

import json
import re
import math
import unicodedata
from collections import defaultdict

import config

ANCHORS = [
    "pm", "dm", "msg", "message", "inbox", "akka", "aunty", "fwb", "ex",
    "text", "txt", "add", "ping", "reach", "private", "personal", "ib",
    "contact", "mes",
]

WINDOW = 4          # words to look at on either side of an anchor
ALPHA = 1.0         # Laplace smoothing
BIAS = "__BIAS__"   # feature representing "anchor present, regardless of context"

# Anchors glued directly onto another short word with no space
# ("dmme" -> dm + me, "pmme" -> pm + me).
GLUED_COMPOUNDS = {
    "dmme": ("dm", "me"),
    "dmmme": ("dm", "me"),
    "pmme": ("pm", "me"),
    "pmmee": ("pm", "me"),
    "msgme": ("msg", "me"),
    "textme": ("text", "me"),
    "addme": ("add", "me"),
}

# Common abbreviations/shorthand that mean the same thing as a full
# anchor word, so they share that anchor's learned vocabulary instead of
# being invisible to the classifier.
ANCHOR_ALIASES = {
    "pvt": "private",
    "prvt": "private",
    "prv": "private",
}


def collapse_runs(tok):
    """aaa -> a, dddd -> d (collapse any run of a repeated letter to one)."""
    return re.sub(r"(.)\1+", r"\1", tok)


# Pre-collapsed lookup so e.g. "mmmeee" -> collapse_runs -> "me" matches a
# glued compound even though the raw token never equals "me" or "dmme".
_COLLAPSED_GLUED_COMPOUNDS = {}
for _k, _v in GLUED_COMPOUNDS.items():
    _ck = collapse_runs(_k)
    if _ck not in _COLLAPSED_GLUED_COMPOUNDS:
        _COLLAPSED_GLUED_COMPOUNDS[_ck] = _v

# Pre-collapsed lookup for anchors that themselves contain a natural
# double letter (e.g. "add" -> collapse_runs -> "ad"). Without this,
# letter-repetition evasion of such anchors ("aaadddd" -> "ad") could
# never match the anchor via the direct `collapsed in ANCHORS` check,
# because "ad" != "add". First match wins, de-duplicated.
_COLLAPSED_ANCHORS = {}
for _a in ANCHORS:
    _ca = collapse_runs(_a)
    if _ca not in _COLLAPSED_ANCHORS:
        _COLLAPSED_ANCHORS[_ca] = _a


_ZERO_WIDTH_RE = re.compile(
    "[​-‏‪-‮﻿⁠-⁤­]"
)
_COMBINING_RE = re.compile(r"[̀-ͯ]")


def normalize_unicode(text):
    """
    Folds "fancy font" unicode (fullwidth, mathematical bold/italic,
    circled/squared letters, etc.) and regional-indicator "flag" letters
    back down to plain ASCII, and strips zero-width/combining characters
    used to break a word up for evasion. Keeps the same behavior for
    plain ASCII text — this only changes anything when stylized/unicode
    characters are present.
    """
    if not text:
        return text
    out = []
    for ch in text:
        cp = ord(ch)
        if 0x1F1E6 <= cp <= 0x1F1FF:
            out.append(chr(ord("a") + (cp - 0x1F1E6)))
        else:
            out.append(ch)
    text = "".join(out)
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _COMBINING_RE.sub("", text)
    return text


_WS_SPLIT_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9']")


def tokenize(text):
    # keep digits glued to letters ("9pm" stays "9pm", not stripped to "pm")
    # so the model can actually learn "digit+pm/am = time" from data.
    #
    # Split on WHITESPACE first, then strip punctuation out of each
    # whitespace-delimited chunk, rather than letting punctuation itself
    # act as a token separator. This closes an evasion gap: a plain
    # regex-findall over the whole string treats "..a...ddd" (dots used
    # to break up "add" without any actual spaces) as two unrelated
    # tokens ("a", "ddd"), which no downstream collapse/merge step can
    # reassemble (merge_split_anchors only reassembles runs of true
    # SINGLE-character tokens). Cleaning punctuation out of each
    # whitespace chunk instead turns "..a...ddd" into one token "addd",
    # which collapse_runs/._COLLAPSED_ANCHORS already handles correctly.
    # This is a strict superset of the old behavior for ordinary text:
    # when punctuation only ever appears BETWEEN whitespace-separated
    # words (the normal case), the result is identical either way.
    text = normalize_unicode(text)
    text = text.lower()
    tokens = []
    for chunk in _WS_SPLIT_RE.split(text.strip()):
        cleaned = _NON_ALNUM_RE.sub("", chunk)
        if cleaned:
            tokens.append(cleaned)
    return tokens


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


_TIME_GLUED_RE = re.compile(r"^\d{1,2}(am|pm)$")


def extract_features(text):
    """
    Returns list of (anchor, feature) pairs found in this text.
    feature is either a nearby context word or one of the evasion flags
    (BIAS, DIGIT_GLUED, TIME_SPACED, PUNCT_SPLIT_ANCHOR,
    LETTER_DOUBLED_ANCHOR, GLUED_EXTRA:<word>).
    """
    raw_tokens = tokenize(text)
    tokens, split_flags = merge_split_anchors(raw_tokens)
    found = []

    for i, tok in enumerate(tokens):
        base = None
        is_time_glued = False
        is_time_spaced = False
        is_letter_doubled = False
        glued_extra = None

        m = _TIME_GLUED_RE.match(tok)
        if m:
            base = m.group(1)
            is_time_glued = True
        elif tok in ANCHORS:
            base = tok
        elif tok in ANCHOR_ALIASES:
            base = ANCHOR_ALIASES[tok]
        elif tok in ("pm", "am") and i > 0 and re.match(r"^\d{1,2}$", tokens[i - 1]):
            # spaced time reference: "9 pm now" treated like "9pm now"
            base = tok
            is_time_spaced = True
        elif tok in GLUED_COMPOUNDS:
            base, glued_extra = GLUED_COMPOUNDS[tok]
        else:
            collapsed = collapse_runs(tok)
            if collapsed in ANCHORS:
                base = collapsed
                is_letter_doubled = True
            elif collapsed in _COLLAPSED_ANCHORS:
                base = _COLLAPSED_ANCHORS[collapsed]
                is_letter_doubled = True
            elif collapsed in GLUED_COMPOUNDS:
                base, glued_extra = GLUED_COMPOUNDS[collapsed]
            elif collapsed in _COLLAPSED_GLUED_COMPOUNDS:
                base, glued_extra = _COLLAPSED_GLUED_COMPOUNDS[collapsed]

        if base is None:
            continue

        found.append((base, BIAS))
        if is_time_glued:
            found.append((base, "DIGIT_GLUED"))
        if is_time_spaced:
            found.append((base, "TIME_SPACED"))
        if is_letter_doubled:
            found.append((base, "LETTER_DOUBLED_ANCHOR"))
        if glued_extra:
            found.append((base, glued_extra))
        if split_flags[i]:
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
    import os
    if not os.path.exists(path):
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
