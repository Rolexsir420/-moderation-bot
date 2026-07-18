"""
Hardcoded categorized word/emoji blocklist.

This is DELIBERATELY separate from trainer.py / predict.py. The ML
classifier in predict.py is for judgment calls (does "akka msg me" mean
solicitation or not) where context matters and the answer isn't obvious
from the word alone. This module is for the opposite case: words where
a bare match is basically never innocent, so we don't want to wait for
training data or risk a miss during the "uncertain" review window.

Any hit here is treated as an immediate, deterministic violation:
delete the message + escalate a strike. No scoring, no thresholds.

HOW TO EDIT
-----------
Add/remove words directly in the CATEGORIES dict below. Keep every entry
lowercase. Multi-word phrases (e.g. "fake id") are matched as substrings;
single words are matched on token boundaries so "class" doesn't match "ass".
"""

import re

# --- Category -> set of terms -----------------------------------------

CATEGORIES = {
    "sexual": {
        "sex", "sexy", "porn", "pornography", "nude", "boobs", "penis",
        "pussy", "dick", "ass", "buttocks", "fingering", "foreplay",
        "masturbation", "lick", "kiss", "fucking", "fucked",
    },
    "relationship_dating": {
        "hookup", "hookups", "fwb", "wife", "married", "darling", "baby",
        "milf", "bhabhi", "housewife", "available", "virgin",
    },
    "gender_orientation": {
        "lesbian", "lesbians", "gay", "gays", "bisexual", "transgender",
        "shemale", "trans", "aravaani", "vaanavil",
    },
    "tamil_abusive": {
        "thevidiya", "thevudiya", "tevadiya", "tevudiya", "thevdiya",
        "thevadiya",
        "thayoli", "gommala", "gomma", "kandaroli", "mayiru", "mayirandi",
        "mairu", "mairru", "ommala", "oombu", "oombi", "ombuga", "oompi",
        "ottha", "otha", "olumaari", "olu maari", "koothi", "lavade",
        "punda", "punde", "potta", "funda", "ungoppa", "ungoppan",
        "bastard", "fucker",
    },
    "tamil_sexual_slang": {
        "sunni", "junni", "jumni", "kunju", "kundi", "kundiadi", "kuthi",
        "sappu", "umbu", "oolu",
        # Tamil script
        "புண்டை", "புண்ட", "சுன்னி", "குத்து", "ஊம்பு", "ஊம்ப", "சப்பு",
        "விந்து", "முலை", "பூலு", "பூல்", "கையடி", "ஓத்தா", "ஓக்காளி",
        "ஓம்மால", "குஞ்சு", "பொச்சு",
    },
    "clothing_body": {
        "bra", "panty", "jatti",
    },
    "service_paid": {
        "service", "paid",
    },
    "scam": {
        "fake id",
    },
    "caste_hate": {
        "keel", "jaathi",
    },
}

# Words too generic to hard-block on their own (see conversation) —
# excluded from CATEGORIES by default. Flip HARD_BLOCK_PM_CATEGORY to
# True if you want them enforced identically to everything else anyway.
PM_CATEGORY = {"dm", "pm", "msg", "inbox", "ping", "prvt", "pvt", "add"}
HARD_BLOCK_PM_CATEGORY = False
if HARD_BLOCK_PM_CATEGORY:
    CATEGORIES["pm_solicitation"] = PM_CATEGORY

# "girls", "boys", "women", "aunty" are plain gender references, not
# violations by themselves (a message like "girls hostel curfew" or
# "aunty how are you" is completely normal). Left out on purpose so
# real conversation isn't blocked. Enable only if you actually want any
# use of these words auto-actioned.
# CATEGORIES["gender_reference"] = {"girls", "boys", "women", "aunty"}

EMOJIS = {"🍆", "🍑", "👙", "💋", "🖕", "😘"}

# --- Matching ------------------------------------------------------------

_word_pattern_cache = {}


def _compile(term):
    if term not in _word_pattern_cache:
        if re.search(r"[a-zA-Z]", term) and " " not in term:
            # single latin word -> match on word boundaries
            pat = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        else:
            # phrase or non-latin script -> plain substring, case-insensitive
            pat = re.compile(re.escape(term), re.IGNORECASE)
        _word_pattern_cache[term] = pat
    return _word_pattern_cache[term]


def check_text(text):
    """
    Returns a list of (category, matched_term) for every blocklist hit
    found in text. Empty list = clean message.
    """
    hits = []
    if not text:
        return hits

    for category, terms in CATEGORIES.items():
        for term in terms:
            if _compile(term).search(text):
                hits.append((category, term))

    for emoji in EMOJIS:
        if emoji in text:
            hits.append(("emoji", emoji))

    return hits


def is_violation(text):
    return len(check_text(text)) > 0


if __name__ == "__main__":
    tests = [
        "hey how are you",
        "any girls available for chat",
        "send nude pics",
        "sunni pannu da",
        "call me at 5pm sharp",
        "fake id venuma",
        "🍆🍑 dm me",
    ]
    for t in tests:
        hits = check_text(t)
        print(f"{t!r:35} -> {hits if hits else 'CLEAN'}")
