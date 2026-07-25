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
        "sex", "sexy", "porn", "pornography", "nude", "nudee", "boobs",
        "booby", "penis", "pussy", "dick", "ass", "buttocks", "fingering",
        "foreplay", "masturbation", "lick", "kiss", "fucking", "fucked",
        "fuck", "motherfucker", "incest", "bdsm", "gangbang", "rape",
        "cuckold", "cuck", "bisex", "dildo", "whore", "hoe", "blowjob",
        "bikini", "tits", "video call vanga",
    },
    "relationship_dating": {
        "hookup", "hookups", "fwb", "wife", "married", "darling", "baby",
        "milf", "bhabhi", "housewife", "available", "virgin",
        "couple exchange", "item irukingla", "personal ah pesalam",
        "real meet me",
    },
    "gender_orientation": {
        "lesbian", "lesbians", "gay", "gays", "bisexual", "transgender",
        "shemale", "trans", "aravaani", "vaanavil", "vanavil",
    },
    "tamil_abusive": {
        "thevidiya", "thevudiya", "tevadiya", "tevudiya", "thevdiya",
        "thevadiya",
        "thayoli", "gommala", "gommalaka", "gomma", "kandaroli", "mayiru",
        "mayirandi", "mairu", "mairru", "ommala", "oombu", "oombi",
        "ombuga", "oompi", "ottha", "otha", "olumaari", "olu maari",
        "koothi", "koodhi", "lavade", "punda", "punde", "bunda", "potta",
        "funda", "ungoppa", "ungoppan", "ungomma", "bastard", "fucker",
        "vibachari", "avusari", "avuthu", "munda", "baadu", "wathalaka",
        "okuren", "okanum", "ollu", "olu", "oluka", "oluthu",
        "ugomma",
    },
    "tamil_sexual_slang": {
        "sunni", "junni", "jumni", "kunju", "kundi", "kundiadi", "kuthi",
        "sappu", "umbu", "oolu", "oomba", "nakku", "nakkuren", "poolu",
        "kai adikuren",
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

# Skipped on purpose (see conversation): "item" and "matti" are common
# enough as ordinary words (an item on a list, a person's name, etc.)
# that hard-blocking them risks real false positives. "pool" is
# excluded for the same reason (swimming pool, car pool) -- "poolu"
# (the Tamil-specific spelling) is included above instead, since that
# form essentially never appears in innocent English text. Free-form
# solicitation phrases with no single flaggable word ("personal ah
# pesalam", "video call vanga", "real meet me") aren't coverable by a
# word blocklist at all -- see the note in extract_features/ANCHORS
# about why closing that gap needs a different approach.

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

EMOJIS = set()  # emoji blocklist removed per request -- no emoji is auto-flagged anymore

# --- Matching ------------------------------------------------------------

_phrase_pattern_cache = {}


def _phrase_pattern(term):
    if term not in _phrase_pattern_cache:
        _phrase_pattern_cache[term] = re.compile(re.escape(term), re.IGNORECASE)
    return _phrase_pattern_cache[term]


def collapse_runs(s):
    """'pmm' -> 'pm', 'inboxx' -> 'inbox'. Same idea as trainer.py's
    version, duplicated here on purpose to keep this module decoupled."""
    return re.sub(r"(.)\1+", r"\1", s)


def _strip_digits(tok):
    """'9punda' -> 'punda', 'punda9' -> 'punda'."""
    return re.sub(r"^\d+|\d+$", "", tok)


def _tokenize(text):
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def _normalized_token_variants(text):
    """
    Every token in the message, plus digit-stripped and letter-collapsed
    variants of each, so single-word terms can be matched against
    common evasions (digit-gluing, doubled letters) without needing a
    separate blocklist entry for every possible spelling.
    """
    variants = set()
    for tok in _tokenize(text):
        variants.add(tok)
        stripped = _strip_digits(tok)
        if stripped:
            variants.add(stripped)
        variants.add(collapse_runs(tok))
        if stripped:
            variants.add(collapse_runs(stripped))
    return variants


def check_text(text):
    """
    Returns a list of (category, matched_term) for every blocklist hit
    found in text. Empty list = clean message.

    Single-word latin terms are matched against normalized token
    variants (handles "9punda", "Pmm", "Inboxx"-style evasion).
    Phrases and non-latin script terms are matched as plain
    case-insensitive substrings, unchanged.
    """
    hits = []
    if not text:
        return hits

    token_variants = _normalized_token_variants(text)

    for category, terms in CATEGORIES.items():
        for term in terms:
            if re.search(r"[a-zA-Z]", term) and " " not in term:
                if term.lower() in token_variants:
                    hits.append((category, term))
            else:
                if _phrase_pattern(term).search(text):
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
