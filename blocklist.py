"""
Hardcoded, categorized blocklist. Checked first and deterministically,
before the learned classifier ever runs, because a learned model alone
can't reliably catch everything (and shouldn't have to, for words that are
never okay in this group regardless of context).

Deliberately EXCLUDED from this file: the pm/dm/msg/inbox "wants to talk
privately" category. Those words are too common and too context-dependent
for a hard blocklist, so that detection is left entirely to the learned
classifier in trainer.py / predict.py.

Also deliberately excluded: ambiguous English roots that have other common
meanings (ass, dick, gay) and Tamil words that are also ordinary/affectionate
terms (kunju/kunjukutty = "little one"), to avoid false positives.

check_text(text) -> (is_hit: bool, categories: set[str])
"""

import re
import unicodedata

# Checked FIRST, before any other category, and treated as maximum
# severity by bot.py: an immediate .delban (not .delwarn) plus a loud
# admin alert that also tells the human admin to report the account to
# Telegram directly and, depending on content, to local cybercrime
# authorities — bot action alone is not an adequate response to apparent
# sexual-abuse-of-a-minor solicitation, only a first line of defense.
SEVERE_EXPLOITATION_TERMS = {
    "rape", "raping", "raped", "rapist",
    "molest", "molesting", "molested", "molestation",
    "childporn", "child porn",
    # "cp" on its own is NOT here: as a bare token it's ordinary chat
    # (copy-paste, cricket "cp score", abbreviations) and this category is
    # an instant .delban + urgent alert. Only explicit CSAM-trading phrasing
    # around "cp" is listed, matched as whole words.
    "cp video", "cp videos", "cp link", "cp links", "cp vid", "cp vids",
    "cp collection", "cp sell", "cp selling", "cp available",
}

CATEGORIES = {
    "severe_exploitation": set(SEVERE_EXPLOITATION_TERMS),
    "sexual": {
        "sex", "sexy", "porn", "pornography", "nude", "nudee", "boobs",
        "booby", "penis", "pussy", "dick", "ass", "buttocks", "fingering",
        "foreplay", "masturbation", "lick", "kiss", "fucking", "fucked",
        "fuck", "motherfucker", "incest", "bdsm", "gangbang",
        "cuckold", "cuck", "bisex", "dildo", "whore", "hoe", "blowjob",
        "bikini", "tits", "video call vanga",
    },
    "relationship_dating": {
        "hookup", "hookups", "fwb", "wife", "married", "darling", "baby",
        "milf", "bhabhi", "housewife", "available", "virgin",
        "couple exchange", "item irukingla", "personal ah pesalam",
        "real meet me", "secret ah pesalam", "secret chat",
        "outside pesalam",
        # swinger / couple-swap personals-ad slang ("cpls" = couples,
        # "swp" = swap, "single f/m" = single female/male looking to join
        # a couple) — none of this overlaps ordinary friend-group chat,
        # so plain single-word/phrase matching is safe here.
        # "cpl" stays: real ads use the singular ("Real cpl here",
        # "CPL to CPL", "c-p-l"). The innocent "cpl of days/hours" (= a
        # couple of) is stripped out first by _SAFE_PHRASES_RE below.
        "cpls", "cpl", "swinger", "swingers", "threesome", "3some",
        "swp cpls", "swap cpls", "couple swap", "cpl swap", "bi cpl",
        "single f", "single m", "mfm", "fmf",
    },
    "gender_orientation": {
        "gayboy", "les", "lesbian", "shemale", "trans",
    },
    "tamil_abusive": {
        "punda", "pundai", "koothi", "koodhi", "oombu", "oombi", "mayiru",
        "thevidiya", "thevudiya", "tevadiya", "munda", "baadu",
        "wathalaka", "okuren", "okanum", "ollu", "olu", "oluka", "oluthu",
        "ugomma", "vibachari", "avusari", "avuthu",
    },
    "tamil_sexual_slang": {
        # "kunju" removed: it's an everyday affectionate word ("little
        # one"), per this file's own exclusion note at the top.
        "sunni", "sunniya", "junni", "jumni", "poolu", "nakku",
        "சுன்னி", "கூத்தி", "பூண்டை",
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

# Tamil slang roots with many suffix variants (sunniya, punduka, koothiya,
# etc.) — matched as a prefix rather than requiring an exact-word hit.
# Deliberately NOT used for ambiguous English roots or words with other
# legitimate meanings.
PREFIX_ROOTS = {
    "tamil_abusive": {
        "punda", "pundai", "koothi", "koodhi", "oombu", "oombi", "mayiru",
        "thevidiya", "thevudiya", "tevadiya", "vibachari",
        # "munda" is NOT a prefix root: it prefixes "mundanaal" (day
        # before yesterday). The exact word "munda" is still caught via
        # CATEGORIES above.
    },
    "tamil_sexual_slang": {
        "sunni", "poolu", "nakku",
    },
}

# Emptied per explicit request — no emoji-based blocking.
EMOJIS = set()

_WORD_RE = re.compile(r"[a-zA-Z0-9]+|[஀-௿]+")

# Zero-width/formatting characters sometimes stuffed between letters to
# evade plain substring matching (ன​ொ​ன்​னி-style joiners, word joiners,
# bidi marks). Stripped out before anything else.
_ZERO_WIDTH_RE = re.compile(
    "[​-‏‪-‮﻿⁠-⁤­]"
)
# Combining marks used purely as decoration on top of a base Latin letter
# (not real diacritics in this group's languages), e.g. s̶t̶r̶i̶k̶e̶.
_COMBINING_RE = re.compile(r"[̀-ͯ]")


def _normalize_unicode(text):
    """
    Folds "fancy font" unicode (fullwidth, mathematical bold/italic,
    circled/squared letters, etc.) and regional-indicator "flag" letters
    back down to plain ASCII, and strips zero-width/combining characters
    used to break up a word for evasion. This runs before tokenizing, so
    stylized or unicode-obfuscated text is matched the same as plain text.
    """
    if not text:
        return text

    # Regional indicator symbols (🇦-🇿) aren't touched by NFKC, so map
    # them back to A-Z explicitly first.
    out = []
    for ch in text:
        cp = ord(ch)
        if 0x1F1E6 <= cp <= 0x1F1FF:
            out.append(chr(ord("a") + (cp - 0x1F1E6)))
        else:
            out.append(ch)
    text = "".join(out)

    # NFKC folds most other decorative/lookalike unicode letter blocks
    # (fullwidth "ｈｅｌｌｏ", mathematical "𝐡𝐞𝐥𝐥𝐨"/"𝙷𝙴𝙻𝙻𝙾", circled
    # "Ⓗⓔⓛⓛⓔ", etc.) down to their plain ASCII equivalents.
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _COMBINING_RE.sub("", text)
    return text


def _strip_digits(tok):
    return re.sub(r"\d+", "", tok)


def _collapse_runs(tok):
    # aaa -> a, dddd -> d  (letter-repetition evasion)
    return re.sub(r"(.)\1+", r"\1", tok)


def _token_variants(tok):
    variants = {tok}
    variants.add(_strip_digits(tok))
    variants.add(_collapse_runs(tok))
    variants.add(_collapse_runs(_strip_digits(tok)))
    return {v for v in variants if v}


def _merge_split_words(tokens, words):
    """
    Catches a blocklist word typed with punctuation/spaces/symbols between
    every letter ("r.a.p.e", "r a p e", "r&a&p&e") — the tokenizer already
    splits those into single-character tokens, since only alphanumerics
    are kept and everything else is a separator. This re-joins runs of
    single-character tokens and checks them against each word.
    Returns the set of words found this way.
    """
    found = set()
    i, n = 0, len(tokens)
    while i < n:
        matched = False
        for word in words:
            L = len(word)
            if L < 3 or i + L > n:
                continue
            candidate = tokens[i:i + L]
            if all(len(t) == 1 for t in candidate) and "".join(candidate) == word:
                found.add(word)
                i += L
                matched = True
                break
        if not matched:
            i += 1
    return found


def _phrase_hit(term, lowered):
    """
    Multi-word Latin phrase match.

    Most phrases keep plain substring matching (so Tamil suffixes still
    match, e.g. "video call vanga" inside "video call vangala"). But a
    phrase whose LAST word is a single character ("single f", "single m")
    or which starts with a 1-2 letter abbreviation ("cp video") is
    matched on whole-word boundaries instead — otherwise "single m"
    matches "single match"/"single movie", and "cp video" would match
    inside unrelated words. "single f/" and "single f," still hit because
    "/" and "," aren't letters or digits.
    """
    words = term.split()
    if len(words[-1]) == 1 or len(words[0]) <= 2:
        return re.search(
            r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", lowered
        ) is not None
    return term in lowered


# Innocent phrases removed before matching, so a blocklisted word inside
# them doesn't hit: "oru cpl of days" = "a couple of days".
_SAFE_PHRASES_RE = re.compile(r"(?<![a-z0-9])cpl\s+of(?![a-z0-9])")


def check_text(text):
    """
    Returns (is_hit, categories) where categories is a set of category
    names that matched. is_hit is True if any category matched.
    """
    if not text:
        return False, set()

    text = _normalize_unicode(text)
    lowered = _SAFE_PHRASES_RE.sub(" ", text.lower())
    tokens = _WORD_RE.findall(lowered)
    hit_categories = set()

    # 1) phrase / substring matching (multi-word terms, non-Latin script,
    #    or single words where substring containment is intentional)
    for category, terms in CATEGORIES.items():
        single_word_terms = set()
        for term in terms:
            if any(ord(c) > 0x2000 for c in term):
                # non-Latin script (Tamil): plain substring, unchanged
                if term in lowered:
                    hit_categories.add(category)
            elif " " in term:
                if _phrase_hit(term, lowered):
                    hit_categories.add(category)
            else:
                single_word_terms.add(term)
                # single-word term: check exact token + common evasion
                # variants rather than raw substring (avoids matching
                # "ass" inside "class", "glass", etc.)
                for tok in tokens:
                    if term in _token_variants(tok):
                        hit_categories.add(category)
                        break

        # 1b) letter-by-letter obfuscation of a single-word term
        # ("r.a.p.e", "r a p e") — only meaningful for 3+ letter words.
        if single_word_terms and category not in hit_categories:
            if _merge_split_words(tokens, single_word_terms):
                hit_categories.add(category)

    # 2) prefix-root matching for Tamil slang with heavy suffixing
    for category, roots in PREFIX_ROOTS.items():
        for tok in tokens:
            for root in roots:
                if tok.startswith(root) and len(tok) >= len(root):
                    hit_categories.add(category)
                    break

    return (len(hit_categories) > 0), hit_categories
