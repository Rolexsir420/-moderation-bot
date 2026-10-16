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


# "9pm", "730pm" (from "7:30pm"), "215pm" (from "2.15pm")
_TIME_GLUED_RE = re.compile(r"^\d{1,4}(am|pm)$")
_TIME_NUM_RE = re.compile(r"^\d{1,4}$")

# --- Word-sense rules ---------------------------------------------------
# Some anchors have a common NON-chat meaning. When the message shows that
# meaning, the anchor is skipped entirely (it emits no features at all),
# so the classifier never sees it as a DM word. Other anchors in the same
# message are still checked: "pvt company la work, dm pannunga" still
# scores on "dm".
#
# pvt/private: private company / school / bus / job ("pvt or govt").
# personal:    personal reason / loan / work.
# pm:          prime minister ("pm modi").
# Times ("9pm", "7:30 pm", "2.15 pm", "4 pm") are handled in
# extract_features: a pm/am that follows a number is a time, not a DM ask.
_SENSE_RULES = {
    "private": {
        "company", "companies", "ltd", "limited", "govt", "government", "gov",
        "job", "jobs", "school", "schools", "college", "bus", "hospital",
        "bank", "sector", "office", "firm", "concern", "salary", "tuition",
        "coaching", "clinic", "university", "institute", "vehicle", "car",
        "property", "mnc", "startup", "factory",
    },
    "personal": {
        "reason", "reasons", "loan", "work", "leave", "problem", "problems",
        "issue", "issues", "emergency", "matter", "use", "account",
    },
    "pm": {"modi", "minister", "prime", "govt", "government", "election"},
    # add: adding a THING, not a person to chat with — "ticket ah add
    # panitingala", "en name ah list la add pannunga", "team la add pannu",
    # "items la add pannikonga". Deliberately NOT here: number, contact,
    # whatsapp, insta, group, id — "add my number" IS a DM ask.
    "add": {
        "ticket", "tickets", "list", "name", "names", "team", "teams", "cart",
        "item", "items", "song", "songs", "playlist", "seat", "seats",
        "booking", "score", "runs", "points", "match", "amount", "money",
        "balance", "bill", "expense", "expenses", "salt", "sugar", "milk",
        "grocery", "groceries", "vegetables", "photo", "photos", "album",
        "sticker", "stickers", "emoji", "emojis", "movie", "show",
    },
}


# "me" in English/Tanglish; compared after letter-run collapse ("meee"->"me")
_SELF_WORDS = {"me", "ena", "enai", "enaya", "enaiyum", "enayum", "yenai", "yena"}

# "come / let's talk / do it" words. Right next to pvt/private/personal/pm
# they make it a DM ask ("pvt vaa", "private pesalam", "pm vaanga") even if
# the message, or someone's message just before it, mentions a company.
# Compared after letter-run collapse, so "vaaa"->"va", "pesalaam"->"pesalam".
_REQUEST_WORDS = {
    "va", "vanga", "vange", "vaga", "varia", "variya", "varingala", "varuvingala",
    "pesalam", "pesalama", "pesulam", "pesulama", "pesu", "pesunga",
    "chat", "come", "cm", "pannu", "panu", "pannunga", "panunga", "pannuga",
    "panuga", "pannuge", "pannunge", "msg", "dm", "pm", "ib", "text",
}

# which neighbours cancel each anchor's sense rule
_SENSE_GUARDS = {
    "private": _SELF_WORDS | _REQUEST_WORDS,
    "personal": _SELF_WORDS | _REQUEST_WORDS,
    "pm": _SELF_WORDS | _REQUEST_WORDS,
    # "ticket add pannunga" is a normal request, so only "me" guards add
    "add": _SELF_WORDS,
}


# --- Tanglish helper file (tanglish.json, next to this file) -------------
# Plain word lists the group admin can edit without touching code:
#   spellings   - many spellings -> one word ("panugaa" -> "pannunga")
#   tags        - meaning signals (REQUEST / PAST / NEG / TARGET / ASK_AROUND)
#   sense_rules, self_words, request_words - override the built-in lists
#                 above. If the file is missing or broken, the built-in
#                 lists are used and spellings/tags are simply off.
# All matching is done after collapse_runs, so stretched letters
# ("vaaangaaa") match their listed spelling automatically.
_CANON = {}       # collapsed spelling -> canonical word
_TAG_OF = {}      # collapsed word -> tag name
TANGLISH_LOADED = False


def _load_tanglish():
    global _SENSE_RULES, _SELF_WORDS, _REQUEST_WORDS, _SENSE_GUARDS, TANGLISH_LOADED
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tanglish.json")
    try:
        with open(path, encoding="utf-8") as f:
            lex = json.load(f)
    except (OSError, ValueError) as e:
        print(f"[WARN] tanglish.json not loaded ({type(e).__name__}: {e}); using built-in lists")
        return
    for canon, spellings in lex.get("spellings", {}).items():
        for s in spellings + [canon]:
            _CANON[collapse_runs(s.lower())] = canon.lower()
    for tag, words in lex.get("tags", {}).items():
        for w in words:
            c = collapse_runs(w.lower())
            _TAG_OF.setdefault(_CANON.get(c, c), tag)
            _TAG_OF.setdefault(c, tag)
    if "sense_rules" in lex:
        _SENSE_RULES = {k: {w.lower() for w in v} for k, v in lex["sense_rules"].items()}
    if "self_words" in lex:
        _SELF_WORDS = {collapse_runs(w.lower()) for w in lex["self_words"]}
    if "request_words" in lex:
        _REQUEST_WORDS = {collapse_runs(w.lower()) for w in lex["request_words"]}
    _SENSE_GUARDS = {
        "private": _SELF_WORDS | _REQUEST_WORDS,
        "personal": _SELF_WORDS | _REQUEST_WORDS,
        "pm": _SELF_WORDS | _REQUEST_WORDS,
        "add": _SELF_WORDS,
    }
    TANGLISH_LOADED = True


_load_tanglish()


def canonical(tok):
    """'panugaa' -> 'pannunga' via tanglish.json; anchors are never changed."""
    if tok in ANCHORS or tok in ANCHOR_ALIASES or tok in GLUED_COMPOUNDS:
        return tok
    return _CANON.get(collapse_runs(tok), tok)


def tag_of(tok):
    """Meaning tag for a (canonical) token, or None."""
    return _TAG_OF.get(tok) or _TAG_OF.get(collapse_runs(tok))


def is_direct_ask(text):
    """
    True if the message itself directly asks for a private chat: a
    pvt/private/personal/pm anchor right next to a come/talk/me word
    ("Pvt vaa", "pm vaanga", "private pesalam", "pvt enna"). bot.py uses
    this to refuse context rescues for such messages, so posting "Pvt vaa"
    right after someone mentions their pvt company can't slip through.
    ("add" is not included: "Add me" replying to a game invite is a
    legitimate rescue case.)
    """
    tokens, _ = merge_split_anchors(tokenize(text))
    for i, tok in enumerate(tokens):
        base = ANCHOR_ALIASES.get(tok, tok)
        if base not in ("private", "personal", "pm"):
            continue
        guards = _SENSE_GUARDS[base]
        for j in (i - 1, i + 1):
            if 0 <= j < len(tokens) and collapse_runs(tokens[j]) in guards:
                return True
    return False


def parent_explains(parent_text, text):
    """
    Conversation rule for REPLIES: True if the message being replied to
    gives the reply's anchor its ordinary meaning.
      "Game vaanga"        <- "Add me"        (add to the game)
      "list la add pannu"  <- "add pannitan"  (added to the list)
      "pvt or govt?"       <- "pvt la"        (private company)
    The parent's sense words count even if the reply says "me" — the
    reply is answering an invite. Direct asks ("Pvt vaa") never qualify.
    """
    if not parent_text or is_direct_ask(text):
        return False
    parent_tokens = set(tokenize(parent_text))
    for tok in merge_split_anchors(tokenize(text))[0]:
        base = ANCHOR_ALIASES.get(tok, tok)
        if base not in ANCHORS:
            c = collapse_runs(tok)
            base = c if c in ANCHORS else _COLLAPSED_ANCHORS.get(c)
        if base is None and tok in GLUED_COMPOUNDS:
            base = GLUED_COMPOUNDS[tok][0]
        sense = _SENSE_RULES.get(base)
        if sense and _has_sense(parent_tokens, sense):
            return True
    return False


def _has_sense(tokens, sense_words):
    """
    True if any token is a sense word, or (for sense words of 4+ letters)
    starts with one, so Tamil suffixes still count: "teamla", "listla",
    "ticketah", "companyla". Short words (bus, car, gov, ltd) are exact-
    match only, so "card"/"carry"/"business" don't trigger "car"/"bus".
    """
    for tok in tokens:
        if tok in sense_words:
            return True
        for w in sense_words:
            if len(w) >= 4 and tok.startswith(w):
                return True
    return False


def extract_features(text):
    """
    Returns list of (anchor, feature) pairs found in this text.
    feature is either a nearby context word or one of the evasion flags
    (BIAS, DIGIT_GLUED, TIME_SPACED, PUNCT_SPLIT_ANCHOR,
    LETTER_DOUBLED_ANCHOR, GLUED_EXTRA:<word>).
    """
    raw_tokens = tokenize(text)
    tokens, split_flags = merge_split_anchors(raw_tokens)
    # sense rules look at the words as typed; context features use the
    # tanglish.json canonical spelling so "panugaa" and "pannunga" are one
    # word to the model.
    token_set = set(tokens)
    tokens = [canonical(t) for t in tokens]
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
        elif tok in ("pm", "am") and i > 0 and _TIME_NUM_RE.match(tokens[i - 1]):
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

        # times are never a DM ask: "9pm", "7:30 pm", "2.15 pm", "4 pm bro"
        if base in ("pm", "am") and (
            is_time_glued
            or is_time_spaced
            or (i > 0 and _TIME_NUM_RE.match(tokens[i - 1]))
        ):
            continue
        # non-chat word sense present anywhere in the message -> skip
        # ...unless the anchor is aimed at the sender ("add me", "pvt enna")
        # or sits next to a come/talk word ("pvt vaa", "private pesalam"):
        # that's a DM ask whatever else the message says.
        sense_words = _SENSE_RULES.get(base)
        guards = _SENSE_GUARDS.get(base, _SELF_WORDS)
        aimed_at_me = glued_extra == "me" or any(
            collapse_runs(tokens[j]) in guards
            for j in (i - 1, i + 1) if 0 <= j < len(tokens)
        )
        if sense_words and not aimed_at_me and _has_sense(token_set, sense_words):
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
            # A word with a meaning tag in tanglish.json counts ONCE, as its
            # tag: the model learns "NEG near pm" once for every refusal
            # word instead of each word separately. (Counting both the
            # word and its tag double-counts the evidence, which in a
            # Naive Bayes model dilutes everything else — it made plain
            # "add me" stop scoring as a DM ask.)
            tag = tag_of(tokens[j])
            found.append((base, "TAG_" + tag if tag else tokens[j]))

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
