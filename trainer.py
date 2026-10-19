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
    "pv": "private",
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
_PUNCT_SPLIT_RE = re.compile(r"[^a-z0-9']+")


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
    #
    # Punctuation INSIDE a chunk: if it splits off single letters ("d.m",
    # "..a...ddd", "7:30") it's evasion or a number, so the pieces are
    # JOINED; if every piece is a real word ("dm,pannunga",
    # "electrician,dm", "girls.dm.me") it separates words, so the pieces
    # are KEPT SEPARATE.
    text = normalize_unicode(text)
    text = text.lower()
    tokens = []
    for chunk in _WS_SPLIT_RE.split(text.strip()):
        pieces = [p for p in _PUNCT_SPLIT_RE.split(chunk) if p]
        if not pieces:
            continue
        if len(pieces) == 1 or any(len(p) == 1 for p in pieces):
            tokens.append("".join(pieces))
        else:
            tokens.extend(pieces)
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
_CANT_WORDS = set()     # "can't / didn't see" words (canonical)
_NEG_BEFORE = set()     # "don't", "no", "never" right before a DM word
_NEG_AFTER = set()      # "pannathinga", "vendam" right after a DM word
_REPORTED_AFTER = set() # "pannuchu", "panraanga": someone else did it
_GROUP_WORDS = set()    # "in the group / here / in vc" (collapsed)
_TOPIC_WORDS = set()    # words that make a conversation about messaging
_PRIVATE_WORDS = set()  # any of these cancels the group-location rule
# anchors that mean "send a message" in general, so a group location next
# to them makes them innocent; dm/pm/ib/inbox/private are private by nature
_GROUP_RULE_ANCHORS = {"msg", "message", "text", "txt", "mes"}
_DETERMINERS = {collapse_runs(w) for w in ("antha", "andha", "indha", "intha", "inda", "anda",
                                             "that", "this", "those", "these", "ithu", "athu")}
_LOC_FILLERS = {"la", "le", "lu", "ku", "il", "ile", "in", "ah", "a", "e", "ey", "ye", "le"}
_REQUEST_VERBS = set()  # "do it / come" verbs that keep a DM ask a DM ask
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
    for w in lex.get("cant_words", []):
        c = collapse_runs(w.lower())
        _CANT_WORDS.add(_CANON.get(c, c))
    for w in lex.get("request_verbs", []):
        c = collapse_runs(w.lower())
        _REQUEST_VERBS.add(_CANON.get(c, c))
    for w in lex.get("neg_before", []):
        _NEG_BEFORE.add(collapse_runs(w.lower()))
    for w in lex.get("neg_after", []):
        c = collapse_runs(w.lower())
        _NEG_AFTER.add(_CANON.get(c, c))
    for w in lex.get("reported_after", []):
        c = collapse_runs(w.lower())
        _REPORTED_AFTER.add(_CANON.get(c, c))
    for w in lex.get("topic_words", []):
        _TOPIC_WORDS.add(collapse_runs(w.lower()))
    for w in lex.get("group_words", []):
        _GROUP_WORDS.add(collapse_runs(w.lower()))
    for w in lex.get("private_words", []):
        _PRIVATE_WORDS.add(collapse_runs(w.lower()))
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
    tokens, _ = split_glued(*merge_split_anchors(tokenize(text)))
    for i, tok in enumerate(tokens):
        base = ANCHOR_ALIASES.get(tok) or ANCHOR_ALIASES.get(collapse_runs(tok), tok)
        if base not in ("private", "personal", "pm"):
            continue
        guards = _SENSE_GUARDS[base]
        for j in (i - 1, i + 1):
            if 0 <= j < len(tokens) and collapse_runs(tokens[j]) in guards:
                return True
    return False


# "to me" words: "text pannu enaku" asks to be texted personally
_TO_ME = {collapse_runs(w) for w in ("enaku", "enakku", "enku", "ennaku", "yenaku",
                                     "yenakku", "enakum", "enakkum", "naaku", "ennai")}


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
    reply_tokens = split_glued(*merge_split_anchors(tokenize(text)))[0]
    # parent is about the group / group VC ("grp vc va", "group la vaa"),
    # reply is about texting/messaging ("illa da veetila aal irukanga text
    # panure"): answering a group plan. Not when the reply is aimed at "me"
    # ("text me") or uses any private word ("pvt la text pannu").
    parent_group = any(collapse_runs(t) in _GROUP_WORDS
                       or any(collapse_runs(t).startswith(g) and len(g) >= 3 for g in _GROUP_WORDS)
                       for t in parent_tokens)
    if parent_group and _GROUP_WORDS:
        has_private = any(collapse_runs(t) in _PRIVATE_WORDS
                          or ANCHOR_ALIASES.get(collapse_runs(t)) == "private"
                          or collapse_runs(t) in _TO_ME
                          for t in reply_tokens)
        for i, tok in enumerate(reply_tokens):
            if _anchor_base(tok) in _GROUP_RULE_ANCHORS:
                aimed = any(collapse_runs(reply_tokens[j]) in _SELF_WORDS
                            for j in (i - 1, i + 1) if 0 <= j < len(reply_tokens))
                if not aimed and not has_private:
                    return True
    for tok in reply_tokens:
        base = ANCHOR_ALIASES.get(tok) or ANCHOR_ALIASES.get(collapse_runs(tok), tok)
        if base not in ANCHORS:
            c = collapse_runs(tok)
            base = c if c in ANCHORS else _COLLAPSED_ANCHORS.get(c)
        if base is None and tok in GLUED_COMPOUNDS:
            base = GLUED_COMPOUNDS[tok][0]
        sense = _SENSE_RULES.get(base)
        if sense and base == "add":
            # "group" in the PARENT is too generic ("Anyone studying for group
            # exams" <- "Addd panuga"); adding-to-a-group counts only when the
            # reply itself says it
            sense = sense - {"group", "grp", "groups", "gc", "channel"}
        if sense and _has_sense(parent_tokens, sense):
            return True
    return False


def _anchor_base(tok):
    """The anchor a single token stands for, or None (no time handling)."""
    if tok in ANCHORS:
        return tok
    c = collapse_runs(tok)
    if tok in ANCHOR_ALIASES or c in ANCHOR_ALIASES:
        return ANCHOR_ALIASES.get(tok) or ANCHOR_ALIASES[c]
    if c in ANCHORS:
        return c
    if c in _COLLAPSED_ANCHORS:
        return _COLLAPSED_ANCHORS[c]
    if tok in GLUED_COMPOUNDS:
        return None  # "msgme", "dmme": aimed at me, never a complaint
    return None


def is_cant_complaint(text):
    """
    "msg panna mudiyala", "en txt pakkala", "un msg padika mudila": a DM
    word next to a can't/didn't word (tanglish.json cant_words), with no
    request verb in the message and not aimed at "me". Whether such a
    message is innocent depends on the conversation around it (see
    context_is_about_messages), so this only says "it's that kind".
    """
    if not _CANT_WORDS:
        return False
    tokens = [canonical(t) for t in split_glued(*merge_split_anchors(tokenize(text)))[0]]
    if any(collapse_runs(t) in _REQUEST_VERBS or t in _REQUEST_VERBS for t in tokens):
        return False
    for i, tok in enumerate(tokens):
        if _anchor_base(tok) is None:
            continue
        if any(collapse_runs(tokens[j]) in _SELF_WORDS
               for j in (i - 1, i + 1) if 0 <= j < len(tokens)):
            return False
        lo, hi = max(0, i - WINDOW), min(len(tokens), i + WINDOW + 1)
        if any(collapse_runs(tokens[j]) in _CANT_WORDS or tokens[j] in _CANT_WORDS
               for j in range(lo, hi) if j != i):
            return True
    return False


_POSSESSIVE = {"un", "unga", "ungal", "ungaloda", "unoda", "unnoda", "your", "ur", "yours",
               "avan", "avanoda", "ava", "avaloda", "avanga", "avangaloda", "his", "her",
               "their", "ivan", "iva", "ivanga"}


def mentions_someones_message(text):
    """'un msg padika mudila', 'unga text varala': about someone's own
    message, so it belongs to the conversation even without topic words."""
    tokens = [canonical(t) for t in split_glued(*merge_split_anchors(tokenize(text)))[0]]
    for i, tok in enumerate(tokens):
        if _anchor_base(tok) and i > 0 and collapse_runs(tokens[i - 1]) in {collapse_runs(w) for w in _POSSESSIVE}:
            return True
    return False


def is_refusal(text):
    """
    (Also covers REPORTS of what someone else did: "antha ponnu enaku dm
    pannuchu", "private msg panraanga" via tanglish.json reported_after.)
    "Don't pm me", "No dm please", "yaarum dm pannathinga", "pm vendam",
    "DM PM not allowed", "Dm invite pana kudadhu", "text panatha": a DM
    word right after a neg_before word, or followed within 3 words by a
    neg_after word (tanglish.json), with no request verb in the message.
    Used by bot.py as a decision rule; it does NOT change training
    features, so the model keeps learning from refusal examples.
    """
    if not (_NEG_BEFORE or _NEG_AFTER):
        return False
    tokens = [canonical(t) for t in split_glued(*merge_split_anchors(tokenize(text)))[0]]
    if any((t in _REQUEST_VERBS or tag_of(t) == "REQUEST") and t not in _NEG_AFTER
           and t not in ("please", "pls", "plz") for t in tokens):
        return False
    for i, tok in enumerate(tokens):
        if _anchor_base(tok) is None:
            continue
        if i > 0 and collapse_runs(tokens[i - 1]) in _NEG_BEFORE:
            return True
        if i + 1 < len(tokens) and collapse_runs(tokens[i + 1]) in _SELF_WORDS:
            continue  # "dm me, not interested in others" is still an ask
        if any(tokens[j] in _NEG_AFTER or collapse_runs(tokens[j]) in _NEG_AFTER
               or tokens[j] in _REPORTED_AFTER or collapse_runs(tokens[j]) in _REPORTED_AFTER
               for j in range(i + 1, min(len(tokens), i + 4))):
            return True
    return False


_STRONG_ANCHORS = {"dm", "pm", "ib", "inbox", "private"}
# a 10-digit Indian mobile number, also partly masked ("98xxxxxx10")
_PHONE_RE = re.compile(r"(?:91)?[6-9][0-9x]{9}")
_CONTACT_WORDS = {"call", "cal", "whatsapp", "whatsap", "watsapp", "watsap", "wa", "contact", "ping", "msg", "text", "dm"}


def is_strong_ask(text):
    """
    "dm pannunga", "pm me", "pvt vaa", "inbox wagga", "ib pannunga": a DM
    word that is ALWAYS private (dm/pm/ib/inbox/pvt) directly followed by
    a request verb or "me". Not for times ("9 pm vaa"), not when the
    message talks to/about admin ("emergency admin pm pannunga"), not for
    refusals, and not when the pvt/pm sense rule applies (pvt company,
    pm modi). bot.py mutes these even when other words in the sentence
    would pull the learned score down.
    """
    if is_refusal(text):
        return False
    tokens = [canonical(t) for t in split_glued(*merge_split_anchors(tokenize(text)))[0]]
    if any(collapse_runs(t).startswith("admin") for t in tokens):
        return False
    # sharing a phone number to be called/messaged: "en number 98xxxxxx10 call pannunga"
    raw = tokenize(text)
    joined = []  # "98765 43210" -> one token
    for t in raw:
        if joined and t.isdigit() and joined[-1].isdigit():
            joined[-1] += t
        else:
            joined.append(t)
    if any(_PHONE_RE.fullmatch(t) for t in joined) and any(
            t in _CONTACT_WORDS or collapse_runs(t) in _CONTACT_WORDS or _anchor_base(t) for t in tokens):
        return True
    for i, tok in enumerate(tokens):
        base = _anchor_base(tok)
        # "text me", "txt me outside", "msg me": asking to be messaged
        if base in ("text", "txt", "msg", "message", "mes") and i + 1 < len(tokens):
            nxt = collapse_runs(tokens[i + 1])
            if nxt in ("me", "mee"):
                # "text me in group" / "msg me grp la" = in the group: fine,
                # unless a private word says otherwise ("txt me outside of grp")
                after = [collapse_runs(t) for t in tokens[i + 2:i + 5]]
                in_group = any(a in _GROUP_WORDS for a in after)
                private = any(collapse_runs(t) in _PRIVATE_WORDS for t in tokens)
                if in_group and not private:
                    continue
                return True
            # "can i text u", "shall i msg you" (but not "text u back")
            if nxt in ("u", "you", "yu") and i >= 2 and collapse_runs(tokens[i - 1]) == "i" \
                    and collapse_runs(tokens[i - 2]) in ("can", "shall", "may", "should") \
                    and not (i + 2 < len(tokens) and collapse_runs(tokens[i + 2]) in ("back", "later")):
                return True
    token_set = set(tokens)
    for i, tok in enumerate(tokens):
        base = _anchor_base(tok)
        if base not in _STRONG_ANCHORS or i + 1 >= len(tokens):
            continue
        if base in ("pm",) and i > 0 and (_TIME_NUM_RE.match(tokens[i - 1]) or _TIME_GLUED_RE.match(tok)):
            continue
        sense = _SENSE_RULES.get(base)
        nxt = tokens[i + 1]
        if (nxt in _REQUEST_VERBS or tag_of(nxt) == "REQUEST" or collapse_runs(nxt) in _SELF_WORDS):
            if sense and _has_sense(token_set, sense) and collapse_runs(nxt) not in _REQUEST_WORDS:
                continue
            return True
    return False


_ASKING_AROUND_RE = None


def is_asking_around(text):
    """
    "Any girls here", "Any akka irukingala", "Any ladies for friendship",
    "Single girl yaravathu irukigala": an ask word, a target word and a
    here/for-friendship word close together (tanglish.json asking_around).
    bot.py gives these .delwarn when they are not already a DM ask.
    """
    global _ASKING_AROUND_RE
    if _ASKING_AROUND_RE is None:
        import os
        try:
            lex = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "tanglish.json"), encoding="utf-8"))["asking_around"]
        except Exception:
            _ASKING_AROUND_RE = False
            return False
        alt = lambda ws: "(" + "|".join(re.escape(w) for w in sorted(ws, key=len, reverse=True)) + ")"
        a, t, h = alt(lex["ask"]), alt(lex["target"]), alt(lex["here"])
        # reversed order ("girl yaravathu irukigala") only with Tamil ask
        # words: "akka any doubt iruka" is an ordinary question
        a2 = alt([w for w in lex["ask"] if w not in ("any", "anyone", "any1", "single")])
        _ASKING_AROUND_RE = re.compile(
            rf"\b{a}\b.{{0,40}}\b{t}\b.{{0,40}}\b{h}\b|\b{t}\b.{{0,20}}\b{a2}\b.{{0,20}}\b{h}\b")
    if not _ASKING_AROUND_RE:
        return False
    return bool(_ASKING_AROUND_RE.search(normalize_unicode(text).lower()))


def context_is_about_messages(context_texts, include_dm_words=True):
    """
    True if the conversation around a message is about messaging: the
    replied-to message or recent messages from other people mention a
    message, a reply, being missing/busy/online, etc. (tanglish.json
    topic_words, plus any DM-type word). "enna pa inaiku un msg ye kaanom"
    -> "Msg panna mudiyala" is part of that conversation.
    """
    for t in context_texts:
        if not t:
            continue
        for tok in merge_split_anchors(tokenize(t))[0]:
            c = collapse_runs(tok)
            if (include_dm_words and _anchor_base(tok)) or c in _TOPIC_WORDS or canonical(tok) in _TOPIC_WORDS:
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


def split_glued(tokens, flags):
    """
    "addpannunga" -> "add" "pannunga", "dmpannunga" -> "dm" "pannunga",
    "dmmmpannugaa" -> "dm" "pannuga", "msgpannu" -> "msg" "pannu".
    Only when the rest is a request verb or "me"-word, so ordinary words
    that merely start with an anchor ("address", "dmk", "pmk", "adding",
    "message") are left alone.
    """
    out, out_flags = [], []
    for tok, fl in zip(tokens, flags):
        done = False
        if tok not in ANCHORS and tok not in GLUED_COMPOUNDS and len(tok) >= 5:
            c = collapse_runs(tok)
            for a in sorted(set(ANCHORS) | set(_COLLAPSED_ANCHORS), key=len, reverse=True):
                if len(a) < 2 or not c.startswith(a):
                    continue
                rest = c[len(a):]
                base = a if a in ANCHORS else _COLLAPSED_ANCHORS[a]
                rest_canon = canonical(rest)
                if (rest_canon in _REQUEST_VERBS or tag_of(rest_canon) == "REQUEST"
                        or rest in _SELF_WORDS or rest_canon in _SELF_WORDS):
                    out += [base, rest_canon]
                    out_flags += [fl, False]
                    done = True
                break
        if not done:
            out.append(tok)
            out_flags.append(fl)
    return out, out_flags


def extract_features(text):
    """
    Returns list of (anchor, feature) pairs found in this text.
    feature is either a nearby context word or one of the evasion flags
    (BIAS, DIGIT_GLUED, TIME_SPACED, PUNCT_SPLIT_ANCHOR,
    LETTER_DOUBLED_ANCHOR, GLUED_EXTRA:<word>).
    """
    raw_tokens = tokenize(text)
    tokens, split_flags = merge_split_anchors(raw_tokens)
    tokens, split_flags = split_glued(tokens, split_flags)
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
        elif collapse_runs(tok) in ANCHOR_ALIASES:
            # stretched alias: "pvtt", "pvttt", "prvvt" -> private
            base = ANCHOR_ALIASES[collapse_runs(tok)]
            is_letter_doubled = True
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
        # (can't/didn't complaints like "msg panna mudiyala" are NOT skipped
        # here: whether they're innocent depends on the conversation, so
        # bot.py decides with is_cant_complaint() + context_is_about_messages())
        # a specific message: "antha msg", "indha text", "that message"
        if (base in _GROUP_RULE_ANCHORS and i > 0 and collapse_runs(tokens[i - 1]) in _DETERMINERS
                and not (i + 1 < len(tokens) and (tokens[i + 1] in _REQUEST_VERBS
                                                  or tag_of(tokens[i + 1]) == "REQUEST"))):
            continue
        # sent IN THE GROUP: "group la msg pannuga", "inga msg pannu",
        # "vc la msg on panungaa", "text here" — allowed. Needs the group
        # word right beside the anchor (fillers like "la" skipped), and no
        # private word anywhere ("grp la illa pvt la msg pannu" still counts).
        if base in _GROUP_RULE_ANCHORS and _GROUP_WORDS and not aimed_at_me:
            def _loc_near(step, max_between):
                j, seen_words = i + step, 0
                while 0 <= j < len(tokens) and seen_words <= max_between:
                    c = collapse_runs(tokens[j])
                    if c in _GROUP_WORDS or any(c.startswith(g) and len(g) >= 3 for g in _GROUP_WORDS):
                        return True
                    if c not in _LOC_FILLERS:
                        seen_words += 1
                    j += step
                return False
            has_private = any(collapse_runs(t) in _PRIVATE_WORDS
                              or ANCHOR_ALIASES.get(collapse_runs(t)) == "private"
                              for t in tokens)
            # "Salem la yaradhu irukingala inga irundha message pannunga":
            # asking around or naming a target means "inga" is a place, not
            # the group, so the rule stays off
            asks_around = any(tag_of(t) in ("ASK_AROUND", "TARGET") for t in tokens)
            if (_loc_near(-1, 1) or _loc_near(1, 0)) and not has_private and not asks_around:
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
