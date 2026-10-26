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
        # "wife" is NOT here on its own ("unnga wife thambi" = your wife's
        # brother, "en wife samayal"): only in swap/share/contact ads, see
        # is_dating_pair
        # "darling" is NOT here on its own: "Vijay darling" fan talk is chat.
        # Pet names warn when said to someone flirting (is_pet_flirt below).
        "hookup", "hookups", "fwb",
        # "married" only in personals-ad phrases ("married woman", "married
        # guy here"): "Recently married and moved here" is normal chat.
        "married woman", "married women", "married guy", "married man",
        "married men", "married here", "married aunty", "married girl",
        "married lady", "married ladies", "married couple", "married cpl",
        "m married", "f married",
        # Tamil / other target words ("married ponnuga irukingla" = asking
        # around for married women); "ponnu" also matches ponnuga/ponnunga
        "married ponnu", "married ponu", "married ponnunga", "married akka",
        "married female", "married females", "married pen", "married penn",
        "married aunties", "married wife", "married wifes",
        # "baby" only in flirting phrases: "baby ku kal vali" is a real baby.
        "hi baby", "hii baby", "hey baby", "baby vc", "baby vaa", "baby va",
        "my baby", "baby dm", "baby pm", "baby msg", "come baby", "sweet baby",
        # "housewife", "available", "bhabhi", "virgin" are NOT here on their
        # own: "en amma housewife", "Is he available?", "Doctor available
        # tomorrow", "virgin mobile" are ordinary chat. They warn only next to
        # a dating/contact word (is_dating_pair below).
        "milf",
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
        # group policy: plain "gay" is also warned (Group Help banned it too)
        # "gays" alone is NOT here: "Hi gays" / "What gays" is a typo for
        # "guys". It warns only next to an 18+ word (is_context_hit below).
        "gay",
        # restored: removed by mistake in the 04-Oct update
        "aravaani", "bisexual", "lesbians", "transgender",
    },
    "tamil_abusive": {
        "punda", "pundai", "koothi", "koodhi", "oombu", "oombi", "mayiru",
        "thevidiya", "thevudiya", "tevadiya", "munda", "baadu",
        # more spellings seen in the group
        "thevadiya", "thevdiya", "thevudya", "thevidya", "tevdiya", "thevadia",
        "thevidia", "thevudia", "dhevidiya", "dhevadiya",
        # f-spelling of punda/pundai (exact words only: "fundamental" is safe)
        "funda", "fundai", "fundae", "fundaa",
        # restored: removed by mistake in the 04-Oct update
        "bastard", "bunda", "fucker", "gomma", "gommala", "gommalaka", "kandaroli", "lavade", "mairru", "mairu", "mayirandi", "olu maari", "olumaari", "ombuga", "ommala", "oompi", "punde", "tevudiya", "thayoli", "ungomma", "ungoppa", "ungoppan",
        "wathalaka", "okuren", "okanum", "ollu", "olu", "oluka", "oluthu",
        "ugomma", "vibachari", "avusari",
        # "avuthu" removed: also means "it'll happen" ("job avuthu")
    },
    "tamil_sexual_slang": {
        # "kunju" removed: it's an everyday affectionate word ("little
        # one"), per this file's own exclusion note at the top.
        "sunni", "sunniya", "junni", "jumni", "poolu", "nakku",
        # restored: removed by mistake in the 04-Oct update
        "kai adikuren", "kundi", "kundiadi", "kuthi", "nakkuren", "oolu", "oomba", "sappu", "umbu", "ஊம்ப", "ஊம்பு", "ஓக்காளி", "ஓத்தா", "ஓம்மால", "கையடி", "சப்பு", "புண்ட", "புண்டை", "பூலு", "பூல்", "பொச்சு", "முலை", "விந்து",
        "சுன்னி", "கூத்தி", "பூண்டை",
    },
    "clothing_body": {
        "bra", "panty", "jatti", "jatty", "jaddi", "jattiya",
        # "jetty" = jatti (inner wear) here, but also a pier ("Marina jetty"):
        # only in body phrases
        "jetty smell", "jetty potu", "jetty kalate", "jetty kalati", "akka jetty",
        "aunty jetty", "un jetty", "unga jetty", "jetty pic", "jetty photo",
        "jetty kaatu", "jetty color", "jetty colour", "jettiya",
    },
    # "service" / "paid" alone are ordinary ("bike service", "paid review",
    # "bill paid"); they warn only in a paid-service ad (is_context_hit)
    "service_paid": set(),
    "scam": set(),
    # group rule 3: never call a member "Fake ID" -> .delmute 3h (bot.py)
    "fake_id_accusation": {
        "fake id", "fake ids", "fakeid", "fakeids", "fake idu", "fake ida",
        "fake account", "fake acc", "fake accnt", "fake profile",
        "ஃபேக் ஐடி", "பேக் ஐடி", "fake ஐடி",
    },
    # swapping (18+) groups -> .delmute 3h (bot.py)
    "group_exchange": {
        "group exchange", "grp exchange", "groups exchange", "grps exchange",
        "exchange group", "exchange grp", "exchange groups", "group xchange",
        "grp xchange",
    },
    # caste words alone are ordinary ("jaathi veri vendam" = no casteism);
    # they warn only with an insult / low-high word (is_context_hit)
    "caste_hate": set(),
}


# --- Learned 21-Oct-2026 ---------------------------------------------------
# English: from two public profanity lists (github.com/dsojevic/profanity-list
# severity 3-4, github.com/coffee-and-fun/google-profanity-words). Every word
# was checked against 84k real words (Tamil dataset comments marked "not
# offensive", this group's history, the Dakshina Tamil word list) through the
# same matcher check_text uses; words that hit real chat were left out
# ("damn", "hell", "anal" = அனல், "kum" = -kum ending, "cok" ~ "cook",
# "paki" ~ "pakki", "semen" ~ "semmeen", "pawn", "jerk", "knob", "xxx" ...).
# Tamil/Tanglish: words the Tamil offensive dataset (DravidianCodeMix) shows
# almost only in offensive comments. Mild banter ("loosu", "mokka", "naai",
# "komali", "potta") is NOT added: in a friend group that is chat.
_LEARNED = {
    "sexual": {
        "1 man 1 jar", "1m1j", "1man1jar", "2 girls 1 cup", "2g1c", "2girls1cup", "4r5e",
        "acrotomophile", "acrotomophilia", "alabama hot pocket", "alabama tuna melt",
        "alaskan pipeline", "algophile", "algophilia", "anilingus", "anus", "apotemnophile",
        "apotemnophilia", "arrse", "auto erotic", "autoerotic", "b00bs", "babeland", "baby batter",
        "baby gravy", "baby juice", "ball batter", "ball gag", "ball gravy", "ball kicking",
        "ball licking", "ball sack", "ball sucking", "ballbag", "ballcuzi", "ballgag", "ballsack",
        "bang bros", "bang bus", "bangbros", "bangbus", "bareback", "barely legal", "barenaked",
        "bastinado", "bbw", "beastial", "beastiality", "beastility", "beaver cleaver",
        "beaver lips", "beestiality", "bellesa", "bestial", "bestiality", "big breasts",
        "big knockers", "bimbos", "birdlock", "blonde action", "blonde on blonde action",
        "blow job", "blow your load", "blowjobs", "blue waffle", "bluewaffle", "blumpkin",
        "bondage", "boner", "boob", "boobie", "booobs", "boooobs", "booooobs", "booooooobs",
        "booty call", "breasts", "brown shower", "brown showers", "brunette action", "buceta",
        "bukkake", "bullet vibe", "bullet vibrator", "bung hole", "bunghole", "busty",
        "buttcheeks", "butthole", "buttmunch", "buttplug", "camel toe", "camgirl", "camslut",
        "camwhore", "canadian porch swing", "choad", "chocolate rosebud", "chocolate rosebuds",
        "chode", "cholerophile", "cholerophilia", "cialis", "circle jerk", "circlejerk", "cl1t",
        "claustrophile", "claustrophilia", "cleveland accordion", "cleveland hot waffle",
        "cleveland steamer", "clit", "clitface", "clitoris", "clits", "coimetrophile",
        "coimetrophilia", "cokmuncher", "coksucka", "collared", "collaring", "coochie", "coochy",
        "cooter", "coprolagnia", "coprophile", "coprophilia", "cornhole", "creampie", "cum",
        "cum shot", "cum shots", "cumbubble", "cumdumpster", "cumguzzler", "cumjockey", "cummer",
        "cumming", "cums", "cumshot", "cumshots", "cumslut", "cumtart", "cunilingus",
        "cunillingus", "cunnie", "cunnilingus", "cyalis", "cyberfuc", "daterape", "ddlg",
        "deep throat", "deepthroat", "dendrophile", "dendrophilia", "dildos", "dinks", "dipsea",
        "dirsa", "dirty pillows", "dirty sanchez", "dishabiliophile", "dishabiliophilia",
        "dog style", "doggie style", "doggiestyle", "dogging", "doggy style", "doggystyle",
        "dolcett", "domination", "dominatrix", "domme", "dommes", "donkey punch", "donkeyribber",
        "doochbag", "double dong", "double penetration", "dp action", "dry hump", "dutch rudder",
        "dystychiphile", "dystychiphilia", "ecchi", "edgeplay", "ejaculate", "ejaculated",
        "ejaculates", "ejaculating", "ejaculatings", "ejaculation", "ejakulate", "electroplay",
        "emetophile", "emetophilia", "erotic", "erotism", "escort", "eskimo trebuchet", "felatio",
        "felch", "felching", "fellate", "fellating", "fellatio", "feltch", "female squirting",
        "femdom", "figging", "finger bang", "fingerbang", "fingerbanging", "fingered", "fisted",
        "fisting", "foot fetish", "foot job", "footjob", "french rudder", "frolicme", "frottage",
        "frotting", "futanari", "g spot", "gangbanged", "gangbangs", "gaybob", "gaydo", "gaysex",
        "gaywad", "genitals", "genitorture", "gerontophile", "gerontophilia", "girl on top",
        "goatcx", "goatse", "gokkun", "golden shower", "golden showers", "gooch", "goregasm",
        "grope", "hand job", "handjob", "hardcoresex", "hentai", "heshe", "hippophile",
        "hippophilia", "hoare", "hoer", "homoerotic", "hooker", "hore", "horniest", "horny",
        "hotsex", "humping", "intercourse", "jack off", "jackoff", "jailbait", "jerk mate",
        "jerk off", "jerkmate", "jism", "jiz", "jizm", "jizz", "juggs", "kennebunkport surprise",
        "kentucky klondike", "kentucky tractor puller", "kinbaku", "kinkster", "kinky", "kiunt",
        "knobbing", "knobed", "knobjocky", "knobjokey", "kondums", "kooch", "kumer", "kummer",
        "kumming", "kums", "kunilingus", "kynophile", "kynophilia", "labia",
        "leather straight jacket", "lemon party", "lemonparty", "leningrad steamer", "literotica",
        "lovemaking", "lusting", "m45terbate", "ma5terb8", "ma5terbate", "male squirting",
        "master bate", "masterb8", "masterbat3", "masterbate", "masterbation", "masterbations",
        "masturbate", "meat spin", "meatspin", "menage a trois", "menages a trois", "menophile",
        "menophilia", "mexican pancake", "milwaukee blizzard", "minge", "missionary position",
        "mississippi birdbath", "mound of venus", "mr hands", "mrhands", "muff", "muff diver",
        "muff diving", "muffdiver", "muffdiving", "mvtube", "n1gga", "n1gger", "nambla", "nawashi",
        "necrophile", "necrophilia", "nimpho", "nimphomania", "nimphomaniac", "nipple",
        "nipple clamp", "nipple clamps", "nipples", "nob jokey", "nobjocky", "nobjokey",
        "nsfw images", "nudity", "nutsack", "nutten", "nympho", "nymphomania", "nymphomaniac",
        "octopussy", "omorashi", "one cup two girls", "one guy one jar", "one jar one man",
        "one man one jar", "only fans", "onlyfans", "orgasim", "orgasims", "orgasmic", "orgasms",
        "orgy", "p0rn", "paedo bear", "paedobear", "paedophile", "paedophilia", "pain slut",
        "painslut", "panamanian petting zoo", "panooch", "panties", "parthenophile",
        "parthenophilia", "peckerhead", "pedo bear", "pedobear", "pedophile", "pedophilia",
        "pegging", "phagophile", "phagophilia", "phonesex", "pimpis", "poinephile", "poinephilia",
        "pole smoker", "polesmoker", "ponyboy", "ponygirl", "ponyplay", "poof", "poon", "poonani",
        "poonany", "poontang", "poop chute", "poopchute", "pornhub", "porno", "pornos",
        "prince albert piercing", "proctophile", "proctophilia", "pron", "pthc", "pube", "pubes",
        "punanny", "punany", "pusies", "pusse", "pussi", "pussies", "pussylicking", "pussys",
        "pusy", "queaf", "queef", "quim", "raging boner", "ramen yarmulke", "renob",
        "reverse cowgirl", "rhabdophile", "rhabdophilia", "rhypophile", "rhypophilia", "rimjaw",
        "rimjob", "rimming", "rusty trombone", "santorum", "scatophile", "scatophilia", "schlong",
        "scissoring", "seplophile", "seplophilia", "sexo", "shag", "shagger", "shaggin",
        "shagging", "shaved beaver", "shibari", "shlong", "shota", "shrimping", "skank", "skeet",
        "skittle harvest", "skittles harvest", "slut", "slutbag", "sluts", "smut", "snowballing",
        "sodomise", "sodomist", "sodomize", "sodomy", "splooge", "splooge moose", "spooge",
        "spread legs", "spunk", "strap on", "strapon", "strappado", "suicide girls",
        "sultry women", "t1tt1e5", "t1tties", "taphephile", "taphephilia", "teets", "thanatophile",
        "thanatophilia", "throating", "throbbing boner", "thumbzilla", "tit", "titt", "tittie5",
        "titties", "titty", "topless", "traumatophile", "traumatophilia", "tribadism", "tribbing",
        "tub girl", "tubgirl", "twink", "two girls one cup", "upskirt", "urethra play", "urophile",
        "urophilia", "v14gra", "v1gra", "va j j", "vagina", "venus mound", "viagra", "vibrator",
        "vjayjay", "vorarephile", "vorarephilia", "voyeur", "voyeurweb", "vulva", "wet dream",
        "wiitwd", "wolfbagging", "worldsex", "wrapping men", "wrinkled starfish", "xhamster",
        "xnxx", "xrated", "xtube", "xvideos", "xyrophile", "xyrophilia", "yaoi", "yellow shower",
        "yellow showers", "yiffy", "zippo cat", "zippocat", "zoophile", "zoophilia",
    },
    "tamil_abusive": {
        "5h1t", "5hit", "a55", "apeshit", "ar5e", "arse", "arsehole", "assbag", "assbandit",
        "assbanger", "assbite", "assclown", "asscock", "asscracker", "asses", "assface",
        "assfucker", "assfukka", "assgoblin", "asshat", "asshead", "asshole", "assholes",
        "asshopper", "assjacker", "asslick", "asslicker", "assmonkey", "assmunch", "assmuncher",
        "asspirate", "assshole", "asssucker", "asswad", "asswhole", "asswipe", "b17ch", "b1tch",
        "bampot", "bastardo", "bellend", "biatch", "big cock", "bitch", "bitcher", "bitchers",
        "bitches", "bitchin", "bitching", "black cock", "bollock", "bollocks", "bollok", "bollox",
        "bugger", "bullshit", "c0ck", "c0cksucker", "cawk", "clusterfuck", "cnut", "cock sucker",
        "cockbite", "cockburger", "cockface", "cockhead", "cockjockey", "cockknoker", "cockmaster",
        "cockmongler", "cockmongruel", "cockmonkey", "cockmunch", "cockmuncher", "cocknose",
        "cocknugget", "cocks", "cockshit", "cocksmith", "cocksmoker", "cocksuck", "cocksucked",
        "cocksucker", "cocksucking", "cocksucks", "cocksuka", "cocksukka", "cunt", "cuntface",
        "cunthole", "cuntlick", "cuntlicker", "cuntlicking", "cuntrag", "cunts", "cyberfuck",
        "cyberfucked", "cyberfucker", "cyberfuckers", "cyberfucking", "d1ck", "dickbag",
        "dickbeater", "dickface", "dickhead", "dickhole", "dickjuice", "dickmilk", "dickmonger",
        "dickslap", "dicksucker", "dickwad", "dickweasel", "dickweed", "dickwod", "dipshit",
        "dlck", "doosh", "douche", "douchebag", "duche", "dumbshit", "dumshit", "fannyfucker",
        "fatass", "fcuk", "fcuker", "fcuking", "fingerfuck", "fingerfucked", "fingerfucker",
        "fingerfuckers", "fingerfucking", "fingerfucks", "fistfuck", "fistfucked", "fistfucker",
        "fistfuckers", "fistfucking", "fistfuckings", "fistfucks", "fooker", "fucka", "fuckers",
        "fuckhead", "fuckheads", "fuckin", "fuckings", "fuckingshitmotherfucker", "fuckme",
        "fucks", "fucktard", "fucktards", "fuckwad", "fuckwads", "fuckwhit", "fuckwit", "fuckwits",
        "fuk", "fuker", "fukker", "fukkin", "fuks", "fukwhit", "fukwit", "fux", "fux0r", "gayass",
        "giant cock", "huge cock", "jackass", "kawk", "knobead", "knobend", "knobhead", "kock",
        "kunt", "kunts", "l3itch", "m0f0", "m0fo", "massive cock", "mo fo", "mof0", "mofo",
        "mothafuck", "mothafucka", "mothafuckas", "mothafuckaz", "mothafucked", "mothafucker",
        "mothafuckers", "mothafuckin", "mothafucking", "mothafuckings", "mothafucks", "motherfuck",
        "motherfucked", "motherfuckers", "motherfuckin", "motherfucking", "motherfuckings",
        "motherfuckka", "motherfucks", "muthafuckker", "mutherfucker", "nobhead", "numbnuts",
        "penisfucker", "phuck", "phuk", "phuked", "phuking", "phukked", "phukking", "phuks",
        "phuq", "piece of shit", "pigfucker", "pises", "pising", "pisof", "piss", "piss pig",
        "pissed", "pisser", "pissers", "pisses", "pissflap", "pissflaps", "pissin", "pissing",
        "pissoff", "pisspig", "scroat", "scrote", "sh1t", "shit bag", "shit bagger", "shit brain",
        "shit breath", "shit cunt", "shit eating", "shit face", "shit faced", "shit fit",
        "shit head", "shit heel", "shit hole", "shit house", "shit load", "shit pot",
        "shit spitter", "shit stain", "shitass", "shitbag", "shitbagger", "shitblimp", "shitbrain",
        "shitbreath", "shitcunt", "shitdick", "shite", "shiteating", "shitey", "shitface",
        "shitfaced", "shitfit", "shitfuck", "shitfull", "shithead", "shitheel", "shithole",
        "shithouse", "shitings", "shitload", "shitpot", "shits", "shitspitter", "shitstain",
        "shitted", "shitter", "shitters", "shittiest", "shitting", "shittings", "shity", "shiz",
        "shiznit", "smeg", "smegma", "son of a bitch", "throbbing cock", "thundercunt", "titfuck",
        "tittiefucker", "tittyfuck", "tittywank", "titwank", "tosser", "tw4t", "twat", "twathead",
        "twatlips", "twatty", "twunt", "twunter", "w00se", "wank", "wanker", "wanky", "whoar",
        # Tamil / Tanglish
        # "kai adi..." = masturbate (slang, every hit in the dataset was this)
        "kai adi", "kai adich", "kai adika", "kai adikk", "kaiadi", "கை அடிச்ச", "கை அடிக்க",
        "fap", "faps", "fapping", "fapped", "fapper",
        "otha", "oththa", "gotha", "badu", "devidiya", "dhevidiya",
        "தேவிடியா", "தேவுடியா", "கூதி", "வேசி", "மயிரு",
    },
    "caste_hate": {
        "beaner", "beaners", "boiolas", "buddha head", "buddhahead", "bulldyke", "butt pirate",
        "camel jockey", "camel jockies", "cameljockey", "cameljockies", "carpet muncher",
        "carpetmuncher", "chinc", "chink", "chinks", "chinky", "coons", "curry muncher",
        "currymuncher", "darkey", "darkie", "darkies", "darky", "doon coon", "dooncoon",
        "dune coon", "fag", "fagbag", "fagg", "fagging", "faggit", "faggitt", "faggot", "faggs",
        "fagot", "fagots", "fags", "fagtard", "fudge packer", "fudgepacker", "gaylord", "gaytard",
        "gook", "gook eye", "gookie", "gooks", "gooky", "heeb", "jigaboo", "jiggaboo", "jiggerboo",
        "kike", "kyke", "lesbo", "lezzie", "negro", "neonazi", "nig nog", "nigerian hurricane",
        "nigg3r", "nigg4h", "nigga", "niggah", "niggas", "niggaz", "nigger", "niggers", "niglet",
        "nignog", "petrol sniffer", "petrolsniffer", "pikey", "pikeys", "pnigerophile",
        "pnigerophilia", "pnigophile", "pnigophilia", "queerbait", "queerhole", "rag head",
        "rag heads", "raghead", "ragheads", "retard", "slant eye", "slanteye", "spearchucker",
        "spic", "spick", "spicks", "spics", "swamp guinea", "towel head", "towelhead", "tranny",
        "wet back", "wetback", "white power", "whitepower", "zipper head", "zipperhead",
    },
}
for _cat, _words in _LEARNED.items():
    CATEGORIES[_cat] |= _words

# Tamil slang roots with many suffix variants (sunniya, punduka, koothiya,
# etc.) — matched as a prefix rather than requiring an exact-word hit.
# Deliberately NOT used for ambiguous English roots or words with other
# legitimate meanings.
PREFIX_ROOTS = {
    # every ending of masturbate (+ common misspellings): masturbating,
    # mastrubation, masterbate, masturbation...
    "sexual": {"masturb", "mastrub", "masterba", "mastarba", "masturp", "mastub"},
    "tamil_abusive": {
        "punda", "pundai", "koothi", "koodhi", "oombu", "oombi", "mayiru",
        "thevidiya", "thevudiya", "tevadiya", "vibachari",
        "thevadiya", "thevdiya", "thevudya", "thevidya", "tevdiya", "dhevidiya",
        "dhevadiya", "fundai",
        # "munda" is NOT a prefix root: it prefixes "mundanaal" (day
        # before yesterday). The exact word "munda" is still caught via
        # CATEGORIES above.
    },
    "tamil_sexual_slang": {
        "sunni", "poolu", "nakku",
        # restored root from the original list
        "kundiadi",
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


# real words whose stretched-letter fold lands on a blocklist word:
# "less" -> "les" ("chance less" is not a slur)
_REAL_WORDS = {"less", "oothaa", "ootha", "oodha"}   # ஊதா (purple) is not "otha"


def _token_variants(tok):
    if tok in _REAL_WORDS:
        return {tok}
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
    # nothing to join unless 3+ one-letter pieces sit in a row
    if not re.search(r"(?:^|,)(?:[^,],){2}[^,](?:,|$)", ",".join(tokens)):
        return found
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
_SAFE_PHRASES_RE = re.compile(
    r"(?<![a-z0-9])cpl\s+of(?![a-z0-9])"
    # ஒத்த = "one / single": "nee oththa sollu sollu", "otha roova", "otha
    # paarvai". The swear "Otha mass da" has no counting word after it.
    r"|(?<![a-z0-9])o+t+h*t*h*a+\s+(?:sol+u\w*|va+rth\w*|pa+rv\w*|ro+[ov]a\w*|ru+pa\w*|ru+bai\w*|"
    r"a+l|na+l|kan+u|paiyan|ponn?u|pul+a|nimis\w*|thadav\w*|va+t+i|kai|kaal|ve+du|mara+m|"
    r"vartha\w*|var+u\w*|rasa|raja)(?![a-z0-9])")


# Disguised CSAM spam: "𝐃*𝐌 𝐅𝐎𝐑 𝐂𝐇!!𝐃 𝐏𝐑𝐎*𝐍", "ch1ld pr0n", "kids nudes".
# Split by whitespace first, strip symbols inside each chunk, undo leet
# ("!"/"1" -> i, "0" -> o) and repeated letters, then look for a child word
# followed (within 2 words) by a porn word as a WHOLE word, so
# "kids pronunciation" or "child video call" never match.
_CSAM_LEET = str.maketrans({"!": "i", "1": "i", "|": "i", "0": "o", "3": "e",
                            "4": "a", "@": "a", "$": "s", "5": "s", "7": "t"})
_CSAM_CHILD = {"child", "chid", "chld", "chlid", "childs", "children", "kid",
               "kids", "kiddie", "kidie", "minor", "minors", "underage", "baby"}
_CSAM_PORN = {"porn", "pron", "prn", "porno", "prono", "xxx", "nude", "nudes",
              "sexvideo", "sexvideos"}


def _csam_chunks(text):
    out = []
    for chunk in text.lower().split():
        c = re.sub(r"[^a-z]", "", chunk.translate(_CSAM_LEET))
        c = "xxx" if re.fullmatch(r"x{2,}", c) else re.sub(r"(.)\1+", r"\1", c)
        if c:
            out.append(c)
    # re-join letter-by-letter spellings: "c h i l d" -> "child"
    joined, i = [], 0
    while i < len(out):
        j = i
        while j < len(out) and len(out[j]) == 1:
            j += 1
        if j - i >= 3:
            joined.append("".join(out[i:j])); i = j
        else:
            joined.append(out[i]); i += 1
    return joined


def is_csam_spam(text):
    if not text:
        return False
    ch = _csam_chunks(_normalize_unicode(text))
    for i, w in enumerate(ch):
        if any(w.startswith(c) and w[len(c):] in _CSAM_PORN for c in _CSAM_CHILD):
            return True  # glued: "childporn", "kidsnudes"
        if w in _CSAM_CHILD and any(x in _CSAM_PORN for x in ch[i + 1:i + 3]):
            return True
    return False


# Words that are dating/18+ only in an ad shape: a WHO word (housewife,
# aunty, girls, bhabhi...) together with an OFFER/CONTACT word (available,
# ping, dm, chat, irukingla...). "Doctor from chennai Housewives ping me",
# "Housewife irukingla", "girls available dm" -> warn; "en amma housewife",
# "Is he available?" -> nothing.
_DATING_WHO = {"housewife", "housewifes", "housewives", "housewive", "bhabhi", "bhabi", "aunty",
               "aunties", "auntys", "girls", "girl", "ladies", "lady", "female", "females",
               "women", "womens", "virgin", "virgins", "ponnu", "ponnunga", "ponuga", "ponnuga",
               "boys", "male", "males", "gents"}
_DATING_ONLY_WHO = {"housewife", "housewifes", "housewives", "housewive", "bhabhi", "bhabi",
                    "virgin", "virgins"}
_DATING_OFFER = {"available", "availabe", "avaliable", "avl", "ping", "dm", "pm", "msg", "inbox", "ib",
                 "chat", "call", "vc", "contact", "number", "meet", "service", "paid", "needed",
                 "wanted", "venum", "need", "want", "looking", "friendship", "frndship", "relationship",
                 "here", "irukingla", "irukinga", "irukeengala", "irukaanga", "iruka", "irukanga",
                 "iruke", "irukehalq", "irukeengala", "irukingala", "any", "anyone", "night", "real"}

_AVAILABLE_SERVICE = {"realmeet", "meet", "meetup", "vc", "videocall", "massage", "service", "services",
                      "paid", "price", "genuine", "night", "nightchat", "fullbody", "escort"}

_WIFE_AD = {"swap", "swapping", "swaping", "share", "sharing", "exchange", "available", "avl",
            "dm", "pm", "inbox", "ib", "ping", "meet", "realmeet", "cuckold", "cuck", "service",
            "bull", "hotwife"}


# Pet names (darling, chellam, chlm, papa, babe...) are casual: "Hy
# chellam", "Hi darling", "love u chlm", "Okayy papa" -> chat, never warned.
# They warn only next to something sexual or a private invite: "Thaniya vaaa
# papa", "darling send pic", "hot chellam", "kiss me babe". "lonely" is
# never warned: "Pesalaama ngaa feeling lonely" can be an honest wish to talk.
_PET_WORDS = {"darling", "darlin", "darlng", "darlinggg", "chelam", "chellam", "chelame", "chellame",
              "chelamae", "chellamae", "chelama", "chlm", "chlmm", "papa", "paapa", "pappa", "babe",
              "babes", "honey", "sweetheart", "sweety", "sweetie", "செல்லம்",
              "செல்லமே", "பாப்பா", "டார்லிங்"}
# sexual words or a private invite only (no greetings, love, miss, cute, umma)
_PET_FLIRT = {"kiss", "kis", "sexy", "hot", "bed", "naked", "nude", "nudes", "thaniya", "thaniyaa",
              "tania", "thanimai", "alone", "pvt", "private", "personal", "photo", "pic", "pics",
              "selfie", "vc", "videocall", "romance", "romantic", "date", "hug", "touch",
              "body", "inbox", "dm", "pm"}


def _fold(w):
    return re.sub(r"(.)\1+", r"\1", w)


def is_pet_flirt(text):
    if not text:
        return False
    raw = _normalize_unicode(text).lower().split()
    words = [re.sub(r"[^a-z\u0b80-\u0bff]", "", w) for w in raw]
    words = [w for w in words if w]
    folded = [_fold(w) for w in words]
    pet = {_fold(p) for p in _PET_WORDS}
    pets = [i for i, w in enumerate(folded) if w in pet or words[i] in _PET_WORDS]
    if pets:
        others = [w for i, w in enumerate(folded) if i not in pets]
        if any(w in {_fold(f) for f in _PET_FLIRT} for w in others):
            return True
    return False


def is_dating_pair(text):
    if not text:
        return False
    words = [re.sub(r"[^a-z]", "", w) for w in _normalize_unicode(text).lower().split()]
    words = [w for w in words if w]
    # "wife swap", "wife sharing", "my wife available dm": a much narrower
    # list than for housewife, since "en wife ku call pannanum" is ordinary
    if any(w in ("wife", "wifes", "wives", "wifey") for w in words) and any(
            w in _WIFE_AD for w in words):
        return True
    who = [i for i, w in enumerate(words) if w in _DATING_ONLY_WHO or w.startswith("housewi")]
    if who:
        return any(w in _DATING_OFFER or w.startswith(("irukk", "iruk")) for w in words)
    # "available" with a who-word: "girls available", "aunty available"
    if any(w in ("available", "availabe", "avaliable", "avl") for w in words):
        # "girls available", "realmeet available", "video call available",
        # "Dm me I am available" -> ad; "Is he available?" -> chat
        joined = " ".join(words)
        return (any(w in _DATING_WHO or w in _AVAILABLE_SERVICE for w in words)
                or (re.search(r"(?<![a-z])(i am|iam|im|i m|me) (available|avl)", joined) is not None
                    and any(w in ("dm", "pm", "msg", "inbox", "ib", "ping", "text") for w in words))
                or "video call" in joined)
    return False


# 18+ group ads: "Ex grrrpp", "s3x grp", "s*x group link", "sex gc join".
# "sex"-like words + a group word always warn. Plain "ex" (also = my ex /
# old) only when a group word comes RIGHT after it and the message is short
# or has an ad word: "en ex grp la iruntha" (my old group) stays clean.
_SEXLIKE = {"sex", "sx", "sexx", "secs", "seggs", "sexy"}
_GROUP_WORDS = {"grp", "gp", "group", "groups", "gc", "channel", "chanel", "link", "links", "vc",
                "vids", "video", "videos", "chat"}
_AD_WORDS = {"join", "link", "links", "dm", "pm", "msg", "inbox", "free", "paid", "vaanga", "vanga",
             "come", "add", "telegram"}


def is_sex_group(text):
    if not text:
        return False
    low = _normalize_unicode(text).lower()
    words = []
    for w in low.split():
        w = w.replace("3", "e").replace("*", "e").replace("€", "e")
        w = re.sub(r"[^a-z]", "", w)
        w = re.sub(r"(.)\1+", r"\1", w)
        if w:
            words.append(w)
    gw = {re.sub(r"(.)\1+", r"\1", g) for g in _GROUP_WORDS}
    for i, w in enumerate(words):
        nxt = words[i + 1:i + 3]
        if w in {re.sub(r"(.)\1+", r"\1", x) for x in _SEXLIKE} and any(n in gw for n in nxt):
            return True
        if w == "ex" and i + 1 < len(words) and words[i + 1] in gw - {"chat", "vc", "video", "videos", "vids"}:
            if len(words) <= 3 or any(x in _AD_WORDS for x in words):
                return True
    return False


# Words that are fine alone and only hit in a context (same idea as
# is_dating_pair). Returns a set of categories.
_CASTE_WORDS = {"jaathi", "jathi", "jaadhi", "jadhi", "jati", "jaati", "saathi", "sathi", "caste"}
_CASTE_INSULT = {"keel", "kezh", "kizh", "kilh", "kezhu", "low", "lower", "lowest", "thaazhntha",
                 "thazhntha", "thaalntha", "thalntha", "naaye", "naye", "nai", "naai", "kevalam",
                 "pariah", "inferior", "dog", "dogs", "scum", "kudi", "kedu", "payale", "payalae",
                 "pasanga", "kootam", "koottam"}
_CASTE_AGAINST = {"vendam", "venam", "ethuku", "edhuku", "yethuku", "illa", "ila", "no", "stop",
                  "ozhiga", "ozhika", "ozhiyanum", "kidaiyathu", "kidayathu", "against", "vida",
                  "vidunga", "paakatheenga", "pakatheenga", "paakathinga"}
_GAYS_18 = {"sex", "sexy", "chat", "dm", "pm", "inbox", "meet", "dating", "date", "nude", "nudes",
            "only", "boys", "men", "vc", "call", "hookup", "top", "bottom", "tg", "telegram",
            "group", "grp", "link"}
_PAID_AD = {"girls", "girl", "aunty", "aunties", "ladies", "call", "vc", "video", "massage", "body",
            "available", "avl", "meet", "realmeet", "night", "dm", "pm", "msg", "inbox", "ib",
            "genuine", "escort", "spa", "nude", "nudes", "sex", "service", "paid", "price",
            "rate", "venumo", "venuma", "housewife", "housewives", "real", "verified", "demo"}


def is_context_hit(text):
    if not text:
        return set()
    words = [re.sub(r"[^a-z]", "", w) for w in _normalize_unicode(text).lower().split()]
    words = [w for w in words if w]
    ws = set(words)
    out = set()
    # caste: "keel jaathi naaye" -> hate; "jaathi veri vendam" -> fine
    if ws & _CASTE_WORDS or "keel" in ws:
        insult = ws & _CASTE_INSULT or any(w.startswith(("naai", "naay", "naaig", "nayg", "thalthap",
                                                          "thazhthap", "thaazhthap")) for w in ws)
        if insult and not ws & _CASTE_AGAINST:
            out.add("caste_hate")
    # "gays" next to an 18+ word: "gays only chat", "gays dm"
    if ("gays" in ws or "gayz" in ws) and ws & _GAYS_18:
        out.add("gender_orientation")
    # paid/service ad: "paid service available", "service venumo dm"
    if ws & {"paid", "service", "services"}:
        if len(ws & _PAID_AD - {"paid", "service"}) >= 1 and (
                "paid" in ws and "service" in ws or ws & (_PAID_AD - {"paid", "service", "price", "rate", "real"})):
            out.add("service_paid")
    return out


_TERM_CACHE = {}


def _split_terms(category, terms):
    key = (category, len(terms))
    if key not in _TERM_CACHE:
        script = {t for t in terms if any(ord(c) > 0x2000 for c in t)}
        phrase = {t for t in terms - script if " " in t}
        _TERM_CACHE[key] = (script, phrase, terms - script - phrase)
    return _TERM_CACHE[key]


def check_text(text):
    """
    Returns (is_hit, categories) where categories is a set of category
    names that matched. is_hit is True if any category matched.
    """
    if not text:
        return False, set()

    if is_csam_spam(text):
        return True, {"severe_exploitation"}
    _dating = is_dating_pair(text) or is_pet_flirt(text)
    _sexgroup = is_sex_group(text)
    _context = is_context_hit(text)

    text = _normalize_unicode(text)
    # "@" inside a word is an "a": "m@sturbate", "s@xy" (as "Vij@y" = Vijay)
    text = re.sub(r"(?<=[A-Za-z])@(?=[A-Za-z])", "a", text)
    lowered = _SAFE_PHRASES_RE.sub(" ", text.lower())
    tokens = _WORD_RE.findall(lowered)
    hit_categories = set()

    # 1) phrase / substring matching (multi-word terms, non-Latin script,
    #    or single words where substring containment is intentional)
    # every spelling variant of every word in the message, computed once
    tok_vars = set()
    for tok in tokens:
        tok_vars |= _token_variants(tok)
    for category, terms in CATEGORIES.items():
        script_terms, phrase_terms, single_word_terms = _split_terms(category, terms)
        for term in script_terms:
            # non-Latin script (Tamil): plain substring, unchanged
            if term in lowered:
                hit_categories.add(category)
                break
        if category not in hit_categories:
            for term in phrase_terms:
                if _phrase_hit(term, lowered):
                    hit_categories.add(category)
                    break
        # single-word term: exact token + common evasion variants rather
        # than raw substring (avoids matching "ass" inside "class")
        if category not in hit_categories and tok_vars & single_word_terms:
            hit_categories.add(category)

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

    if _dating:
        hit_categories.add("relationship_dating")
    if _sexgroup:
        hit_categories.add("sexual")
    hit_categories |= _context
    return (len(hit_categories) > 0), hit_categories
