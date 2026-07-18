# Group Moderation Bot

Two independent moderation layers, checked in order:

1. **Hardcoded blocklist** (`blocklist.py`) — categorized explicit/abusive/
   caste/scam terms and emojis. Any match is treated as certain: the
   message is deleted immediately and the sender takes a strike, which
   escalates warn → mute → ban via `strikes.py`. No training data needed,
   no ambiguity.
2. **Learned classifier** (`trainer.py` / `predict.py`) — a small
   per-anchor Naive-Bayes-style model trained on your own labeled
   examples, for judgment calls the blocklist can't make: is "akka msg
   me" solicitation or an innocent question? Only runs if the blocklist
   found nothing.

The bot also tracks conversation context to distinguish an instigator
from a provoked victim, and every admin correction becomes new training
data that retrains the classifier immediately.

## Files

| File | Purpose |
|---|---|
| `config.py` | All settings, read from env vars |
| `generate_session.py` | Run ONCE locally to log in with your phone/OTP and get a `SESSION_STRING` |
| `blocklist.py` | Hardcoded categorized word/emoji blocklist — checked first, deterministic |
| `trainer.py` | Feature extraction + training (`extract_features`, `train`) for the learned classifier |
| `predict.py` | Scores a message against the learned classifier, returns warn/safe/uncertain |
| `feedback_loop.py` | Turns an admin decision into a new labeled example + retrain |
| `conversation_tracker.py` | Rolling per-chat message history + instigator lookup |
| `strikes.py` | Persistent strike counts + escalation (warn → mute → ban), used by the blocklist path |
| `bot.py` | Pyrogram **userbot** wiring all of the above together |
| `weights.json`, `labeled_data.jsonl` | Seed data (your existing labeled examples) for the learned classifier |
| `.env.example` | Template for local/Railway environment variables |

## ⚠️ This is a userbot, not a bot-token bot

`bot.py` logs in as your **real Telegram account** (phone number + OTP),
not an official `@BotFather` bot. This means:

- It can see and act on full group history, not just messages sent
  after it joined.
- It needs to be an **admin of the group on your real account** for
  delete/mute/ban to work — same permission requirement as a bot
  account would have.
- **Telegram's Terms of Service restrict automated behavior on regular
  user accounts.** Frequent automated deletes/mutes/bans from a
  personal account are more likely to trigger Telegram's spam/abuse
  detection than the same actions from an official bot account, which
  can result in your account being limited or banned. This is a real
  risk with your own account — use a dedicated/secondary Telegram
  account for this rather than your primary one if possible.

## One-time login (required before first run, and before every fresh deploy)

Railway can't show you an OTP prompt during boot, so you have to log in
once, locally, and turn that login into a reusable string:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export API_ID=...        # from https://my.telegram.org
export API_HASH=...
export PHONE_NUMBER=+91XXXXXXXXXX
python generate_session.py
```

Enter the OTP Telegram texts you (and your 2FA password if you have
Two-Step Verification on). It prints:

```
SESSION_STRING=1BVtsO...
```

Copy that whole value — this is what goes in `SESSION_STRING` locally
and on Railway from then on. **Treat it like a password**: anyone with
this string can act as your account without needing your phone again.
Never commit it to git. If it ever leaks, revoke it immediately from
Telegram Settings → Devices → find the matching session → Terminate.

## Local setup

```bash
cp .env.example .env   # fill in API_ID, API_HASH, SESSION_STRING (from above), ADMIN_CHAT_ID
export $(cat .env | xargs)
python bot.py
```

## Deploying on Railway

1. Push this folder to a GitHub repo, then "New Project → Deploy from GitHub" in Railway.
2. Railway will detect `Procfile` and run `python bot.py` as a worker.
3. Add a **Volume**, mount it at `/data`.
4. Set service variables (Settings → Variables): `API_ID`, `API_HASH`,
   `SESSION_STRING` (from `generate_session.py`, run locally first —
   **do not** set `PHONE_NUMBER` on Railway, it can't handle the OTP
   prompt), `ADMIN_CHAT_ID`, `DATA_DIR=/data`, plus any threshold
   overrides from `.env.example` you want to change.
5. On first boot, if `/data/weights.json` doesn't exist yet, copy the seed
   `weights.json`/`labeled_data.jsonl` from this repo into the volume once
   (e.g. via a one-off Railway shell session), so training history isn't lost.
6. **Make sure your account is promoted to admin in the group**, with
   delete + restrict/ban permissions. Every Telegram API call in `bot.py`
   is wrapped in `try/except RPCError: pass`, so if it lacks
   permissions, actions will silently fail instead of erroring loudly —
   test with a throwaway message first.
7. Deploy. Check logs for `Starting moderation bot...`.
8. If the session ever gets revoked (you'll see login errors in the
   logs), rerun `generate_session.py` locally and update
   `SESSION_STRING` on Railway.

## How a message is handled, in order

1. **Blocklist check** (`blocklist.check_text`). Any hit (sexual terms,
   Tamil abusive/slang, caste terms, scam terms, certain emojis, etc.) →
   delete the message, call `strikes.add_strike(user_id)`, act on the
   returned tier:
   - strike 1–2 → **warn** (message posted in-group, logged to admin)
   - strike 3–4 → **mute** for `MUTE_MINUTES`
   - strike 5+ → **ban**
   This is a flat rule regardless of what else is in the message — a
   blocklist word always goes through this same ladder, whether or not
   DM-solicitation wording is also present.

2. **If no blocklist hit**, the learned classifier scores the message
   using anchors `pm dm msg message inbox akka aunty fwb ex`:
   - score ≥ `WARN_THRESHOLD` (default 1.5) → **mute** directly
     (`DM_MUTE_HOURS`, default 3h), tagged "no dm"
   - score in between, but at least one anchor word matched → **also
     mute directly**. This covers cases like "any akka dm" that
     mention solicitation but don't score high enough to clear the
     threshold on their own — treated as mute rather than sent to
     admin review.
   - score ≤ `SAFE_THRESHOLD` (default -1.5), or no anchor word
     present at all → no action.

   In practice this means the old "uncertain → admin review queue"
   path rarely fires anymore — almost every anchor-containing message
   resolves to mute, and everything else resolves to safe. The
   `fb_warn` / `fb_safe` admin buttons and `feedback_loop.py` retrain
   flow are still wired in for the rare remaining edge case, but the
   classifier won't be getting much live correction data this way
   anymore. If you notice it drifting, add examples to
   `labeled_data.jsonl` directly and rerun `python trainer.py`.

3. Every message, regardless of outcome, is recorded into
   `conversation_tracker`'s rolling window for the instigator/victim
   logic below.

## How the instigator/victim logic works

For any message treated as abusive (blocklist hit, or classifier
mute), the bot looks backward through recent chat history (reply chain
first, then a time-windowed fallback) for an earlier abusive message
aimed at the same user. If found, that earlier sender is treated as the
instigator: the victim's message is deleted with a soft note and no
extra strike, and the mute goes to the instigator instead.

**This part is deterministic** — reply chains and timestamps aren't
guesses. **What isn't deterministic** is the upstream flag itself (blocklist
hit or classifier mute). Two real limitations to know about:

- If the instigator's opening message wasn't flagged in the first
  place, this logic can't reach back to protect the person who got
  baited into a stronger reply.
- Someone can game this by staying just under the threshold while
  baiting a target into an over-threshold reply.

Because of this, victim-side actions are intentionally soft (delete +
note, no strike), and every action is logged to `ADMIN_CHAT_ID` so a
human can catch what the automation missed.

## Deploying on Railway

1. Push this folder to a GitHub repo, then "New Project → Deploy from GitHub" in Railway.
2. Railway will detect `Procfile` and run `python bot.py` as a worker.
3. Add a **Volume**, mount it at `/data`.
4. Set service variables (Settings → Variables): `API_ID`, `API_HASH`,
   `BOT_TOKEN`, `ADMIN_CHAT_ID`, `DATA_DIR=/data`, plus any threshold
   overrides from `.env.example` you want to change.
5. On first boot, if `/data/weights.json` doesn't exist yet, copy the seed
   `weights.json`/`labeled_data.jsonl` from this repo into the volume once
   (e.g. via a one-off Railway shell session), so training history isn't lost.
6. **Make sure the bot account is promoted to admin in the group**, with
   delete + restrict/ban permissions. Every Telegram API call in `bot.py`
   is wrapped in `try/except RPCError: pass`, so if the bot lacks
   permissions, actions will silently fail instead of erroring loudly —
   test with a throwaway message first.
7. Deploy. Check logs for `Starting moderation bot...`.

## Editing the blocklist

`blocklist.py` has categorized word sets you can edit directly — no
retraining needed, changes apply on next bot restart. Multi-word phrases
(`"fake id"`) match as substrings; single words match on token
boundaries. The `PM_CATEGORY` (`dm`, `pm`, `msg`, `inbox`...) is
deliberately excluded from hard-blocking by default since those words
are too common in innocent messages — that's left to the learned
classifier, which understands context.

## Improving classifier accuracy

Your dataset is small and imbalanced per anchor (`pm`, `dm`, `inbox`,
`message`, `fwb`'s safe side especially, `ex` has none). Since live
admin review rarely fires anymore (see above), add real examples
directly to `labeled_data.jsonl` — one JSON object per line,
`{"text": "...", "warn": true/false}` — and run `python trainer.py` to
regenerate `weights.json`. Aim for roughly 8–10 real examples per anchor
in both directions before trusting auto-actions on that anchor.
