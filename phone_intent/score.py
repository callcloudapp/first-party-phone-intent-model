"""Score a call log: every number gets P1-P4, a 0-100 score, the best local time to call, and why.

  python -m phone_intent.score my_calls.csv --out scores.csv            # the starter model
  python -m phone_intent.score my_calls.csv --models my_models/         # models you trained on your own data

The CSV is your own dialer / CRM export, one row per call. Columns are found by name (case-insensitive):
  number     phone, phone_number, number, to, dialed, contact_phone      (E.164 is best; +1 is assumed for 10 digits)
  called at  called_at, date, timestamp, created_at, start_time, time    (ISO 8601; a UTC offset is best)
  outcome    outcome, disposition, result, status, call_result           (free text, mapped below)
  duration   duration, duration_s, talk_time, seconds                    (optional: a long call with no outcome = a person)
  caller id  caller_id, from, from_number                                (optional)
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import FEATURES, build_features
from .labels import DEAD, NO_PERSON, PERSON
from .tiers import tier

STARTER = Path(__file__).resolve().parent / "models"
HOURS = [h + 0.5 for h in range(9, 17)]  # 9:30 .. 16:30 local: business hours (evening calls are too rare to trust)
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Free-text outcomes -> what happened. Order matters: the first rule that matches wins.
_RULES = [
    (DEAD, r"wrong number|bad number|invalid|disconnect|not in service|no longer in service|unallocated|does not exist|fax"),
    (NO_PERSON, r"voice ?mail|\bvm\b|left (a )?message|machine|answering|screen|ivr|automated|phone menu|no answer|"
                r"not answered|unanswered|busy|ring(ing)? out|no one|nobody|keine antwort|mailbox"),
    (PERSON, r"connect|conversation|spoke|talked|answered|human|person|meeting|demo|interested|not interested|call ?back|"
             r"gatekeeper|referr|not the right|decision maker|hung up|qualified|do not call|dnc|reached"),
]
_IGNORE = r"cancel|dropped|abandon|failed|error|skipped by dialer"


def normalize_outcome(text: str, duration_s: float | None = None) -> str | None:
    t = (text or "").strip().lower()
    if t and re.search(_IGNORE, t):
        return None
    for label, pattern in _RULES:
        if t and re.search(pattern, t):
            return label
    if duration_s is not None and duration_s >= 30:
        return PERSON
    return None


def _col(df: pd.DataFrame, names: list[str]) -> str | None:
    low = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    return None


def _e164(raw: str) -> str:
    d = re.sub(r"[^\d+]", "", str(raw))
    if d.startswith("+"):
        return d
    d = d.lstrip("0") if len(d) > 11 else d
    if len(d) == 10:
        return "+1" + d
    if len(d) == 11 and d.startswith("1"):
        return "+" + d
    return "+" + d if d else ""


def load_calls(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    c_num = _col(raw, ["phone", "phone_number", "number", "to", "dialed", "contact_phone", "dialed_number", "e164"])
    c_at = _col(raw, ["called_at", "date", "timestamp", "created_at", "start_time", "time", "call_date", "datetime"])
    c_out = _col(raw, ["outcome", "disposition", "result", "status", "call_result", "call_outcome"])
    c_dur = _col(raw, ["duration", "duration_s", "talk_time", "seconds", "call_duration"])
    c_cid = _col(raw, ["caller_id", "from", "from_number", "caller"])
    c_car = _col(raw, ["carrier_family", "carrier"]); c_lt = _col(raw, ["lrn_line_type", "line_type", "phone_type"]); c_po = _col(raw, ["ported"])
    missing = [n for n, c in (("number", c_num), ("called at", c_at), ("outcome", c_out)) if c is None]
    if missing:
        raise SystemExit(f"{path}: could not find a column for {', '.join(missing)} (columns: {', '.join(raw.columns)})")
    dur = pd.to_numeric(raw[c_dur], errors="coerce") if c_dur else pd.Series(np.nan, index=raw.index)
    return pd.DataFrame({
        "number": [_e164(x) for x in raw[c_num]],
        "created_at": pd.to_datetime(raw[c_at], utc=True, errors="coerce", format="mixed"),
        "caller_id": [_e164(x) if x else "" for x in raw[c_cid]] if c_cid else "",
        "outcome": [normalize_outcome(o, None if np.isnan(d) else d) for o, d in zip(raw[c_out], dur)],
        "raw_outcome": raw[c_out],
        **({"carrier_family": raw[c_car].str.lower()} if c_car else {}),
        **({"lrn_line_type": raw[c_lt].str.lower().replace({"mobile": "wireless", "cell": "wireless"})} if c_lt else {}),
        **({"ported": pd.to_numeric(raw[c_po].str.lower().replace({"true": "1", "false": "0", "yes": "1", "no": "0"}), errors="coerce")} if c_po else {}),
    }).dropna(subset=["created_at"]).query("number != ''")


def score_calls(calls: pd.DataFrame, models: Path = STARTER, as_of: datetime | None = None) -> pd.DataFrame:
    # The starter ships two model pairs: plain, and one that also reads carrier lookups (carrier_family / lrn_line_type
    # / ported), which ranks clearly better (forward test 10-18: within-list AUC 0.70 vs 0.64). Carrier columns in the
    # CSV + the starter models = the carrier pair.
    has_carrier = any(c in calls.columns and calls[c].astype(str).str.len().gt(0).any() for c in ("carrier_family", "lrn_line_type", "ported"))
    if has_carrier and Path(models) == STARTER and (STARTER / "carrier" / "person_model.txt").exists():
        models = STARTER / "carrier"
    person = lgb.Booster(model_file=str(models / "person_model.txt"))
    dead = lgb.Booster(model_file=str(models / "dead_model.txt"))
    as_of = as_of or datetime.now(timezone.utc)
    feats_used = person.feature_name()  # the model's own inputs (a carrier-trained model also wants carrier columns)
    carrier_cols = [c for c in ("carrier_family", "lrn_line_type", "ported") if c in calls.columns]
    hist = calls[["number", "created_at", "caller_id", "outcome"] + carrier_cols].copy()
    hist["created_at"] = hist["created_at"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    # one extra "next dial" per number, now: its features describe everything known before calling again
    nxt = pd.DataFrame({"number": hist["number"].unique()})
    nxt["created_at"] = as_of.strftime("%Y-%m-%dT%H:%M:%S%z")
    last_cid = calls.sort_values("created_at").groupby("number")["caller_id"].last()
    nxt["caller_id"] = nxt["number"].map(last_cid).fillna("")
    nxt["outcome"] = None
    for c in carrier_cols:  # a number's carrier facts carry over to its next dial
        nxt[c] = nxt["number"].map(calls.groupby("number")[c].last())
    nxt["_next"] = True
    feats = build_features(pd.concat([hist.assign(_next=False), nxt], ignore_index=True))
    rows = feats[feats["_next"] == True].reset_index(drop=True)  # noqa: E712

    # best local weekday x hour for each number
    best_p = np.full(len(rows), -1.0); best_h = np.zeros(len(rows)); best_d = np.zeros(len(rows), dtype=int)
    for d in range(5):
        for h in HOURS:
            x = rows[feats_used].copy(); x["local_hour"] = h; x["local_weekday"] = d
            p = person.predict(x)
            better = p > best_p
            best_p[better], best_h[better], best_d[better] = p[better], h, d
    p_dead = dead.predict(rows[dead.feature_name()])
    # a number that came back dead and never reached a person since is bad data whatever the model says
    known_dead = (rows["prior_dead"] > 0) & (rows["days_since_last_person"].isna() | (rows["prior_person"] == 0))
    p_dead = np.where(known_dead, 1.0, p_dead)
    t = tier(best_p, p_dead)

    def why(r, tr):
        if tr == "P4":
            return "came back as a bad number" if r.prior_dead > 0 else "looks like a dead or wrong line"
        bits = []
        if r.prior_person > 0:
            bits.append(f"reached a person {int(r.prior_person)}x" + (f", last {int(r.days_since_last_person)}d ago" if r.days_since_last_person == r.days_since_last_person else ""))
        elif r.prior_tries > 0:
            bits.append(f"no person in {int(r.prior_tries)} tries")
        if r.prior_no_person > 0 and r.prior_person == 0:
            bits.append("voicemail / no answer so far")
        return "; ".join(bits) or "first look"

    out = pd.DataFrame({
        "number": rows["number"], "tier": t, "score": np.round(best_p * 100, 1),
        "best_local_time": [f"{DAYS[d]} {int(h)}:{'30' if h % 1 else '00'}" for d, h in zip(best_d, best_h)],
        "tries": rows["prior_tries"].astype(int), "people_reached": rows["prior_person"].astype(int),
        "dead_chance": np.round(p_dead * 100, 1),
        "state": rows["state"].astype(str),
    })
    out["reason"] = [why(r, tr) for r, tr in zip(rows.itertuples(), t)]
    order = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
    return out.sort_values(["tier", "score"], key=lambda s: s.map(order) if s.name == "tier" else -s).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score phone numbers P1-P4 from your own call log.")
    ap.add_argument("calls", help="CSV export of your calls, one row per call")
    ap.add_argument("--out", default="scores.csv")
    ap.add_argument("--models", default=str(STARTER), help="folder with person_model.txt + dead_model.txt")
    ap.add_argument("--as-of", help="score as of this time (ISO 8601), default now")
    a = ap.parse_args()
    calls = load_calls(a.calls)
    unknown = calls["outcome"].isna().mean()
    print(f"{len(calls):,} calls, {calls['number'].nunique():,} numbers; outcomes not recognised on {unknown:.0%} "
          f"(e.g. {', '.join(calls.loc[calls['outcome'].isna(), 'raw_outcome'].value_counts().head(5).index)})")
    as_of = datetime.fromisoformat(a.as_of) if a.as_of else None
    res = score_calls(calls, Path(a.models), as_of)
    res.to_csv(a.out, index=False)
    print(res["tier"].value_counts().reindex(["P1", "P2", "P3", "P4"]).fillna(0).astype(int).to_string())
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
