"""Inputs to the phone intent model. Everything is known BEFORE the dial and exists in any company's own call log
(a number, when it was called, what happened):

  the number itself     country, area code (NPA), state / region (from the numbering plan, no lookup needed)
  the moment            the callee's LOCAL hour and weekday (from the number's time zone)
  the caller ID         whether it shares the callee's area code / state (when the log has the caller ID)
  the number's history  earlier dials of THIS number, and WHEN they happened:
                          tries, people, voicemails / no-answers, dead results, overall person rate
                          hours since the last try, days since the first, days since the last person / last miss
                          tries and people in the last 30 days, a person rate that weighs recent dials more
                          (30-day half-life), people reached near this local hour and on this weekday before

History only uses dials strictly BEFORE the one being scored, so nothing about a dial leaks into its own inputs.
Carrier data (carrier family, LRN line type, ported) is optional: CARRIER_FEATURES, used only with --carrier.
"""
from __future__ import annotations

from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import phonenumbers
from phonenumbers import geocoder, timezone as pn_tz

from .labels import DEAD, NO_PERSON, PERSON

CATEGORICAL = ["country", "npa", "state"]
NUMERIC = [
    "local_hour", "local_weekday", "caller_same_npa", "caller_same_state",
    "prior_tries", "prior_person", "prior_no_person", "prior_dead", "prior_person_rate",
    "hours_since_last", "days_since_first", "days_since_last_person", "days_since_last_miss",
    "tries_30d", "person_30d", "decayed_person_rate", "person_near_hour", "person_same_weekday",
]
FEATURES = CATEGORICAL + NUMERIC
CARRIER_CATEGORICAL = ["carrier_family", "lrn_line_type"]
CARRIER_FEATURES = CARRIER_CATEGORICAL + ["ported"]

HALF_LIFE_DAYS = 30.0
NEAR_HOUR = 1.5  # "near this hour" = within 1.5 h of local time


@lru_cache(maxsize=300_000)
def number_facts(e164: str) -> tuple[str, str, str, str]:
    """(country code, NPA / leading digits, state or region, IANA time zone) for a number; cached."""
    try:
        p = phonenumbers.parse(e164, None)
    except phonenumbers.NumberParseException:
        return ("", "", "", "")
    nsn = phonenumbers.national_significant_number(p)
    cc = str(p.country_code)
    npa = nsn[:3] if cc == "1" and len(nsn) == 10 else nsn[:2]
    region = geocoder.description_for_number(p, "en") or ""
    zones = [z for z in pn_tz.time_zones_for_number(p) if z != "Etc/Unknown"]
    return (cc, npa, region, zones[0] if zones else "")


@lru_cache(maxsize=1000)
def _zone(name: str) -> ZoneInfo | None:
    try:
        return ZoneInfo(name) if name else None
    except Exception:  # noqa: BLE001 - an unknown zone just means no local time
        return None


def _local_time(df: pd.DataFrame) -> None:
    df["local_hour"] = np.nan
    df["local_weekday"] = np.nan
    for tz, idx in df.groupby("tz").groups.items():
        z = _zone(tz)
        if z is None:
            continue
        local = df.loc[idx, "created_at"].dt.tz_convert(z)
        df.loc[idx, "local_hour"] = (local.dt.hour + local.dt.minute / 60.0).to_numpy()
        df.loc[idx, "local_weekday"] = local.dt.weekday.to_numpy()


def _history(df: pd.DataFrame) -> None:
    """Date-aware history per number, strictly before each row (df sorted by number, then time)."""
    n = len(df)
    num = df["number"].to_numpy()
    t = df["created_at"].to_numpy().astype("datetime64[s]").astype(np.int64) / 86400.0  # days
    out = df["outcome"].to_numpy()
    hour = df["local_hour"].to_numpy()
    wday = df["local_weekday"].to_numpy()
    cols = {k: np.full(n, np.nan) for k in ("days_since_last_person", "days_since_last_miss", "tries_30d", "person_30d",
                                           "decayed_person_rate", "person_near_hour", "person_same_weekday")}
    lam = np.log(2) / HALF_LIFE_DAYS
    prev = None
    for i in range(n):
        if num[i] != prev:  # a new number: empty history
            prev = num[i]
            last_p = last_m = None
            past_t, past_person, past_hour, past_wday = [], [], [], []
            dw_p = dw_t = 0.0
            t_dw = t[i]
        # decay the running sums to this row's time
        dw_p *= np.exp(-lam * (t[i] - t_dw)); dw_t *= np.exp(-lam * (t[i] - t_dw)); t_dw = t[i]
        if last_p is not None:
            cols["days_since_last_person"][i] = t[i] - last_p
        if last_m is not None:
            cols["days_since_last_miss"][i] = t[i] - last_m
        if past_t:
            pt = np.asarray(past_t); pp = np.asarray(past_person)
            recent = pt >= t[i] - 30
            cols["tries_30d"][i] = recent.sum()
            cols["person_30d"][i] = (pp & recent).sum()
            cols["decayed_person_rate"][i] = (dw_p + 0.07) / (dw_t + 1.0)  # shrunk toward the fleet's ~7% rate
            if not np.isnan(hour[i]):
                ph = np.asarray(past_hour)
                cols["person_near_hour"][i] = (pp & (np.abs(ph - hour[i]) <= NEAR_HOUR)).sum()
                cols["person_same_weekday"][i] = (pp & (np.asarray(past_wday) == wday[i])).sum()
        else:
            cols["tries_30d"][i] = 0
            cols["person_30d"][i] = 0
        # this row joins the history only if it was informative
        o = out[i]
        if o is not None and o == o:  # not None / NaN
            is_p = o == PERSON
            past_t.append(t[i]); past_person.append(is_p); past_hour.append(hour[i]); past_wday.append(wday[i])
            dw_t += 1.0
            if is_p:
                dw_p += 1.0
                last_p = t[i]
            elif o in (NO_PERSON, DEAD):
                last_m = t[i]
    for k, v in cols.items():
        df[k] = v


def build_features(dials: pd.DataFrame) -> pd.DataFrame:
    """dials: created_at (UTC or ISO 8601 with offset), number (E.164), caller_id (E.164 or empty, optional), outcome
    (labels.PERSON / NO_PERSON / DEAD / None), optional carrier columns. Returns one row per dial with FEATURES (and
    CARRIER_FEATURES when present) + outcome, sorted by number then time."""
    df = dials.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, format="ISO8601")
    if "caller_id" not in df:
        df["caller_id"] = ""
    df = df.sort_values(["number", "created_at"], kind="stable").reset_index(drop=True)

    facts = pd.DataFrame([number_facts(x) for x in df["number"].astype(str)], columns=["country", "npa", "state", "tz"])
    cfacts = pd.DataFrame([number_facts(str(x)) if isinstance(x, str) and x.startswith("+") else ("", "", "", "")
                           for x in df["caller_id"]], columns=["c_country", "c_npa", "c_state", "c_tz"])
    df = pd.concat([df, facts, cfacts], axis=1)
    _local_time(df)

    df["caller_same_npa"] = np.where(df["c_npa"] == "", np.nan, (df["c_npa"] == df["npa"]).astype(float))
    df["caller_same_state"] = np.where(df["c_state"] == "", np.nan, (df["c_state"] == df["state"]).astype(float))

    informative = df["outcome"].notna()
    for name, flag in (("prior_tries", informative), ("prior_person", df["outcome"] == PERSON),
                       ("prior_no_person", df["outcome"] == NO_PERSON), ("prior_dead", df["outcome"] == DEAD)):
        df[name] = flag.astype(int).groupby(df["number"]).cumsum() - flag.astype(int)
    df["prior_person_rate"] = np.where(df["prior_tries"] > 0, df["prior_person"] / df["prior_tries"].clip(lower=1), np.nan)
    g = df.groupby("number")["created_at"]
    df["hours_since_last"] = (df["created_at"] - g.shift(1)).dt.total_seconds() / 3600.0
    df["days_since_first"] = (df["created_at"] - g.transform("min")).dt.total_seconds() / 86400.0
    _history(df)

    for c in CARRIER_CATEGORICAL:
        df[c] = df[c].fillna("").astype(str) if c in df else ""
    df["ported"] = pd.to_numeric(df["ported"], errors="coerce") if "ported" in df else np.nan
    for c in CATEGORICAL + CARRIER_CATEGORICAL:
        df[c] = df[c].fillna("").astype("category")
    return df
