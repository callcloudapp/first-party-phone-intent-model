"""P1-P4 phone intent tiers, TitanX-style: one tier per NUMBER (contact), not per dial.

  P1  high intent       call first: the most likely to put a real person on the line
  P2  moderate          worth calling when the P1s are done
  P3  low / no intent   rarely a person (voicemail, screeners, gatekeepers, never answers)
  P4  bad data          the number does not work (dead, disconnected, wrong line)

Two models make the tier: P(a real person answers) and P(the number is dead). Each number is scored at its best
calling hour (the hour of the day, in its own time zone, where P(person) peaks), which is also the time we recommend.

Tiers are WITHIN ONE LIST (a customer's numbers ranked against each other): P4 when P(dead) is high; of the rest,
P1 = the top P1_SHARE by P(person), P2 = the next P2_SHARE, P3 = everything else. A fixed cut-off ("2x the average")
put ~1% of numbers in P1/P2 and misfired on lists whose base rate differs; ranking within the list does neither.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

P1_SHARE, P2_SHARE, DEAD_P = 0.10, 0.20, 0.5
CALL_HOURS = [h + 0.5 for h in range(8, 19)]  # 8:30 .. 18:30 local


def best_hour_scores(model, rows: pd.DataFrame, features: list[str], hours=CALL_HOURS) -> tuple[np.ndarray, np.ndarray]:
    """P(person) for each row at each candidate local hour; returns (best score, best hour)."""
    scores = []
    for h in hours:
        x = rows[features].copy()
        x["local_hour"] = h
        scores.append(model.predict(x))
    s = np.vstack(scores)  # hours x rows
    i = s.argmax(axis=0)
    return s[i, np.arange(s.shape[1])], np.asarray(hours)[i]


def tier(p_person: np.ndarray, p_dead: np.ndarray, lists: np.ndarray | None = None) -> np.ndarray:
    """P1-P4 per row; `lists` = which list each row belongs to (ranked separately), None = one list."""
    p_person, p_dead = np.asarray(p_person, float), np.asarray(p_dead, float)
    lists = np.zeros(len(p_person)) if lists is None else np.asarray(lists)
    out = np.full(len(p_person), "P3", dtype=object)
    out[p_dead >= DEAD_P] = "P4"
    for l in np.unique(lists):
        live = np.where((lists == l) & (p_dead < DEAD_P))[0]
        if not len(live):
            continue
        rank = pd.Series(p_person[live]).rank(ascending=False, method="first").to_numpy() / len(live)
        out[live[rank <= P1_SHARE]] = "P1"
        out[live[(rank > P1_SHARE) & (rank <= P1_SHARE + P2_SHARE)]] = "P2"
    return out
