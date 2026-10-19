"""phone-intent: score phone numbers P1-P4 from your own call log, or train models on it.

  phone-intent score my_calls.csv [--out scores.csv] [--models DIR] [--as-of ISO]
  phone-intent train my_calls.csv --out my_models/
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from .features import CATEGORICAL, FEATURES, build_features
from .labels import DEAD, PERSON
from .score import STARTER, load_calls, score_calls
from .train import PARAMS


def _score(a) -> None:
    calls = load_calls(a.calls)
    unknown = calls["outcome"].isna()
    print(f"{len(calls):,} calls, {calls['number'].nunique():,} numbers")
    if unknown.any():
        top = calls.loc[unknown, "raw_outcome"].value_counts().head(5)
        print(f"outcomes not recognised on {unknown.mean():.0%} of calls (ignored): " + ", ".join(f"{k!r} x{v}" for k, v in top.items()))
    res = score_calls(calls, Path(a.models), datetime.fromisoformat(a.as_of) if a.as_of else None)
    res.to_csv(a.out, index=False)
    print(res["tier"].value_counts().reindex(["P1", "P2", "P3", "P4"]).fillna(0).astype(int).to_string())
    print(f"wrote {a.out}")


def _train(a) -> None:
    calls = load_calls(a.calls)
    df = build_features(calls[["number", "created_at", "caller_id", "outcome"]].assign(
        created_at=lambda d: d["created_at"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")))
    df = df[df["outcome"].notna()].copy()
    df["y"] = (df["outcome"] == PERSON).astype(int)
    df["y_dead"] = (df["outcome"] == DEAD).astype(int)
    if len(df) < 5000 or df["y"].sum() < 200:
        sys.exit(f"only {len(df):,} usable calls / {df['y'].sum():,} people: too few to train well; use the starter model")
    df = df.sort_values("created_at")
    cut = int(len(df) * 0.85)  # the newest 15% decides how many trees
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    card = {"calls": int(len(df)), "people": int(df["y"].sum()), "features": FEATURES}
    for target, name in (("y", "person_model.txt"), ("y_dead", "dead_model.txt")):
        tr, va = df.iloc[:cut], df.iloc[cut:]
        dtr = lgb.Dataset(tr[FEATURES], tr[target], categorical_feature=CATEGORICAL, free_raw_data=False)
        dva = lgb.Dataset(va[FEATURES], va[target], categorical_feature=CATEGORICAL, reference=dtr, free_raw_data=False)
        m = lgb.train(PARAMS, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        full = lgb.Dataset(df[FEATURES], df[target], categorical_feature=CATEGORICAL, free_raw_data=False)
        lgb.train(PARAMS, full, int(m.best_iteration * 1.1) + 1).save_model(str(out / name))
        card[name] = {"trees": int(m.best_iteration * 1.1) + 1, "validation": m.best_score.get("valid_0", {})}
    (out / "model_card.json").write_text(json.dumps(card, indent=1, default=float))
    print(f"trained on {len(df):,} calls ({df['y'].mean():.1%} reached a person); models in {out}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="phone-intent", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="score every number in a call log P1-P4")
    s.add_argument("calls"); s.add_argument("--out", default="scores.csv")
    s.add_argument("--models", default=str(STARTER)); s.add_argument("--as-of")
    t = sub.add_parser("train", help="train your own models on a call log")
    t.add_argument("calls"); t.add_argument("--out", required=True)
    a = ap.parse_args()
    (_score if a.cmd == "score" else _train)(a)


if __name__ == "__main__":
    main()
