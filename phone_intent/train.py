"""Train and evaluate the phone intent model ("will a real person answer this dial?") on a CallCloud-format dial
export (the columns of label_callcloud). For your own CSV, use `phone-intent train` instead (phone_intent/cli.py).

  python -m phone_intent.train --dials data/dials.csv --out models/

Evaluation, all on dials the model never trained on:
  newest weeks      trained on everything before the last --test-days, tested on those days (the deployment case)
  unseen customers  a share of whole customers (workspaces) held out of training (the public-user case)
  first dials       the subset where a number has never been dialed before (no history to lean on)
Reported per split: base rate, ROC AUC, PR AUC, and the share of people found in the top 10% / 20% of scores (lift).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import features as F
from .features import CATEGORICAL, FEATURES, build_features
from .labels import DEAD, PERSON, label_callcloud
from .tiers import best_hour_scores, tier

PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=7,
              # area code / state are wide categories: smooth them hard so the model learns patterns, not one customer's list
              cat_smooth=200, min_data_per_group=1000, cat_l2=50, max_cat_threshold=16)


TRANSCRIPT_LABELS = None  # optional: a JSONL of {"id": dial id, "label": person|voicemail|screener|ivr|silence|unclear}


def load_transcript_labels(path: str | None = TRANSCRIPT_LABELS) -> dict[str, str]:
    """The voicemail detector's nightly verdicts per dial id (CallCloud only); {} when the store is not there."""
    if not path or not Path(path).exists():
        return {}
    out = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            try:
                j = json.loads(line)
                out[j["id"]] = j["label"]
            except (ValueError, KeyError):
                continue
    return out


def load_callcloud(path: str, carriers: str | None = "data/carriers.csv", transcripts: str | None = TRANSCRIPT_LABELS) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    line_s = pd.to_numeric(raw["line_s"], errors="coerce")
    tx = load_transcript_labels(transcripts) if "id" in raw else {}
    tl = raw["id"].map(tx) if tx else pd.Series([None] * len(raw))
    outcome = [label_callcloud(d, f, a, l, t) for d, f, a, l, t in zip(raw["disposition"], raw["failure_reason"], raw["ai"], line_s, tl)]
    df = pd.DataFrame({
        "created_at": raw["created_at"], "workspace": raw["workspace"], "number": raw["number"],
        "caller_id": raw["caller_id"], "line_type": raw["label"], "outcome": outcome,
    })
    if carriers and Path(carriers).exists():
        car = pd.read_csv(carriers, dtype=str, keep_default_na=False)
        car = car[car["found"] == "1"][["number", "carrier_family", "lrn_line_type", "ported"]]
        df = df.merge(car, on="number", how="left")
    return df


def report(name: str, y: np.ndarray, p: np.ndarray, ws: np.ndarray | None = None) -> dict:
    if len(y) == 0 or y.min() == y.max():
        return {"split": name, "n": int(len(y))}
    order = np.argsort(-p)
    found = lambda q: float(y[order[: max(1, int(len(y) * q))]].sum() / y.sum())  # noqa: E731
    r = {"split": name, "n": int(len(y)), "people": int(y.sum()), "base_rate": round(float(y.mean()), 4),
         "auc": round(float(roc_auc_score(y, p)), 4), "pr_auc": round(float(average_precision_score(y, p)), 4),
         "people_in_top10pct": round(found(0.10), 3), "people_in_top20pct": round(found(0.20), 3)}
    if ws is not None:
        aucs, wts = [], []
        for w in np.unique(ws):
            m = ws == w
            if y[m].sum() >= 20 and (1 - y[m]).sum() >= 20:
                aucs.append(roc_auc_score(y[m], p[m])); wts.append(m.sum())
        if aucs:
            r["within_customer_auc"] = round(float(np.average(aucs, weights=wts)), 4)
            r["customers"] = len(aucs)
    print(json.dumps(r))
    return r


def fit(train: pd.DataFrame, valid: pd.DataFrame, target: str = "y") -> lgb.Booster:
    dtr = lgb.Dataset(train[FEATURES], train[target], categorical_feature=CATEGORICAL, free_raw_data=False)
    dva = lgb.Dataset(valid[FEATURES], valid[target], categorical_feature=CATEGORICAL, reference=dtr, free_raw_data=False)
    return lgb.train(PARAMS, dtr, num_boost_round=2000, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dials", default="data/dials.csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--test-days", type=int, default=14)
    ap.add_argument("--holdout-customers", type=float, default=0.2)
    ap.add_argument("--final", action="store_true", help="after evaluating, train the shipped models on ALL the data")
    ap.add_argument("--carrier", action="store_true", help="also use carrier lookups (carrier_family, lrn_line_type, ported)")
    a = ap.parse_args()
    global FEATURES, CATEGORICAL
    if a.carrier:
        FEATURES = F.FEATURES + F.CARRIER_FEATURES
        CATEGORICAL = F.CATEGORICAL + F.CARRIER_CATEGORICAL

    df = build_features(load_callcloud(a.dials))
    df = df[df["outcome"].notna()].copy()
    df["y"] = (df["outcome"] == PERSON).astype(int)
    df["y_dead"] = (df["outcome"] == DEAD).astype(int)
    print(f"{len(df):,} informative dials, {df['y'].sum():,} people ({df['y'].mean():.2%}), {df['number'].nunique():,} numbers, "
          f"{df['workspace'].nunique()} customers")

    cut = df["created_at"].max() - pd.Timedelta(days=a.test_days)
    rng = np.random.default_rng(7)
    ws = np.array(sorted(df["workspace"].unique()))  # fixed order: the same customers are held out every run
    held = set(rng.choice(ws, size=max(1, int(len(ws) * a.holdout_customers)), replace=False))
    is_held = df["workspace"].isin(held)
    old, new = df["created_at"] < cut, df["created_at"] >= cut

    train = df[old & ~is_held]
    # early stopping on the newest slice of the training window, never on a test split
    vcut = train["created_at"].max() - pd.Timedelta(days=7)
    model = fit(train[train["created_at"] < vcut], train[train["created_at"] >= vcut])
    print(f"trees: {model.best_iteration}")

    results = []
    for name, part in (("newest weeks, known customers", df[new & ~is_held]),
                       ("unseen customers (all dates)", df[is_held]),
                       ("unseen customers, first dial of a number", df[is_held & (df["prior_tries"] == 0)]),
                       ("newest weeks, first dial of a number", df[new & ~is_held & (df["prior_tries"] == 0)]),
                       ("newest weeks, numbers with history", df[new & ~is_held & (df["prior_tries"] > 0)]),
                       ("unseen customers, numbers with history", df[is_held & (df["prior_tries"] > 0)]),
                       ("newest weeks, number has a carrier lookup", df[new & ~is_held & (df["carrier_family"] != "")]),
                       ("newest weeks, first dial WITH a carrier lookup", df[new & ~is_held & (df["carrier_family"] != "") & (df["prior_tries"] == 0)])):
        p = model.predict(part[FEATURES], num_iteration=model.best_iteration)
        results.append(report(name, part["y"].to_numpy(), p, part["workspace"].to_numpy()))

    imp = pd.Series(model.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    print("feature importance (gain share):")
    print((imp / imp.sum()).round(3).to_string())

    # P1-P4: the dead-number model, then each test dial's number tiered at its best hour
    dead_model = fit(train[train["created_at"] < vcut], train[train["created_at"] >= vcut], "y_dead")
    avg = float(train["y"].mean())
    tier_report = {}
    for name, part in (("newest weeks, known customers", df[new & ~is_held]), ("unseen customers", df[is_held]),
                       ("newest weeks, first dial", df[new & ~is_held & (df["prior_tries"] == 0)])):
        pp, hour = best_hour_scores(model, part, FEATURES)
        pdead = dead_model.predict(part[FEATURES])
        t = tier(pp, pdead, part["workspace"].to_numpy())
        g = pd.DataFrame({"tier": t, "y": part["y"].to_numpy(), "dead": part["y_dead"].to_numpy()}).groupby("tier")
        rows = g.agg(share=("y", "size"), person_rate=("y", "mean"), dead_rate=("dead", "mean"))
        rows["share"] = rows["share"] / len(part)
        rows["lift_vs_list"] = rows["person_rate"] / part["y"].mean()
        print(f"\nP-tiers: {name} (list average reaches a person {part['y'].mean():.1%})")
        print(rows.round(3).to_string())
        tier_report[name] = rows.round(4).reset_index().to_dict(orient="records")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps({"results": results, "tiers": tier_report, "importance": (imp / imp.sum()).round(4).to_dict(),
                                               "held_out_customers": len(held)}, indent=1))
    model.save_model(str(out / "eval_model.txt"), num_iteration=model.best_iteration)
    dead_model.save_model(str(out / "eval_dead_model.txt"), num_iteration=dead_model.best_iteration)
    if a.final:
        # the shipped models: every dial, the tree counts early stopping chose (plus 10% for the extra data)
        full = lgb.Dataset(df[FEATURES], df["y"], categorical_feature=CATEGORICAL, free_raw_data=False)
        person = lgb.train(PARAMS, full, num_boost_round=int(model.best_iteration * 1.1) + 1)
        dfull = lgb.Dataset(df[FEATURES], df["y_dead"], categorical_feature=CATEGORICAL, free_raw_data=False)
        dead = lgb.train(PARAMS, dfull, num_boost_round=int(dead_model.best_iteration * 1.1) + 1)
        person.save_model(str(out / "person_model.txt")); dead.save_model(str(out / "dead_model.txt"))
        (out / "model_card.json").write_text(json.dumps({
            "target": "a real person answers the dial", "dials": int(len(df)), "people": int(df["y"].sum()),
            "features": FEATURES, "trained_through": str(df["created_at"].max()), "evaluation": results, "tiers": tier_report}, indent=1))
        print("shipped models written:", out / "person_model.txt", out / "dead_model.txt")


if __name__ == "__main__":
    main()
