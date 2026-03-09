#!/usr/bin/env python3
"""Clean LOSO evaluation for expression using explicit-chain IgLM embeddings.

Fixes applied:
- Heavy sequences are embedded with [HEAVY], light with [LIGHT] (no heuristic).
- Leave-one-assay-out over source_file.
- Leakage-safe split by pair_id within source pool.
- Baseline comparison: Dummy mean, Ridge, XGBoost.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from iglm.model.IgLM import CHECKPOINT_DICT, VOCAB_FILE  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_csv", type=Path, default=Path("PDW_GeneralAntibodies/expression_fitness/expression_unified_fitness_minmax_with_source.csv"))
    p.add_argument("--model_name", type=str, default="IgLM", choices=["IgLM", "IgLM-S"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.80, help="Within source pool, fraction for train over train+val")

    p.add_argument("--cache_npz", type=Path, default=Path("PDW_GeneralAntibodies/expression_fitness/iglm_typed_chain_cache.npz"))
    p.add_argument("--cache_map", type=Path, default=Path("PDW_GeneralAntibodies/expression_fitness/iglm_typed_chain_cache_map.json"))
    p.add_argument("--recompute_embeddings", action="store_true")

    p.add_argument("--out_dir", type=Path, default=Path("PDW_GeneralAntibodies/expression_fitness/output_loso_fixed"))
    return p.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"source_file", "heavy", "light", "y"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns: {sorted(miss)}")
    out = df.copy()
    out["source_file"] = out["source_file"].astype(str)
    out["heavy"] = out["heavy"].astype(str).str.strip().str.upper()
    out["light"] = out["light"].astype(str).str.strip().str.upper()
    out["y"] = pd.to_numeric(out["y"], errors="raise")
    if "pair_id" not in out.columns:
        out["pair_id"] = out["heavy"] + "|" + out["light"]
    out = out[(out["heavy"] != "") & (out["light"] != "")].reset_index(drop=True)
    return out


def load_vocab_map(vocab_file: str) -> dict[str, int]:
    m = {}
    with open(vocab_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            t = line.strip()
            if t:
                m[t] = i
    return m


def embed_seq(model: torch.nn.Module, device: torch.device, tok2id: dict[str, int], seq: str, chain_token: str, species_token: str = "[HUMAN]") -> np.ndarray:
    tokens = [chain_token, species_token] + list(seq) + ["[SEP]"]
    unk = tok2id.get("[UNK]", 1)
    ids = [tok2id.get(t, unk) for t in tokens]
    if any(i == unk for i in ids):
        raise ValueError("Unknown token")
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(x, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states[-1][0]
    seq_hs = hs[2:-1]
    return seq_hs.mean(dim=0).cpu().numpy().astype(np.float32)


def compute_or_load_typed_cache(df: pd.DataFrame, args: argparse.Namespace) -> dict[str, np.ndarray]:
    heavy = sorted(df["heavy"].drop_duplicates().tolist())
    light = sorted(df["light"].drop_duplicates().tolist())
    needed_keys = {f"H::{s}" for s in heavy} | {f"L::{s}" for s in light}

    if args.cache_npz.exists() and args.cache_map.exists() and not args.recompute_embeddings:
        npz = np.load(args.cache_npz, allow_pickle=True)
        key_to_typed = json.loads(args.cache_map.read_text())
        typed2emb = {typed: np.asarray(npz[k], dtype=np.float32) for k, typed in key_to_typed.items()}
        if needed_keys.issubset(set(typed2emb.keys())):
            print(f"Loaded typed chain cache: {len(typed2emb)} entries")
            return typed2emb
        print("Typed cache incomplete; recomputing.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = transformers.GPT2LMHeadModel.from_pretrained(CHECKPOINT_DICT[args.model_name]).to(device)
    model.eval()
    tok2id = load_vocab_map(VOCAB_FILE)

    typed2emb: dict[str, np.ndarray] = {}
    total = len(heavy) + len(light)
    done = 0

    for s in heavy:
        key = f"H::{s}"
        try:
            typed2emb[key] = embed_seq(model, device, tok2id, s, "[HEAVY]")
        except Exception:
            pass
        done += 1
        if done % 200 == 0 or done == total:
            print(f"embedded {done}/{total}")

    for s in light:
        key = f"L::{s}"
        try:
            typed2emb[key] = embed_seq(model, device, tok2id, s, "[LIGHT]")
        except Exception:
            pass
        done += 1
        if done % 200 == 0 or done == total:
            print(f"embedded {done}/{total}")

    if not typed2emb:
        raise RuntimeError("No embeddings computed")

    args.cache_npz.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    key_to_typed = {}
    for i, (typed, emb) in enumerate(typed2emb.items()):
        k = f"emb_{i:07d}"
        payload[k] = emb
        key_to_typed[k] = typed
    np.savez_compressed(args.cache_npz, **payload)
    args.cache_map.write_text(json.dumps(key_to_typed))
    print(f"Saved typed cache: {args.cache_npz}")
    return typed2emb


def build_xy(df: pd.DataFrame, typed2emb: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    xs, ys, idx = [], [], []
    for i, r in df.iterrows():
        h = typed2emb.get(f"H::{r['heavy']}")
        l = typed2emb.get(f"L::{r['light']}")
        if h is None or l is None:
            continue
        xs.append(np.concatenate([h, l]).astype(np.float32))
        ys.append(float(r["y"]))
        idx.append(i)
    if not xs:
        raise RuntimeError("No rows featurized")
    used = df.loc[idx].reset_index(drop=True)
    return np.stack(xs), np.asarray(ys, dtype=np.float32), used


def pair_split(source_df: pd.DataFrame, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pair_tbl = source_df.groupby("pair_id", as_index=False)["y"].mean()
    nbins = min(10, max(2, pair_tbl["y"].nunique()))
    pair_tbl["y_bin"] = pd.qcut(pair_tbl["y"], q=nbins, duplicates="drop")
    tr_pairs, va_pairs = train_test_split(
        pair_tbl,
        train_size=train_frac,
        random_state=seed,
        stratify=pair_tbl["y_bin"].astype(str),
    )
    tr_set = set(tr_pairs["pair_id"])
    va_set = set(va_pairs["pair_id"])
    m_tr = source_df["pair_id"].isin(tr_set).to_numpy()
    m_va = source_df["pair_id"].isin(va_set).to_numpy()
    return m_tr, m_va


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def r2(y, p):
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def fit_ridge(X_tr, y_tr, X_va, y_va) -> tuple[Ridge, dict[str, Any]]:
    alphas = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    best = None
    best_rmse = np.inf
    for a in alphas:
        m = Ridge(alpha=a)
        m.fit(X_tr, y_tr)
        p = m.predict(X_va)
        s = rmse(y_va, p)
        if s < best_rmse:
            best_rmse = s
            best = (m, {"alpha": a})
    return best


def fit_xgb(X_tr, y_tr, X_va, y_va, seed: int) -> tuple[XGBRegressor, dict[str, Any]]:
    grid = [
        {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
        {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0},
        {"max_depth": 4, "learning_rate": 0.1, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0},
    ]
    best = None
    best_rmse = np.inf
    for hp in grid:
        m = XGBRegressor(
            n_estimators=2500,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            eval_metric="rmse",
            early_stopping_rounds=80,
            **hp,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        p = m.predict(X_va)
        s = rmse(y_va, p)
        if s < best_rmse:
            best_rmse = s
            best = (m, {**hp, "best_iteration": int(getattr(m, "best_iteration", m.n_estimators))})
    return best


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data_csv)
    typed2emb = compute_or_load_typed_cache(df, args)
    X, y, used = build_xy(df, typed2emb)

    assays = sorted(used["source_file"].unique().tolist())
    rows = []

    for i, holdout in enumerate(assays, start=1):
        test_df = used[used["source_file"].eq(holdout)].reset_index(drop=True)
        src_df = used[~used["source_file"].eq(holdout)].reset_index(drop=True)

        m_tr, m_va = pair_split(src_df, train_frac=args.train_frac, seed=args.seed)

        X_src, y_src, src_used = build_xy(src_df, typed2emb)
        X_test, y_test, _ = build_xy(test_df, typed2emb)

        X_tr, y_tr = X_src[m_tr], y_src[m_tr]
        X_va, y_va = X_src[m_va], y_src[m_va]

        # Dummy baseline
        mean_pred = float(y_tr.mean())
        p_dummy = np.full_like(y_test, mean_pred, dtype=np.float32)

        # Ridge baseline
        ridge_model, ridge_hp = fit_ridge(X_tr, y_tr, X_va, y_va)
        p_ridge = ridge_model.predict(X_test)

        # XGBoost model
        xgb_model, xgb_hp = fit_xgb(X_tr, y_tr, X_va, y_va, seed=args.seed)
        n_trees = max(100, int(xgb_hp["best_iteration"]) + 1)
        xgb_refit = XGBRegressor(
            n_estimators=n_trees,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=args.seed,
            eval_metric="rmse",
            max_depth=int(xgb_hp["max_depth"]),
            learning_rate=float(xgb_hp["learning_rate"]),
            subsample=float(xgb_hp["subsample"]),
            colsample_bytree=float(xgb_hp["colsample_bytree"]),
            reg_lambda=float(xgb_hp["reg_lambda"]),
        )
        X_trva = np.concatenate([X_tr, X_va], axis=0)
        y_trva = np.concatenate([y_tr, y_va], axis=0)
        xgb_refit.fit(X_trva, y_trva, verbose=False)
        p_xgb = xgb_refit.predict(X_test)

        rows.extend([
            {
                "holdout_assay": holdout,
                "model": "dummy_mean",
                "n_train": int(len(y_tr)),
                "n_val": int(len(y_va)),
                "n_test": int(len(y_test)),
                "mae": mae(y_test, p_dummy),
                "rmse": rmse(y_test, p_dummy),
                "r2": r2(y_test, p_dummy),
                "hyperparams": "{}",
            },
            {
                "holdout_assay": holdout,
                "model": "ridge",
                "n_train": int(len(y_tr)),
                "n_val": int(len(y_va)),
                "n_test": int(len(y_test)),
                "mae": mae(y_test, p_ridge),
                "rmse": rmse(y_test, p_ridge),
                "r2": r2(y_test, p_ridge),
                "hyperparams": json.dumps(ridge_hp),
            },
            {
                "holdout_assay": holdout,
                "model": "xgboost",
                "n_train": int(len(y_tr)),
                "n_val": int(len(y_va)),
                "n_test": int(len(y_test)),
                "mae": mae(y_test, p_xgb),
                "rmse": rmse(y_test, p_xgb),
                "r2": r2(y_test, p_xgb),
                "hyperparams": json.dumps({**xgb_hp, "n_trees_refit": n_trees}),
            },
        ])
        print(f"[{i}/{len(assays)}] {holdout}: xgb R2={r2(y_test,p_xgb):.4f}")

    out = pd.DataFrame(rows)
    out_path = args.out_dir / "loso_results_expression_iglm_fixed.csv"
    out.to_csv(out_path, index=False)

    summary = out.groupby("model").agg(
        mean_mae=("mae", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_r2=("r2", "mean"),
        std_r2=("r2", "std"),
    ).reset_index()
    summary_path = args.out_dir / "loso_summary_expression_iglm_fixed.csv"
    summary.to_csv(summary_path, index=False)

    print("\nSaved:")
    print(out_path)
    print(summary_path)
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
