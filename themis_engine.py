from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

EBAR_DEFAULT = 6.6

@dataclass
class EngineConfig:
    ebar: float = EBAR_DEFAULT
    c_min: float = 0.01
    c_max: float = 1.0
    c_steps: int = 100
    t_min: float = 0.0
    t_max: float = 1.0
    t_steps: int = 101
    t_cap: Optional[float] = None


def normalise_actor_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        "idx", "e", "pop_m", "gdp_cap", "headline_price", "effective_price",
        "alpha_base", "alpha_cov", "alpha_trf", "weight", "alpha_base_low",
        "alpha_base_high", "alpha_cov_low", "alpha_cov_high", "k_low", "k_high",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    if "weight" not in out.columns or out["weight"].sum() <= 0:
        out["weight"] = out["pop_m"] * out["e"]
    wsum = out["weight"].sum()
    if wsum > 0:
        out["weight"] = out["weight"] / wsum
    out["role"] = np.where(out["e"] > EBAR_DEFAULT, "Contributor", "Beneficiary")
    return out


def arrays(actor_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    df = normalise_actor_df(actor_df)
    return {
        "names": df["name"].astype(str).to_numpy(),
        "e": df["e"].astype(float).to_numpy(),
        "pop": df["pop_m"].astype(float).to_numpy(),
        "gdp_cap": df["gdp_cap"].astype(float).to_numpy(),
        "alpha_base": df["alpha_base"].astype(float).to_numpy(),
        "alpha_cov": df["alpha_cov"].astype(float).to_numpy(),
        "alpha_trf": df["alpha_trf"].astype(float).to_numpy(),
        "weights": df["weight"].astype(float).to_numpy(),
    }


def solve_tminus(e: np.ndarray, pop: np.ndarray, tplus: float, ebar: float = EBAR_DEFAULT,
                 active: Optional[np.ndarray] = None) -> float:
    if active is None:
        active = np.ones_like(e, dtype=bool)
    excess = np.maximum(e - ebar, 0.0) * pop
    deficit = np.maximum(ebar - e, 0.0) * pop
    denom = float(deficit[active].sum())
    if denom <= 1e-12:
        return 0.0
    return float(tplus * excess[active].sum() / denom)


def preference_values(e: np.ndarray, alpha_base: np.ndarray, alpha_cov: np.ndarray, alpha_trf: np.ndarray,
                      c: float, tplus: float, tminus: float, ebar: float = EBAR_DEFAULT) -> np.ndarray:
    transfer_effect = np.where(e > ebar, -tplus * (e - ebar), tminus * (ebar - e))
    vals = alpha_base + alpha_cov * c + alpha_trf * transfer_effect
    return np.maximum(vals, 0.0)


def price_for_coverage(preferences: np.ndarray, weights: np.ndarray, c_target: float) -> float:
    order = np.argsort(-preferences)
    cum = 0.0
    for idx in order:
        cum += float(weights[idx])
        if cum + 1e-12 >= c_target:
            return float(preferences[idx])
    return float(np.min(preferences))


def transfer_accounting(actor_df: pd.DataFrame, p: float, tplus: float, tminus: float,
                        ebar: float = EBAR_DEFAULT, join_flags: Optional[np.ndarray] = None) -> pd.DataFrame:
    df = normalise_actor_df(actor_df).copy()
    if join_flags is None:
        join_flags = np.ones(len(df), dtype=bool)
    df["joins"] = join_flags
    df["gap"] = ebar - df["e"]
    df["status"] = np.where(df["e"] > ebar, "Contributor", "Beneficiary")
    df["collected_per_cap"] = p * df["e"]
    df["sent_per_cap"] = np.where(df["e"] > ebar, tplus * p * (df["e"] - ebar), 0.0)
    df["received_per_cap"] = np.where(df["e"] <= ebar, tminus * p * (ebar - df["e"]), 0.0)
    df["net_transfer_per_cap"] = df["received_per_cap"] - df["sent_per_cap"]
    df["retained_domestic_per_cap"] = p * df["e"] - (np.where(df["e"] > ebar, tplus * p * (df["e"] - ebar), 0.0))
    df["total_sent_mEUR"] = np.where(df["joins"], df["sent_per_cap"] * df["pop_m"], 0.0)
    df["total_received_mEUR"] = np.where(df["joins"], df["received_per_cap"] * df["pop_m"], 0.0)
    df["net_total_mEUR"] = df["total_received_mEUR"] - df["total_sent_mEUR"]
    return df


def run_mechanism(actor_df: pd.DataFrame, config: Optional[EngineConfig] = None,
                  params: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
    if config is None:
        config = EngineConfig()
    df = normalise_actor_df(actor_df)
    arr = arrays(df)
    e, pop, weights, names = arr["e"], arr["pop"], arr["weights"], arr["names"]
    alpha_base = arr["alpha_base"] if params is None else params.get("alpha_base", arr["alpha_base"])
    alpha_cov = arr["alpha_cov"] if params is None else params.get("alpha_cov", arr["alpha_cov"])
    alpha_trf = arr["alpha_trf"] if params is None else params.get("alpha_trf", arr["alpha_trf"])
    tmax = config.t_max if config.t_cap is None else min(config.t_max, config.t_cap)
    c_grid = np.round(np.linspace(config.c_min, config.c_max, config.c_steps), 4)
    t_grid = np.round(np.linspace(config.t_min, tmax, config.t_steps), 4)
    records: List[Tuple[float,float,float,float,float,np.ndarray]] = []
    best = None
    for tplus in t_grid:
        tminus_expected = solve_tminus(e, pop, float(tplus), ebar=config.ebar)
        for c in c_grid:
            prefs = preference_values(e, alpha_base, alpha_cov, alpha_trf, float(c), float(tplus), tminus_expected, ebar=config.ebar)
            price = price_for_coverage(prefs, weights, float(c))
            objective = float(c) * price
            rec = (objective, float(c), float(price), float(tplus), float(tminus_expected), prefs)
            records.append(rec)
            if best is None or objective > best[0] + 1e-12 or (abs(objective - best[0]) <= 1e-12 and c > best[1]):
                best = rec
    objective, c_star, p_star, tplus_star, tminus_expected, prefs_star = best
    join = prefs_star + 1e-9 >= p_star
    actual_coverage = float(weights[join].sum())
    tminus_actual = solve_tminus(e, pop, tplus_star, ebar=config.ebar, active=join)
    accounting = transfer_accounting(df, p_star, tplus_star, tminus_actual, ebar=config.ebar, join_flags=join)
    # curves for selected T+
    selected_curve_rows = []
    for c in c_grid:
        prefs = preference_values(e, alpha_base, alpha_cov, alpha_trf, float(c), tplus_star, tminus_actual, ebar=config.ebar)
        price = price_for_coverage(prefs, weights, float(c))
        selected_curve_rows.append({"coverage": float(c), "feasible_price": float(price), "objective": float(c)*float(price)})
    curve_df = pd.DataFrame(selected_curve_rows)
    # best frontier by T+
    frontier = []
    for t in sorted(set(r[3] for r in records)):
        subset = [r for r in records if r[3] == t]
        r = max(subset, key=lambda z: z[0])
        frontier.append({"Tplus": r[3], "Tminus_expected": r[4], "c": r[1], "p": r[2], "objective": r[0]})
    actor_results = df.copy()
    actor_results["preference_at_solution"] = prefs_star
    actor_results["joins"] = join
    actor_results["join_status"] = np.where(join, "Joiner", "Non-joiner")
    return {
        "p_star": float(p_star), "c_star": float(c_star), "Tplus_star": float(tplus_star),
        "Tminus_expected": float(tminus_expected), "Tminus_actual": float(tminus_actual),
        "objective": float(objective), "actual_coverage": actual_coverage,
        "actor_results": actor_results, "accounting": accounting, "curve": curve_df,
        "frontier_by_Tplus": pd.DataFrame(frontier), "preferences": prefs_star,
        "join_flags": join, "config": config,
    }


def diagnostics(actor_df: pd.DataFrame, res: Dict[str, Any], tolerance: float = 1e-5) -> pd.DataFrame:
    df = normalise_actor_df(actor_df)
    ar = res["actor_results"]
    acc = res["accounting"]
    join = res["join_flags"]
    p = res["p_star"]
    c = res["c_star"]
    weights = df["weight"].to_numpy(float)
    prefs = ar["preference_at_solution"].to_numpy(float)
    actual_coverage = float(weights[join].sum())
    rows = []
    def add(name, passed, detail): rows.append({"Check": name, "Status": "PASS" if passed else "FAIL", "Detail": detail})
    add("All joiners accept selected price", bool(np.all(prefs[join] + 1e-9 >= p)), f"minimum joiner willingness = {prefs[join].min() if join.any() else np.nan:.2f}, p* = {p:.2f}")
    add("All non-joiners below selected price", bool(np.all(prefs[~join] < p + 1e-8)), f"maximum non-joiner willingness = {prefs[~join].max() if (~join).any() else np.nan:.2f}, p* = {p:.2f}")
    add("Actual coverage meets target", actual_coverage + 1e-9 >= c, f"actual = {actual_coverage:.4f}, target = {c:.4f}")
    balance = float(acc["total_sent_mEUR"].sum() - acc["total_received_mEUR"].sum())
    add("Transfer pool balances", abs(balance) < max(1e-4, tolerance*max(1, abs(float(acc["total_sent_mEUR"].sum())))), f"balance = {balance:.6f} mEUR")
    add("Emissions weights sum to 1", abs(float(df["weight"].sum()) - 1.0) < 1e-6, f"sum = {float(df['weight'].sum()):.6f}")
    add("No missing alpha parameters", not df[["alpha_base", "alpha_cov", "alpha_trf"]].isna().any().any(), "alpha_base, alpha_cov, alpha_trf present")
    add("Fixed world-average benchmark", abs(res["config"].ebar - EBAR_DEFAULT) < 1e-12, f"ē = {res['config'].ebar}")
    return pd.DataFrame(rows)


def draw_params(actor_df: pd.DataFrame, rng: np.random.Generator, k_base: float = 20000.0) -> Dict[str, np.ndarray]:
    df = normalise_actor_df(actor_df)
    def tri(low, mode, high):
        low = np.minimum(low, mode); high = np.maximum(high, mode)
        out = np.empty_like(mode, dtype=float)
        for i in range(len(mode)):
            if abs(high[i] - low[i]) < 1e-12: out[i] = mode[i]
            else: out[i] = rng.triangular(low[i], mode[i], high[i])
        return out
    base = tri(df["alpha_base_low"].to_numpy(float), df["alpha_base"].to_numpy(float), df["alpha_base_high"].to_numpy(float))
    cov = tri(df["alpha_cov_low"].to_numpy(float), df["alpha_cov"].to_numpy(float), df["alpha_cov_high"].to_numpy(float))
    k_low = float(df["k_low"].replace(0, np.nan).min()) if "k_low" in df else 10000.0
    k_high = float(df["k_high"].replace(0, np.nan).max()) if "k_high" in df else 30000.0
    if not np.isfinite(k_low): k_low = 10000.0
    if not np.isfinite(k_high): k_high = 30000.0
    k = float(rng.triangular(k_low, k_base, k_high))
    trf = np.minimum(20.0, k / np.maximum(df["gdp_cap"].to_numpy(float), 1.0))
    return {"alpha_base": base, "alpha_cov": cov, "alpha_trf": trf, "k": np.array([k])}


def run_monte_carlo(actor_df: pd.DataFrame, n: int = 500, seed: int = 42,
                    config: Optional[EngineConfig] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if config is None: config = EngineConfig(c_steps=60, t_steps=61)
    rng = np.random.default_rng(seed)
    rows = []
    names = normalise_actor_df(actor_df)["name"].astype(str).tolist()
    for i in range(n):
        params = draw_params(actor_df, rng)
        res = run_mechanism(actor_df, config=config, params=params)
        row = {
            "run": i+1, "p_star": res["p_star"], "c_star": res["c_star"], "Tplus_star": res["Tplus_star"],
            "Tminus_actual": res["Tminus_actual"], "objective": res["objective"], "actual_coverage": res["actual_coverage"],
        }
        ar = res["actor_results"]
        for nm, j in zip(ar["name"], ar["joins"]):
            row[f"joins__{nm}"] = bool(j)
        rows.append(row)
    mc = pd.DataFrame(rows)
    join_rows = []
    for nm in names:
        col = f"joins__{nm}"
        if col in mc:
            join_rows.append({"Actor": nm, "Join probability": float(mc[col].mean())})
    return mc, pd.DataFrame(join_rows)


def split_country_from_group(actor_df: pd.DataFrame, country_df: pd.DataFrame, country_name: str) -> pd.DataFrame:
    """Split a country from its existing actor group.
    Creates a new individual actor with per-country alpha values from country_data.csv.
    Handles residual groups from previous splits.
    """
    actors = normalise_actor_df(actor_df).copy()
    countries = country_df.copy()
    match = countries[countries["country"].astype(str).str.lower() == country_name.lower()]
    if match.empty:
        return actors
    ctry = match.iloc[0]
    parent_group = str(ctry.get("actor_group", ""))
    parent_idx = ctry.get("actor_idx")
    # Find parent: try exact idx first, then name match (including residual groups)
    parent = pd.DataFrame()
    if pd.notna(parent_idx):
        parent = actors[actors["idx"].astype(float) == float(parent_idx)]
    if parent.empty:
        # Try matching group name including "residual" variants
        mask = actors["name"].astype(str).str.contains(parent_group, case=False, na=False)
        parent = actors[mask]
    if parent.empty:
        return actors
    # Check country isn't already split out
    if actors["name"].astype(str).str.upper().str.contains(country_name.upper(), na=False).any():
        return actors
    p = parent.iloc[0].copy()
    pop = float(ctry["population_m"]); e = float(ctry["emissions_cap"]); gdp = float(ctry["gdp_cap"])
    total_emissions = p["pop_m"] * p["e"]
    remaining_pop = max(float(p["pop_m"]) - pop, 0.0)
    remaining_emissions = max(total_emissions - pop * e, 0.0)
    if remaining_pop > 1e-9:
        actors.loc[parent.index[0], "pop_m"] = remaining_pop
        actors.loc[parent.index[0], "e"] = remaining_emissions / remaining_pop
        actors.loc[parent.index[0], "gdp_cap"] = max(float(p["gdp_cap"]), 1.0)  # keep parent calibration simple
        actors.loc[parent.index[0], "name"] = f"{p['name']} residual"
    else:
        actors = actors.drop(parent.index[0])
    new = p.copy()
    new["idx"] = float(actors["idx"].max()) + 1
    new["name"] = str(ctry["country"]).upper()
    new["e"] = e; new["pop_m"] = pop; new["gdp_cap"] = gdp
    new["headline_price"] = float(ctry.get("headline_price", 0) or 0)
    # Use per-country alpha values if available, otherwise inherit parent
    if "alpha_base_own" in ctry.index and pd.notna(ctry.get("alpha_base_own")) and float(ctry["alpha_base_own"]) != 0:
        new["alpha_base"] = float(ctry["alpha_base_own"])
    elif "alpha_base_own" in ctry.index and float(ctry.get("alpha_base_own", -1)) == 0:
        new["alpha_base"] = 0.0
    if "alpha_cov_own" in ctry.index and pd.notna(ctry.get("alpha_cov_own")) and float(ctry["alpha_cov_own"]) != 0:
        new["alpha_cov"] = float(ctry["alpha_cov_own"])
    elif "alpha_cov_own" in ctry.index and float(ctry.get("alpha_cov_own", -1)) == 0:
        new["alpha_cov"] = 0.0
    new["alpha_trf"] = min(20.0, 20000.0 / max(gdp, 1.0))
    new["narrative"] = f"Split from {p['name']}. Uses country-specific α_base={new['alpha_base']:.1f}, α_cov={new['alpha_cov']:.0f} from Data Bible research."
    actors = pd.concat([actors, pd.DataFrame([new])], ignore_index=True)
    actors["weight"] = actors["pop_m"] * actors["e"]
    return normalise_actor_df(actors)
