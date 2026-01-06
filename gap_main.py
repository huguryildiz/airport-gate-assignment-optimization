import os
import datetime as dt
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# Local modules
import historic_analysis as hist_an
import optimized_analysis as opt_an

# ======================================================
# Config
# ======================================================

EXCEL_PATH = "flight_data.xlsx"
SHEET_ONE_DAY = "one_day_departures"
SHEET_HIST = "historic_gate_allocation"

BUFFER_MIN = 30     # turnaround buffer (minutes)
SLOT_MIN = 15       # time slot length (minutes)

CAP_ALPHA = 1.20    # historic-based capacity multiplier
LAMBDA_SPREAD = 0.50  # weight for load spread term

# Output folders
CSV_DIR = "csv_outputs"
PLOT_DIR = "plot_outputs"
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ======================================================
# Helpers
# ======================================================

def log(msg: str, icon="✨"):
    print(f"\n{icon} {msg}\n")


def to_minutes(series: pd.Series) -> pd.Series:
    """
    Convert time-like values to minutes-from-midnight.
    Accepts HH:MM strings, datetime.time, pandas datetime, integers.
    """
    if np.issubdtype(series.dtype, np.datetime64):
        s = series.dt
        return (s.hour * 60 + s.minute).astype(int)

    first_valid = next((v for v in series if pd.notna(v)), None)
    if isinstance(first_valid, dt.time):
        return series.apply(lambda t: t.hour * 60 + t.minute if pd.notna(t) else 0).astype(int)

    sdt = pd.to_datetime(series, errors="coerce")
    if sdt.notna().any():
        return (sdt.dt.hour * 60 + sdt.dt.minute).astype(int)

    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


# ======================================================
# Load data
# ======================================================

def load_data(excel_path: str):
    xls = pd.ExcelFile(excel_path)
    dep = pd.read_excel(xls, SHEET_ONE_DAY)
    hist = pd.read_excel(xls, SHEET_HIST)

    # One-day required columns (from your GitHub-safe schema)
    # - flight_code, boardingtime, departure_time, passengers, aircraft_wing_span
    required_dep = ["flight_code", "boardingtime", "departure_time", "passengers", "aircraft_wing_span"]
    for c in required_dep:
        if c not in dep.columns:
            raise KeyError(f"Missing column '{c}' in sheet '{SHEET_ONE_DAY}'. Found: {list(dep.columns)}")

    # Historic required columns
    # - departure_gate, passengers, boarding_time OR flight_time (for hourly)
    required_hist = ["departure_gate", "passengers"]
    for c in required_hist:
        if c not in hist.columns:
            raise KeyError(f"Missing column '{c}' in sheet '{SHEET_HIST}'. Found: {list(hist.columns)}")

    return dep, hist


# ======================================================
# Build model inputs
# ======================================================

def build_inputs(dep: pd.DataFrame, hist: pd.DataFrame):
    dep = dep.copy()
    hist = hist.copy()

    # Normalize times
    dep["B_min"] = to_minutes(dep["boardingtime"])
    dep["D_min"] = to_minutes(dep["departure_time"])

    # Basic flight dictionaries
    F = dep["flight_code"].astype(str).tolist()
    P_f = dep.set_index("flight_code")["passengers"].astype(int).to_dict()
    L_f = dep.set_index("flight_code")["aircraft_wing_span"].astype(float).to_dict()
    B_f = dep.set_index("flight_code")["B_min"].astype(int).to_dict()
    D_f = dep.set_index("flight_code")["D_min"].astype(int).to_dict()

    # Hour-of-day weight (based on today passenger pressure at boarding hour)
    dep["Hour"] = (dep["B_min"] // 60).astype(int)
    hour_pax_today = dep.groupby("Hour")["passengers"].sum()
    max_hour = float(hour_pax_today.max()) if len(hour_pax_today) else 1.0
    w_t = {int(h): float(hour_pax_today.loc[h] / max_hour) for h in hour_pax_today.index}
    H_f = dep.set_index("flight_code")["Hour"].astype(int).to_dict()

    # Gate list from historic
    G = sorted(hist["departure_gate"].dropna().astype(int).unique().tolist())

    total_today_pax = int(dep["passengers"].sum())

    # Historic-based capacity K_g
    hist_gate_pax = (
        hist[["departure_gate", "passengers"]]
        .dropna()
        .groupby("departure_gate")["passengers"]
        .sum()
        .reindex(G)
        .fillna(0.0)
    )

    total_hist_pax = float(hist_gate_pax.sum())
    if total_hist_pax <= 0:
        share = pd.Series([1.0 / len(G)] * len(G), index=G)
    else:
        raw_share = hist_gate_pax / total_hist_pax
        base_share = 0.5 / len(G)
        share = raw_share.clip(lower=base_share)
        share = share / share.sum()

    cap_g = {int(g): float(CAP_ALPHA * share.loc[g] * total_today_pax) for g in G}

    return dep, hist, F, G, P_f, L_f, B_f, D_f, H_f, w_t, cap_g, total_today_pax


# ======================================================
# Time-slot overlap matrix
# ======================================================

def build_overlap(F, B_f, D_f, slot_min: int, buffer_min: int):
    T = list(range(0, 24 * 60, slot_min))
    A_ft = {}

    for f in F:
        start = int(B_f[f])
        end = int(D_f[f]) + buffer_min
        for t in T:
            slot_start = t
            slot_end = t + slot_min
            # Overlap check: intervals [start, end) and [slot_start, slot_end)
            A_ft[(f, t)] = 0 if (end <= slot_start or start >= slot_end) else 1

    return T, A_ft


# ======================================================
# Solve model
# ======================================================

def solve_gap(dep, hist):
    dep, hist, F, G, P_f, L_f, B_f, D_f, H_f, w_t, cap_g, total_today_pax = build_inputs(dep, hist)

    # Two modes: 0=contact, 1=coached
    MODES = [0, 1]

    # Discretization
    T, A_ft = build_overlap(F, B_f, D_f, SLOT_MIN, BUFFER_MIN)
    log(f"Flights: {len(F)} | Gates: {len(G)} | Time slots: {len(T)}", "🧱")

    # --- Model ---
    m = gp.Model("GAP_main")
    x = m.addVars(F, G, MODES, vtype=GRB.BINARY, name="x")

    # Each flight assigned to exactly one (gate, mode)
    for f in F:
        m.addConstr(gp.quicksum(x[f, g, c] for g in G for c in MODES) == 1, name=f"Assign_{f}")

    # Non-overlap per gate & time slot (both modes consume gate resource)
    for g in G:
        for t in T:
            m.addConstr(
                gp.quicksum(A_ft[(f, t)] * gp.quicksum(x[f, g, c] for c in MODES) for f in F) <= 1,
                name=f"NonOverlap_g{g}_t{t}",
            )

    # Gate passenger load and capacity
    z = m.addVars(G, lb=0.0, vtype=GRB.CONTINUOUS, name="z")
    for g in G:
        m.addConstr(
            z[g] == gp.quicksum(P_f[f] * gp.quicksum(x[f, g, c] for c in MODES) for f in F),
            name=f"GateLoad_{g}",
        )
        m.addConstr(z[g] <= cap_g[g], name=f"GateCap_{g}")

    # Load spread
    z_max = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="z_max")
    z_min = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="z_min")
    for g in G:
        m.addConstr(z[g] <= z_max, name=f"Zmax_{g}")
        m.addConstr(z[g] >= z_min, name=f"Zmin_{g}")

    # Objective: minimize coached flights (hour-weighted) + load spread
    obj_psl = gp.quicksum(w_t.get(H_f[f], 1.0) * x[f, g, 1] for f in F for g in G)
    spread = z_max - z_min
    norm_spread = spread / total_today_pax if total_today_pax > 0 else spread

    m.setObjective(obj_psl + LAMBDA_SPREAD * norm_spread, GRB.MINIMIZE)

    # Solver settings (keep simple)
    m.Params.MIPGap = 0.10
    m.Params.TimeLimit = 120
    m.Params.MIPFocus = 1

    log("Optimizing...", "🚀")
    m.optimize()

    if m.SolCount == 0 or m.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        raise RuntimeError(f"No feasible solution. Gurobi status={m.Status}")

    # Extract solution
    assigned_gate = {}
    mode_used = {}
    for f in F:
        for g in G:
            for c in MODES:
                if x[f, g, c].X > 0.5:
                    assigned_gate[f] = g
                    mode_used[f] = c

    dep_out = dep.copy()
    dep_out["assigned_gate"] = dep_out["flight_code"].map(assigned_gate)
    dep_out["is_coached_model"] = dep_out["flight_code"].map(mode_used)

    # PSL (upper bound) = 1 - coached_share
    coach_ct = int((dep_out["is_coached_model"] == 1).sum())
    psl = 1.0 - coach_ct / len(dep_out)

    log(f"Objective={m.ObjVal:.4f} | PSL≈{psl*100:.2f}% | coached={coach_ct}/{len(dep_out)}", "✅")

    # Save core outputs
    out_csv = os.path.join(CSV_DIR, "solution_gap.csv")
    dep_out.to_csv(out_csv, index=False)
    log(f"Saved solution: {out_csv}", "💾")

    # Save capacities (useful for analysis)
    cap_df = pd.DataFrame({"gate": list(cap_g.keys()), "K_g": list(cap_g.values())}).sort_values("gate")
    cap_df.to_csv(os.path.join(CSV_DIR, "gate_capacity.csv"), index=False)

    return dep_out, hist


if __name__ == "__main__":
    dep, hist = load_data(EXCEL_PATH)

    # Optional: run historic plots in separate script (recommended)
    hist_an.run_historic_analysis(hist, csv_dir=CSV_DIR, plot_dir=PLOT_DIR)

    # Solve
    dep_solution, hist = solve_gap(dep, hist)

    # Optional: run optimized plots in separate script (recommended)
    opt_an.run_optimized_analysis(dep_solution, csv_dir=CSV_DIR, plot_dir=PLOT_DIR)