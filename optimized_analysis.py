import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def min_to_hhmm(m: int) -> str:
    h = int(m // 60)
    mi = int(m % 60)
    return f"{h:02d}:{mi:02d}"


def run_optimized_analysis(dep_solution: pd.DataFrame, csv_dir="csv_outputs", plot_dir="plot_outputs"):
    """
    Optimized-only diagnostics.
    Expected columns in dep_solution:
      - flight_code
      - B_min, D_min (optional; if missing we compute from boardingtime/departure_time)
      - passengers
      - assigned_gate
      - is_coached_model
    """
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    df = dep_solution.copy()

    # Ensure minutes exist
    if "B_min" not in df.columns or "D_min" not in df.columns:
        # Try to reconstruct if time strings exist
        if "boardingtime" in df.columns and "departure_time" in df.columns:
            b = pd.to_datetime(df["boardingtime"], errors="coerce")
            d = pd.to_datetime(df["departure_time"], errors="coerce")
            df["B_min"] = (b.dt.hour * 60 + b.dt.minute).fillna(0).astype(int)
            df["D_min"] = (d.dt.hour * 60 + d.dt.minute).fillna(0).astype(int)

    # PSL summary
    coached = int((df["is_coached_model"] == 1).sum())
    total = int(len(df))
    psl = 1.0 - coached / total if total else np.nan
    pd.DataFrame({"psl": [psl], "coached": [coached], "total": [total]}).to_csv(
        os.path.join(csv_dir, "optimized_psl_overall.csv"), index=False
    )

    # Gate timeline plot (simple)
    df = df.dropna(subset=["assigned_gate"]).copy()
    df["assigned_gate"] = df["assigned_gate"].astype(int)

    out_tl = df[["flight_code", "assigned_gate", "B_min", "D_min", "passengers", "is_coached_model"]].copy()
    out_tl.to_csv(os.path.join(csv_dir, "optimized_timeline_table.csv"), index=False)

    # Plot: one line segment per flight, grouped by gate
    df = df.sort_values(["assigned_gate", "B_min"])
    gates = sorted(df["assigned_gate"].unique().tolist())

    plt.figure(figsize=(15, 9))
    for yi, g in enumerate(gates):
        sub = df[df["assigned_gate"] == g]
        for _, r in sub.iterrows():
            plt.plot([r["B_min"], r["D_min"]], [yi, yi], linewidth=4, alpha=0.9)

    plt.yticks(range(len(gates)), gates)
    xticks = list(range(240, 1441, 120))
    plt.xticks(xticks, [min_to_hhmm(x) for x in xticks])
    plt.xlim(240, 1440)
    plt.xlabel("Boarding Time (HH:MM)")
    plt.ylabel("Gate")
    plt.title("Optimized Gate Timeline (Boarding–Departure)")
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "optimized_gate_timeline.png"), dpi=200)
    plt.close()

    # Plot: passengers per gate
    gate_pax = df.groupby("assigned_gate")["passengers"].sum().reset_index().sort_values("assigned_gate")
    gate_pax.to_csv(os.path.join(csv_dir, "optimized_total_pax_by_gate.csv"), index=False)

    plt.figure(figsize=(14, 5))
    plt.bar(gate_pax["assigned_gate"].astype(str), gate_pax["passengers"] / 1000.0)
    plt.xlabel("Gate")
    plt.ylabel("Total Pax (K)")
    plt.title("Optimized Total Passengers by Gate")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "optimized_total_pax_by_gate.png"), dpi=200)
    plt.close()