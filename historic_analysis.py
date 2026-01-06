import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def to_minutes(series: pd.Series) -> pd.Series:
    """Convert time-like values to minutes-from-midnight."""
    sdt = pd.to_datetime(series, errors="coerce")
    if sdt.notna().any():
        return (sdt.dt.hour * 60 + sdt.dt.minute).astype(int)
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def min_to_hhmm(m: int) -> str:
    h = int(m // 60)
    mi = int(m % 60)
    return f"{h:02d}:{mi:02d}"


def run_historic_analysis(hist: pd.DataFrame, csv_dir="csv_outputs", plot_dir="plot_outputs"):
    """
    Historic-only diagnostics.
    Expected columns in hist:
      - departure_gate (int)
      - passengers (int)
      - coaches (optional numeric)
      - boarding_time OR flight_time (HH:MM)
    """
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    h = hist.copy()

    # IsCoached: if coaches > 0
    if "coaches" in h.columns:
        coaches = pd.to_numeric(h["coaches"], errors="coerce").fillna(0.0)
        h["is_coached"] = (coaches > 0).astype(int)
    else:
        h["is_coached"] = np.nan

    # PSL (overall)
    if h["is_coached"].notna().any():
        coached_share = float(h["is_coached"].mean())
        psl = 1.0 - coached_share
        psl_df = pd.DataFrame({"psl": [psl], "coached_share": [coached_share]})
        psl_df.to_csv(os.path.join(csv_dir, "historic_psl_overall.csv"), index=False)

    # Coach share per gate
    if h["is_coached"].notna().any():
        gate_stats = (
            h.dropna(subset=["departure_gate"])
             .groupby("departure_gate")["is_coached"]
             .agg(total_flights="count", coached_flights="sum")
             .reset_index()
        )
        gate_stats["coach_share"] = np.where(
            gate_stats["total_flights"] > 0,
            gate_stats["coached_flights"] / gate_stats["total_flights"],
            0.0,
        )
        gate_stats = gate_stats.sort_values("departure_gate")
        gate_stats.to_csv(os.path.join(csv_dir, "historic_coach_share_by_gate.csv"), index=False)

        plt.figure(figsize=(14, 5))
        plt.bar(gate_stats["departure_gate"].astype(str), 100.0 * gate_stats["coach_share"])
        plt.xlabel("Gate")
        plt.ylabel("Coach Share (%)")
        plt.title("Historic Coach Share by Gate")
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "historic_coach_share_by_gate.png"), dpi=200)
        plt.close()

    # Total passengers per gate
    gate_pax = (
        h.dropna(subset=["departure_gate", "passengers"])
         .groupby("departure_gate")["passengers"]
         .sum()
         .reset_index()
         .sort_values("departure_gate")
    )
    gate_pax.to_csv(os.path.join(csv_dir, "historic_total_pax_by_gate.csv"), index=False)

    plt.figure(figsize=(14, 5))
    plt.bar(gate_pax["departure_gate"].astype(str), gate_pax["passengers"] / 1000.0)
    plt.xlabel("Gate")
    plt.ylabel("Total Pax (K)")
    plt.title("Historic Total Passengers by Gate")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "historic_total_pax_by_gate.png"), dpi=200)
    plt.close()

    # Hourly PSL (if time exists)
    time_col = None
    for c in ["boarding_time", "flight_time"]:
        if c in h.columns:
            time_col = c
            break

    if time_col and h["is_coached"].notna().any():
        h["_tmin"] = to_minutes(h[time_col])
        h["hour"] = (h["_tmin"] // 60).astype(int)

        by_hour = (
            h.groupby("hour")["is_coached"]
             .agg(total="count", coached="sum")
             .reset_index()
        )
        by_hour["coached_share"] = np.where(by_hour["total"] > 0, by_hour["coached"] / by_hour["total"], np.nan)
        by_hour["psl"] = 1.0 - by_hour["coached_share"]
        by_hour.to_csv(os.path.join(csv_dir, "historic_psl_by_hour.csv"), index=False)

        plt.figure(figsize=(10, 4))
        plt.plot(by_hour["hour"], 100.0 * by_hour["psl"], marker="o")
        xticks = list(range(0, 25, 2))
        plt.xticks(xticks, [min_to_hhmm(hh * 60) for hh in xticks])
        plt.xlim(0, 24)
        plt.ylim(0, 101)
        plt.xlabel("Time (HH:MM)")
        plt.ylabel("PSL (%)")
        plt.title("Historic PSL by Hour")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "historic_psl_by_hour.png"), dpi=200)
        plt.close()