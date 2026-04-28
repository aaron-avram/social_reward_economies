from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).resolve().parent

EXP_A_DIR = ROOT / "outputs" / "exp_a" / "final_pu_baseline"
EXP_C_DIR = ROOT / "outputs" / "exp_c" / "final_status_sweep"
OUT_DIR = ROOT / "outputs" / "final_report_figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)

PU_RUNS = EXP_A_DIR / "pu_scaling_runs_static.csv"
PU_AGG = EXP_A_DIR / "pu_scaling_aggregate_static.csv"
PU_PROGRESS = EXP_A_DIR / "pu_progression_static.csv"

STATUS_RUNS = EXP_C_DIR / "status_scaling_runs_static.csv"
STATUS_AGG = EXP_C_DIR / "status_scaling_aggregate_static.csv"

# ============================================================
# Helpers
# ============================================================
def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def save_close(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close()


def errorbar_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    yerr: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    ylim=None,
) -> None:
    plt.figure(figsize=(7, 4.6))
    plt.errorbar(
        df[x],
        df[y],
        yerr=df[yerr],
        marker="o",
        capsize=4,
        linewidth=1.8,
    )
    plt.title(title, fontsize=13)
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(alpha=0.25)
    save_close(filename)


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns were found: {candidates}")


# ============================================================
# Main
# ============================================================
def main() -> None:
    # Check files
    for path in [PU_RUNS, PU_AGG, PU_PROGRESS, STATUS_RUNS, STATUS_AGG]:
        require_file(path)

    # Load
    pu_runs = pd.read_csv(PU_RUNS)
    pu_agg = pd.read_csv(PU_AGG)
    pu_progress = pd.read_csv(PU_PROGRESS)
    status_runs = pd.read_csv(STATUS_RUNS)
    status_agg = pd.read_csv(STATUS_AGG)

    # Sort
    if "num_states" in pu_runs.columns:
        pu_runs = pu_runs.sort_values(["num_states"]).reset_index(drop=True)
    if "num_states" in pu_agg.columns:
        pu_agg = pu_agg.sort_values(["num_states"]).reset_index(drop=True)

    status_runs = status_runs.sort_values(["kappa"]).reset_index(drop=True)
    status_agg = status_agg.sort_values(["kappa"]).reset_index(drop=True)

    # ========================================================
    # Experiment A
    # ========================================================

    # Figure A1: Max followers over time for Experiment A
    time_col = find_col(pu_progress, ["t", "timestep", "time"])
    followers_col = find_col(pu_progress, ["top_followers", "max_followers", "final_top_followers"])

    grouped = pu_progress.groupby(time_col)[followers_col]
    ts_df = grouped.agg(["mean", "std", "count"]).reset_index()
    ts_df["ci95"] = 1.96 * ts_df["std"].fillna(0.0) / (ts_df["count"] ** 0.5)

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(ts_df[time_col], ts_df["mean"], linewidth=2)

    if ts_df["ci95"].max() > 0:
        plt.fill_between(
            ts_df[time_col],
            ts_df["mean"] - ts_df["ci95"],
            ts_df["mean"] + ts_df["ci95"],
            alpha=0.2,
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("Maximum followers over time", fontsize=13)
    plt.xlabel("Timestep", fontsize=11)
    plt.ylabel("Maximum number of followers", fontsize=11)
    plt.ylim(-1, 5)
    plt.grid(alpha=0.25)
    save_close("expA_followers_timeseries.png")

    
    # ========================================================
    # Experiment C
    # ========================================================

    # Figure C1: leader is status vs kappa
    errorbar_plot(
        df=status_agg,
        x="kappa",
        y=find_col(status_agg, ["mean_leader_is_status_final"]),
        yerr=find_col(status_agg, ["ci95_leader_is_status_final"]),
        title="Final leader is STATUS vs $\\kappa$",
        xlabel="$\\kappa$",
        ylabel="Probability final leader is STATUS",
        filename="expC_leader_is_status_vs_kappa.png",
        ylim=(-0.05, 1.05),
    )

    # Figure C2: tail welfare vs kappa
    errorbar_plot(
        df=status_agg,
        x="kappa",
        y=find_col(status_agg, ["mean_tail_welfare"]),
        yerr=find_col(status_agg, ["ci95_tail_welfare"]),
        title="Tail welfare vs $\\kappa$",
        xlabel="$\\kappa$",
        ylabel="Mean tail welfare",
        filename="expC_tail_welfare_vs_kappa.png",
    )

    # Figure C3: welfare gap to best norm vs kappa
    # This may not exist in aggregate csv yet, so compute from runs if needed.
    if "mean_welfare_gap_to_best" in status_agg.columns and "ci95_welfare_gap_to_best" in status_agg.columns:
        gap_df = status_agg.copy()
        gap_y = "mean_welfare_gap_to_best"
        gap_ci = "ci95_welfare_gap_to_best"
    else:
        # Build from runs
        gap_runs_col = find_col(status_runs, ["welfare_gap_to_best"])
        grouped = status_runs.groupby("kappa")[gap_runs_col]
        gap_df = grouped.agg(["mean", "std", "count"]).reset_index()
        gap_df["ci95"] = 1.96 * gap_df["std"].fillna(0.0) / (gap_df["count"] ** 0.5)
        gap_y = "mean"
        gap_ci = "ci95"

    errorbar_plot(
        df=gap_df,
        x="kappa",
        y=gap_y,
        yerr=gap_ci,
        title="Welfare gap to best norm vs $\\kappa$",
        xlabel="$\\kappa$",
        ylabel="Mean welfare gap to best norm",
        filename="expC_welfare_gap_vs_kappa.png",
    )

    # Figure C4: probability final norm is optimal vs kappa
    # This may not exist in aggregate csv yet, so compute from runs if needed.
    if "mean_is_final_norm_optimal" in status_agg.columns and "ci95_is_final_norm_optimal" in status_agg.columns:
        opt_df = status_agg.copy()
        opt_y = "mean_is_final_norm_optimal"
        opt_ci = "ci95_is_final_norm_optimal"
    else:
        opt_runs_col = find_col(status_runs, ["is_final_norm_optimal"])
        grouped = status_runs.groupby("kappa")[opt_runs_col]
        opt_df = grouped.agg(["mean", "std", "count"]).reset_index()
        opt_df["ci95"] = 1.96 * opt_df["std"].fillna(0.0) / (opt_df["count"] ** 0.5)
        opt_y = "mean"
        opt_ci = "ci95"

    errorbar_plot(
        df=opt_df,
        x="kappa",
        y=opt_y,
        yerr=opt_ci,
        title="Probability final norm is optimal vs $\\kappa$",
        xlabel="$\\kappa$",
        ylabel="Probability final norm is optimal",
        filename="expC_optimal_norm_probability_vs_kappa.png",
        ylim=(-0.05, 1.05),
    )

    print(f"Saved all figures to: {OUT_DIR}")
    print("Generated files:")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(" -", p.name)



if __name__ == "__main__":
    main()