"""
plot_lap.py
===========
ICS2 Week 3  |  F1 Lap Simulation  |  INTEGRITY CODE SERIES

Generates all required visualizations from a completed lap simulation.

Outputs (saved to assets/outputs/):
  lap_telemetry.png       Primary hero  - 6-panel telemetry dashboard
  thermal_grip_map.png    Secondary     - tyre thermal grip surface
  sensitivity_v_mu.png    Secondary     - sensitivity of lap time to mu0 and T_opt
  residual_energy.png     Secondary     - ERS energy state vs lap fraction
  track_heatmap.png       Secondary     - speed heatmap on XY track
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulation.physics_model import CarParams, TrackParams, run_lap

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

BG  = "#0A0A0C"
P1  = "#13131A"
RED = "#D21E1E"
YEL = "#FFD732"
GRN = "#00E678"
BLU = "#A0C3FF"
WHT = "#F0F0F5"
GRY = "#787880"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   P1,
    "axes.edgecolor":   GRY,
    "axes.labelcolor":  WHT,
    "xtick.color":      GRY,
    "ytick.color":      GRY,
    "text.color":       WHT,
    "grid.color":       "#222230",
    "grid.linewidth":   0.5,
})


def hero_telemetry(res: dict, path: str):
    """6-panel telemetry dashboard."""
    s = res["s"] / 1000   # km

    fig, axes = plt.subplots(6, 1, figsize=(14, 18), dpi=100,
                              sharex=True)
    fig.patch.set_facecolor(BG)
    fig.suptitle("F1 Lap Simulation  |  ICS2 Week 3  |  INTEGRITY CODE SERIES",
                 color=WHT, fontsize=14, fontweight="bold", y=0.995)

    panels = [
        (axes[0], res["v_kmh"],     "Speed [km/h]",         BLU),
        (axes[1], res["T_tire"],    "Tyre Temp [°C]",        YEL),
        (axes[2], res["mu_eff"],    "μ_eff [-]",             GRN),
        (axes[3], res["SOC"]*100,   "SOC [%]",               GRN),
        (axes[4], res["P_MGUK_kW"], "P_MGU-K [kW]",         "#FF8C00"),
        (axes[5], res["wear"],      "Tyre Wear [%]",         RED),
    ]

    for ax, data, label, color in panels:
        ax.plot(s, data, color=color, lw=1.2)
        ax.fill_between(s, data, alpha=0.15, color=color)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True)
        ax.set_facecolor(P1)

    # overlay thermal window band on tyre temp panel
    axes[1].axhline(95, color=RED, lw=0.8, ls="--", alpha=0.6, label="T_opt 95°C")
    axes[1].axhline(75, color=GRY, lw=0.6, ls=":", alpha=0.5)
    axes[1].axhline(115, color=GRY, lw=0.6, ls=":", alpha=0.5)
    axes[1].legend(fontsize=8, loc="upper right")

    # mark SOC gate
    axes[3].axhline(35, color=RED, lw=0.8, ls="--", alpha=0.6, label="Deploy gate 35%")
    axes[3].legend(fontsize=8, loc="upper right")

    axes[5].set_xlabel("Arc length [km]", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.995])
    plt.savefig(path, dpi=100, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {path}")


def thermal_grip_surface(cp: CarParams, path: str):
    """2D contour: mu_eff as function of tyre temp and wear."""
    T_range    = np.linspace(40, 150, 200)
    wear_range = np.linspace(0, 1, 200)
    TT, WW = np.meshgrid(T_range, wear_range)
    fT    = np.exp(-0.5 * ((TT - cp.T_opt) / cp.sigma_T) ** 2)
    MU    = np.clip(cp.mu0 * fT * (1.0 - cp.k_w * WW * 1e4),
                    cp.mu_eff_min, cp.mu_eff_max)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(P1)
    cf = ax.contourf(TT, WW * 100, MU, levels=30, cmap="RdYlGn")
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label("μ_eff", color=WHT)
    cbar.ax.yaxis.set_tick_params(color=WHT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=WHT)
    ax.axvline(cp.T_opt, color=WHT, lw=1.0, ls="--", alpha=0.7,
               label=f"T_opt = {cp.T_opt}°C")
    ax.set_xlabel("Tyre Temperature [°C]")
    ax.set_ylabel("Tyre Wear [%]")
    ax.set_title("Gaussian Grip Window  |  μ_eff(T, wear)",
                 color=WHT, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {path}")


def sensitivity_lap_time(base_cp: CarParams, base_tp: TrackParams, path: str):
    """Sensitivity of lap time to mu0 and T_opt."""
    mu0_range  = np.linspace(1.40, 2.10, 9)
    topt_range = np.linspace(75, 115, 9)
    lap_mu0  = []
    lap_topt = []

    base_lt = run_lap(base_cp, base_tp)["lap_time"]

    for mu0 in mu0_range:
        cp2 = CarParams(); cp2.mu0 = mu0
        lap_mu0.append(run_lap(cp2, base_tp)["lap_time"])

    for topt in topt_range:
        cp2 = CarParams(); cp2.T_opt = topt
        lap_topt.append(run_lap(cp2, base_tp)["lap_time"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=100)
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(P1); ax.grid(True)

    ax1.plot(mu0_range, lap_mu0, color=GRN, lw=2, marker="o", ms=5)
    ax1.axvline(base_cp.mu0, color=RED, lw=1, ls="--", alpha=0.7,
                label=f"baseline μ₀={base_cp.mu0}")
    ax1.set_xlabel("Baseline Friction Coefficient μ₀")
    ax1.set_ylabel("Lap Time [s]")
    ax1.set_title("Sensitivity: Lap Time vs μ₀", color=WHT)
    ax1.legend(fontsize=9)

    ax2.plot(topt_range, lap_topt, color=YEL, lw=2, marker="o", ms=5)
    ax2.axvline(base_cp.T_opt, color=RED, lw=1, ls="--", alpha=0.7,
                label=f"baseline T_opt={base_cp.T_opt}°C")
    ax2.set_xlabel("Optimal Tyre Temperature T_opt [°C]")
    ax2.set_ylabel("Lap Time [s]")
    ax2.set_title("Sensitivity: Lap Time vs T_opt", color=WHT)
    ax2.legend(fontsize=9)

    fig.suptitle("Lap Time Sensitivity Analysis  |  ICS2 Week 3",
                 color=WHT, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {path}")


def ers_energy_audit(res: dict, path: str):
    """ERS energy residual plot: SOC and cumulative regen/deploy."""
    s       = res["s"] / 1000
    SOC     = res["SOC"]
    P_kW    = res["P_MGUK_kW"]
    dt      = res["dt"]
    E_regen = np.cumsum(np.clip(-P_kW, 0, None) * dt) / 1000   # kWh
    E_dep   = np.cumsum(np.clip( P_kW, 0, None) * dt) / 1000

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=100, sharex=True)
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(P1); ax.grid(True)

    ax1.plot(s, SOC * 100, color=GRN, lw=1.5, label="SOC %")
    ax1.axhline(35, color=RED, lw=0.8, ls="--", alpha=0.7, label="Deploy gate 35%")
    ax1.axhline(20, color=RED, lw=0.6, ls=":",  alpha=0.5, label="Lower limit 20%")
    ax1.set_ylabel("State of Charge [%]")
    ax1.set_title("ERS Energy Audit", color=WHT, fontsize=11)
    ax1.legend(fontsize=8)

    ax2.fill_between(s, E_regen, alpha=0.4, color=GRN, label="Cumulative Regen [kWh]")
    ax2.fill_between(s, -E_dep,  alpha=0.4, color=RED,  label="Cumulative Deploy [kWh]")
    ax2.plot(s, E_regen, color=GRN, lw=1.2)
    ax2.plot(s, -E_dep,  color=RED,  lw=1.2)
    ax2.set_xlabel("Arc length [km]")
    ax2.set_ylabel("Energy [kWh]")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {path}")


def track_speed_heatmap(res: dict, path: str):
    """XY track coloured by speed."""
    x     = res["track"]["x"]
    y     = res["track"]["y"]
    v_kmh = res["v_kmh"]

    points  = np.array([x, y]).T.reshape(-1, 1, 2)
    segs    = np.concatenate([points[:-1], points[1:]], axis=1)
    norm    = plt.Normalize(v_kmh.min(), v_kmh.max())
    lc      = LineCollection(segs, cmap="plasma", norm=norm)
    lc.set_array(v_kmh[:-1])
    lc.set_linewidth(3)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.axis("off")
    cb = plt.colorbar(lc, ax=ax, pad=0.02)
    cb.set_label("Speed [km/h]", color=WHT)
    cb.ax.yaxis.set_tick_params(color=WHT)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=WHT)
    ax.set_title("Track Speed Heatmap  |  ICS2 Week 3",
                 color=WHT, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cp = CarParams()
    tp = TrackParams()
    print("Running simulation for visualizations...")
    res = run_lap(cp, tp)
    print(f"Lap time: {res['lap_time']:.2f} s\n")

    print("Generating visualizations...")
    hero_telemetry(res,        os.path.join(OUT_DIR, "lap_telemetry.png"))
    thermal_grip_surface(cp,   os.path.join(OUT_DIR, "thermal_grip_map.png"))
    sensitivity_lap_time(cp, tp, os.path.join(OUT_DIR, "sensitivity_v_mu.png"))
    ers_energy_audit(res,      os.path.join(OUT_DIR, "residual_energy.png"))
    track_speed_heatmap(res,   os.path.join(OUT_DIR, "track_heatmap.png"))
    print("\nAll visualizations saved to assets/outputs/")
