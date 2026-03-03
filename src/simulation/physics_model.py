"""
physics_model.py
================
ICS2 Week 3  |  F1 Lap Simulation  |  INTEGRITY CODE SERIES

Six-state space-marched coupled ODE system.

States integrated at each spatial node ds:
  v         longitudinal velocity          [m/s]
  beta      lateral slip angle proxy       [rad]
  SOC       battery state of charge        [0,1]
  fuel      fuel mass remaining            [kg]
  T_tire    tyre bulk temperature          [C]
  wear      tyre wear state                [0,1]

Governing equations
-------------------
Aero mode filter (spatial ODE):
  dm/ds = (m_raw - m) / tau_aero

Aerodynamic forces:
  F_drag = 0.5 * rho * Cd(m) * A * v^2
  F_down = 0.5 * rho * Cl(m) * A * v^2

Traction ceiling:
  F_trac_max = mu_eff * F_down * 4

Master equation of motion:
  m_car * dv/dt = F_drive - F_drag - F_roll - F_grade - F_brake

Tyre temperature ODE:
  dT/dt = k_heat * E_slip - k_cool * (T - T_track)
  E_slip = (|beta| + 0.25*b) * v^2 * 0.1

Gaussian grip window:
  f_T = exp( -(T - T_opt)^2 / (2 * sigma^2) )
  mu_eff = mu0 * f_T * (1 - k_w * wear)

Wear ODE:
  dw/dt = k_w * E_slip * [0.6 + 0.8*(1 - f_T)]

ERS state of charge ODE:
  dSOC/dt = -P_bat / E_bat

Integration:
  dt_i = ds / (v_i * cos(beta_i) + eps)
  t_lap = sum(dt_i)

Assumptions
-----------
- 2D track, no banking, grade from synthetic curvature profile
- Tyre model is single-bulk-temperature (no radial gradient)
- ERS treated as rear-axle only
- No pit stop, single stint
- Air density constant (no altitude variation)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


# ── Physical constants ────────────────────────────────────────────────────────
G       = 9.81      # gravitational acceleration  [m/s^2]
RHO_AIR = 1.20      # air density                 [kg/m^3]

# ── Car parameters ────────────────────────────────────────────────────────────
@dataclass
class CarParams:
    m_car:         float = 768.0   # car + driver mass          [kg]
    A_ref:         float = 2.00    # frontal reference area     [m^2]
    Crr:           float = 0.012   # rolling resistance coeff   [-]
    eta_dt:        float = 0.92    # drivetrain efficiency      [-]

    # Aero coefficients - High Downforce (m=0)
    Cd_Z:          float = 1.00
    Cl_Z:          float = 3.50
    # Aero coefficients - Low Drag (m=1)
    Cd_X:          float = 0.75
    Cl_X:          float = 2.00
    tau_aero:      float = 40.0    # spatial smoothing length   [m]

    # Powertrain
    P_ICE_max:     float = 450e3   # ICE peak power             [W]
    P_MGUK_max:    float = 350e3   # MGU-K deploy ceiling       [W]
    P_regen_max:   float = 250e3   # regen ceiling              [W]
    E_bat:         float = 4.0e6   # usable battery energy      [J]

    # Tyre thermal
    T_opt:         float = 95.0    # peak grip temperature      [C]
    T_track:       float = 38.0    # track surface temp         [C]
    sigma_T:       float = 20.0    # thermal window half-width  [C]
    k_heat:        float = 1.2e-5  # heating coefficient
    k_cool:        float = 0.10    # cooling coefficient        [1/s]
    mu0:           float = 1.75    # baseline friction          [-]
    k_w:           float = 2.0e-6  # wear rate coefficient
    mu_eff_min:    float = 0.40
    mu_eff_max:    float = 2.20

    # Fuel consumption
    fuel_rate:     float = 0.032   # kg per second at max power

    # Initial conditions
    SOC_init:      float = 0.85
    fuel_init:     float = 30.0    # kg
    T_tire_init:   float = 55.0    # C  (not yet in window)
    wear_init:     float = 0.0


@dataclass
class TrackParams:
    L:             float = 3300.0  # total lap length           [m]
    N:             int   = 1600    # number of spatial nodes
    grade_amp:     float = 0.015   # max grade amplitude        [-]


# ── Track geometry ────────────────────────────────────────────────────────────

def build_track(tp: TrackParams) -> dict:
    """
    Generate synthetic closed circuit curvature and grade profiles.

    Curvature profile: sum of sinusoidal harmonics + 12 Gaussian corners.
    Grade profile:     single sine wave.

    Returns dict with keys: s, kappa, grade, x, y
    """
    s = np.linspace(0, tp.L, tp.N, endpoint=False)
    ds = s[1] - s[0]

    # base curvature from 3 Fourier harmonics
    kappa = (
        0.008 * np.sin(2 * np.pi * s / tp.L) +
        0.004 * np.sin(4 * np.pi * s / tp.L) +
        0.002 * np.sin(6 * np.pi * s / tp.L)
    )

    # add 12 Gaussian corners at realistic positions
    corner_positions = np.linspace(tp.L * 0.05, tp.L * 0.95, 12)
    corner_widths    = np.random.default_rng(42).uniform(60, 140, 12)
    corner_amps      = np.random.default_rng(42).uniform(0.012, 0.030, 12)
    for pos, w, amp in zip(corner_positions, corner_widths, corner_amps):
        kappa += amp * np.exp(-0.5 * ((s - pos) / w) ** 2)

    # grade profile
    grade = tp.grade_amp * np.sin(2 * np.pi * s / tp.L)

    # integrate to get xy for visualisation
    theta = np.cumsum(kappa * ds)
    x = np.cumsum(np.cos(theta) * ds)
    y = np.cumsum(np.sin(theta) * ds)

    return {"s": s, "ds": ds, "kappa": kappa, "grade": grade, "x": x, "y": y}


# ── Aero helpers ──────────────────────────────────────────────────────────────

def aero_coeffs(m: float, cp: CarParams) -> Tuple[float, float]:
    """Linear blend between High-DF and Low-Drag states."""
    Cd = (1 - m) * cp.Cd_Z + m * cp.Cd_X
    Cl = (1 - m) * cp.Cl_Z + m * cp.Cl_X
    return Cd, Cl


# ── Tyre helpers ──────────────────────────────────────────────────────────────

def grip_window(T: float, cp: CarParams) -> float:
    """Gaussian thermal grip function f_T(T)."""
    return np.exp(-0.5 * ((T - cp.T_opt) / cp.sigma_T) ** 2)


def mu_effective(T: float, wear: float, cp: CarParams) -> float:
    """Effective friction coefficient including thermal and wear degradation."""
    fT  = grip_window(T, cp)
    mu  = cp.mu0 * fT * (1.0 - cp.k_w * wear * 1e4)   # wear normalised
    return float(np.clip(mu, cp.mu_eff_min, cp.mu_eff_max))


# ── ERS helpers ───────────────────────────────────────────────────────────────

def ers_power(v: float, F_trac_max: float, SOC: float,
              brake_proxy: float, cp: CarParams) -> Tuple[float, float]:
    """
    Returns (P_MGUK, P_bat).
    P_MGUK > 0  = deploy
    P_MGUK < 0  = regen  (treated as P_bat < 0)
    """
    if brake_proxy > 0.35:
        # regen zone
        P_regen = min(cp.P_regen_max, 0.35 * F_trac_max * v)
        return -P_regen, -P_regen
    else:
        # deploy zone - gate on SOC
        gamma = float(np.clip((SOC - 0.35) / 0.25, 0.0, 1.0))
        P_deploy = gamma * 0.55 * cp.P_MGUK_max
        return P_deploy, P_deploy


# ── Core integrator ───────────────────────────────────────────────────────────

def run_lap(cp: CarParams, tp: TrackParams, verbose: bool = False) -> dict:
    """
    Space-marched integration over one lap.

    Returns dict of arrays (one value per spatial node):
      v, beta, SOC, fuel, T_tire, wear, m_aero,
      F_drag, F_down, F_trac_max, mu_eff_arr,
      P_MGUK, lap_time (scalar), t_cumulative
    """
    track = build_track(tp)
    s     = track["s"]
    ds    = track["ds"]
    kappa = track["kappa"]
    grade = track["grade"]
    N     = tp.N

    # allocate output arrays
    v_arr        = np.zeros(N)
    beta_arr     = np.zeros(N)
    SOC_arr      = np.zeros(N)
    fuel_arr     = np.zeros(N)
    T_arr        = np.zeros(N)
    wear_arr     = np.zeros(N)
    m_arr        = np.zeros(N)
    Fd_arr       = np.zeros(N)
    Fdown_arr    = np.zeros(N)
    Ftrac_arr    = np.zeros(N)
    mu_arr       = np.zeros(N)
    PMGUK_arr    = np.zeros(N)
    dt_arr       = np.zeros(N)
    t_arr        = np.zeros(N)

    # initial state
    v    = 60.0 / 3.6   # 60 km/h start
    SOC  = cp.SOC_init
    fuel = cp.fuel_init
    T    = cp.T_tire_init
    wear = cp.wear_init
    m_ao = 0.0          # aero mode starts at high-DF

    t_elapsed = 0.0
    EPS = 1e-3

    for i in range(N):
        # 1. aero mode filter
        kap_i   = kappa[i]
        kap_abs = np.percentile(np.abs(kappa), 70)
        m_raw   = 1.0 if abs(kap_i) < kap_abs else 0.0
        m_ao   += (ds / cp.tau_aero) * (m_raw - m_ao)
        m_ao    = float(np.clip(m_ao, 0.0, 1.0))

        # 2. aero forces
        Cd, Cl  = aero_coeffs(m_ao, cp)
        F_drag  = 0.5 * RHO_AIR * Cd * cp.A_ref * v ** 2
        F_down  = 0.5 * RHO_AIR * Cl * cp.A_ref * v ** 2

        # 3. effective friction and traction ceiling
        mu_eff  = mu_effective(T, wear, cp)
        F_tmax  = mu_eff * F_down * 4.0

        # 4. slip angle proxy
        beta    = float(np.clip(
            0.35 * np.arctan(v ** 2 * kap_i / G),
            -0.25, 0.25
        ))

        # 5. brake proxy (curvature-based approximation)
        brake_proxy = float(np.clip(abs(kap_i) / 0.04, 0.0, 1.0))

        # 6. ERS
        P_MGUK, P_bat = ers_power(v, F_tmax, SOC, brake_proxy, cp)

        # 7. total drive power
        P_drive = cp.eta_dt * (cp.P_ICE_max + max(P_MGUK, 0.0))
        v_safe  = max(v, 0.1)
        F_drive = min(F_tmax, P_drive / v_safe)

        # 8. longitudinal forces
        theta_g  = grade[i]
        F_roll   = cp.Crr * cp.m_car * G * np.cos(theta_g)
        F_grade  = cp.m_car * G * np.sin(theta_g)
        F_brake  = float(np.clip(
            mu_eff * F_down * 0.55 * brake_proxy,
            0.0, F_tmax
        ))

        # 9. acceleration and velocity update
        a_long  = (F_drive - F_drag - F_roll - F_grade - F_brake) / cp.m_car
        dt_i    = ds / (v * np.cos(beta) + EPS)
        v_new   = float(np.clip(v + a_long * dt_i, 1.0, 120.0))

        # 10. slip energy
        E_slip  = (abs(beta) + 0.25 * brake_proxy) * v ** 2 * 0.1

        # 11. tyre temperature ODE
        fT      = grip_window(T, cp)
        dT      = (cp.k_heat * E_slip - cp.k_cool * (T - cp.T_track)) * dt_i
        T_new   = float(np.clip(T + dT, 40.0, 150.0))

        # 12. wear ODE
        penalty = 0.6 + 0.8 * (1.0 - fT)
        dw      = cp.k_w * E_slip * penalty * dt_i
        wear_new = float(np.clip(wear + dw, 0.0, 1.0))

        # 13. SOC ODE
        dSOC    = -(P_bat / cp.E_bat) * dt_i
        SOC_new = float(np.clip(SOC + dSOC, 0.20, 0.95))

        # 14. fuel mass
        fuel_new = float(np.clip(
            fuel - cp.fuel_rate * (P_drive / cp.P_ICE_max) * dt_i,
            0.0, cp.fuel_init
        ))

        # store
        v_arr[i]     = v
        beta_arr[i]  = beta
        SOC_arr[i]   = SOC
        fuel_arr[i]  = fuel
        T_arr[i]     = T
        wear_arr[i]  = wear
        m_arr[i]     = m_ao
        Fd_arr[i]    = F_drag
        Fdown_arr[i] = F_down
        Ftrac_arr[i] = F_tmax
        mu_arr[i]    = mu_eff
        PMGUK_arr[i] = P_MGUK
        dt_arr[i]    = dt_i
        t_arr[i]     = t_elapsed

        t_elapsed += dt_i

        # advance state
        v    = v_new
        SOC  = SOC_new
        fuel = fuel_new
        T    = T_new
        wear = wear_new

        if verbose and i % 200 == 0:
            print(f"  s={s[i]:.0f}m  v={v*3.6:.1f}km/h  T={T:.1f}C  SOC={SOC:.3f}  wear={wear*100:.2f}%")

    return {
        "s":          s,
        "v":          v_arr,
        "v_kmh":      v_arr * 3.6,
        "beta":       beta_arr,
        "SOC":        SOC_arr,
        "fuel":       fuel_arr,
        "T_tire":     T_arr,
        "wear":       wear_arr * 100,    # percent
        "m_aero":     m_arr,
        "F_drag":     Fd_arr,
        "F_down":     Fdown_arr,
        "F_trac_max": Ftrac_arr,
        "mu_eff":     mu_arr,
        "P_MGUK_kW":  PMGUK_arr / 1e3,
        "dt":         dt_arr,
        "t":          t_arr,
        "lap_time":   float(np.sum(dt_arr)),
        "track":      track,
        "car":        cp,
        "track_p":    tp,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    cp = CarParams()
    tp = TrackParams()
    print("Running F1 lap simulation...")
    result = run_lap(cp, tp, verbose=True)
    print(f"\nLap time: {result['lap_time']:.2f} s")
    print(f"Final SOC: {result['SOC'][-1]*100:.1f}%")
    print(f"Final tyre temp: {result['T_tire'][-1]:.1f} C")
    print(f"Final wear: {result['wear'][-1]:.2f}%")
    print(f"Max speed: {result['v_kmh'].max():.1f} km/h")
