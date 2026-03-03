# INTEGRITY CODE SERIES Week 3 — F1 Lap Simulation

**INTEGRITY CODE SERIES  |  Week 3**

Physics-informed F1 lap simulation using six coupled ODEs
integrated along arc length (space-marched scheme).

## What this repository is

A reproducible, physics-grounded lap simulation with:
- Six simultaneous state variables: v, beta, SOC, fuel, T_tire, wear
- First-order spatial aero mode filter
- Gaussian thermal grip window
- ERS regen/deploy gate logic
- Full longitudinal dynamics with traction ceiling

No proprietary data. No ML. Classical numerical integration only.

## Repository structure

```
ics2_week3_f1_lap_simulation/
├── src/
│   ├── simulation/
│   │   └── physics_model.py      # All governing equations + integrator
│   ├── visualization/
│   │   ├── plot_lap.py           # 5 static visualizations
│   │   └── generate_gif.py       # Animated GIF of tyre thermal evolution
│   └── validation/
│       └── validate_physics.py   # 9 physics consistency tests
├── assets/
│   └── outputs/                  # All generated figures saved here
├── notebooks/
│   └── explore_lap.ipynb         # Optional: interactive exploration
├── linkedin/
│   └── post_draft.txt            # LinkedIn post text
├── run_all.py                    # Master execution script
├── requirements.txt
└── README.md
```

## Execution order

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run everything
python run_all.py
```

This single command runs validation, simulation, all visualizations,
and the GIF generator in the correct dependency order.

## Individual modules

```bash
# Physics validation only
python src/validation/validate_physics.py

# Simulation only
python src/simulation/physics_model.py

# Visualizations only (requires simulation to have run)
python src/visualization/plot_lap.py

# GIF only
python src/visualization/generate_gif.py
```

## Governing equations summary

**Aero mode filter (spatial ODE):**
  dm/ds = (m_raw - m) / tau     tau = 40 m

**Aerodynamic forces:**
  F_drag = 0.5 * rho * Cd(m) * A * v^2
  F_down = 0.5 * rho * Cl(m) * A * v^2

**Traction ceiling:**
  F_trac_max = mu_eff * F_down * 4

**Tyre temperature ODE:**
  dT/dt = k_heat * E_slip - k_cool * (T - T_track)

**Gaussian grip window:**
  f_T = exp( -(T - T_opt)^2 / (2 * sigma^2) )
  mu_eff = mu0 * f_T * (1 - k_w * wear)

**ERS SOC ODE:**
  dSOC/dt = -P_bat / E_bat

**Master equation of motion:**
  m * dv/dt = F_drive - F_drag - F_roll - F_grade - F_brake

**Integration:**
  dt_i = ds / (v_i * cos(beta_i) + eps)
  t_lap = sum(dt_i)

## Key parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| m_car     | 768 kg | Car + driver mass |
| L         | 3300 m | Track length |
| N         | 1600   | Spatial nodes |
| T_opt     | 95 C   | Peak grip temperature |
| sigma     | 20 C   | Thermal window half-width |
| E_bat     | 4.0 MJ | Battery energy |
| P_deploy_max | 350 kW | MGU-K deploy ceiling |
| P_regen_max  | 250 kW | Regen ceiling |

## Outputs

| File | Description |
|------|-------------|
| lap_telemetry.png | 6-panel: v, T, mu, SOC, P_MGUK, wear |
| thermal_grip_map.png | mu_eff contour vs T and wear |
| sensitivity_v_mu.png | Lap time sensitivity to mu0 and T_opt |
| residual_energy.png | ERS energy audit with SOC |
| track_heatmap.png | XY track colored by speed |
| tyre_thermal_evolution.gif | Animated thermal/grip/SOC evolution |

## Reproducibility

Results are deterministic. Random seed is fixed at 42 in track geometry
generation. Running run_all.py twice produces identical outputs.

## INTEGRITY CODE SERIES

Physics-first engineering.
Secure digital integrity systems.
Operationally defensible decisions.
Verification over visibility.
