# INTEGRITY CODE SERIES Week 3 — F1 Lap Simulation

**INTEGRITY CODE SERIES  |  Week 3**

[![CI](https://github.com/felipearocha/Integrity-code-series-3/actions/workflows/ci.yml/badge.svg)](https://github.com/felipearocha/Integrity-code-series-3/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests: 9 passing](https://img.shields.io/badge/tests-9%20passing-brightgreen.svg)](src/validation/validate_physics.py)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Physics-informed F1 lap simulation using six coupled ODEs
integrated along arc length (space-marched scheme).

## Integrity Code Series

Part of an ongoing series of physics-first integrity simulators by Felipe Rocha:

| # | Repo | Domain |
|---|---|---|
| **Week 3** | **[Integrity-code-series-3](https://github.com/felipearocha/Integrity-code-series-3)** | **F1 lap simulation (six coupled ODEs) — this repo** |
| Week 6 | [integrity-code-series-week6-smartphone-galvanic](https://github.com/felipearocha/integrity-code-series-week6-smartphone-galvanic) | Smartphone galvanic corrosion (Laplace + Butler-Volmer) |
| Week 7 | [integrity_code_series_week7_h2_lferw](https://github.com/felipearocha/integrity_code_series_week7_h2_lferw) | LF-ERW H2 conversion (B31.12 + NACE TM0316) |
| Week 8 | [integrity-code-series-week8-creep-fatigue-heater](https://github.com/felipearocha/integrity-code-series-week8-creep-fatigue-heater) | Creep-fatigue 9Cr-1Mo (Norton/Omega + Coffin-Manson) |
| Week 9 | [integrity-code-series-week9-cui](https://github.com/felipearocha/integrity-code-series-week9-cui) | CUI thermohygro-electrochemical (3 PDEs, Strang) |
| Week 10 | [integrity-code-series-week-10_nnph_scc](https://github.com/felipearocha/integrity-code-series-week-10_nnph_scc) | NNpHSCC full-physics (Chen-Sutherby-Xing + BS 7910) |
| Week 11 | [integrity-code-series-week11-erosion-corrosion-multiphase](https://github.com/felipearocha/integrity-code-series-week11-erosion-corrosion-multiphase) | Erosion-corrosion multiphase (NORSOK M-506 + DNV-RP-O501 + G119 + API 579) |
| Bonus | [Vibration-Accelerated-Corrosion-Coupled-Mechano-Electrochemical-Simulation](https://github.com/felipearocha/Vibration-Accelerated-Corrosion-Coupled-Mechano-Electrochemical-Simulation) | Vibration-accelerated corrosion (SDOF + Butler-Volmer + Archard) |
| Bonus | [synthetic-integrity-digital-twin-piml](https://github.com/felipearocha/synthetic-integrity-digital-twin-piml) | Physics-informed neural-network surrogate |
| Bonus | [integrity-data-foundation](https://github.com/felipearocha/integrity-data-foundation) | Engineering data validation baseline |

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
Integrity-code-series-3/
├── src/
│   ├── simulation/
│   │   └── physics_model.py      # All governing equations + integrator
│   ├── visualization/
│   │   ├── plot_lap.py           # 5 static visualizations
│   │   └── generate_gif.py       # Animated GIF of tyre thermal evolution
│   └── validation/
│       └── validate_physics.py   # 9 physics consistency tests
├── docs/
│   └── equations.html            # Rendered (MathJax) governing-equations reference
├── notebooks/
│   └── explore_lap.ipynb         # Optional: interactive exploration
├── assets/
│   └── outputs/                  # Generated at run time — all figures + GIF saved here
├── run_all.py                    # Master execution script
├── pytest.ini                    # Test discovery config (collects the 9 validation tests)
├── requirements.txt
├── CHANGELOG.md
├── LICENSE
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

## Governing Equations

Full rendered (MathJax) reference: **[docs/equations.html](docs/equations.html)** — open in any browser.
Every relation below is transcribed from `src/simulation/physics_model.py`; standard-mechanics
relations are tagged `[SOURCE]` and model-specific proxies/coefficients `[ASSUMED]` in the rendered page.

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

## Escalation Table

| Week | Topic | Key escalation |
|------|-------|---------------|
| **3** | **F1 lap** | **Six coupled state ODEs (v, β, SOC, fuel, T_tire, wear) space-marched along arc length, with a Gaussian thermal grip window, a first-order spatial aero-mode filter, and an ERS deploy/regen gate** |
| 9 | CUI | 3 coupled PDEs, Strang splitting |
| 10 | NNpHSCC | Chen-Sutherby-Xing crack growth, crack colony, COV=61.2% epistemic |
| 11 | Erosion-corrosion | Coupled DNV erosion + NORSOK CO2 + Beggs-Brill flow + G119 synergy + API 579 Part 5 FFS |

## Cybersecurity (STRIDE)

This is a self-contained, offline research simulation: no network calls, no external
inputs, no secrets, and no persisted state beyond generated figures. The STRIDE attack
surface is therefore limited to code and parameter integrity. Mitigations in scope:

- **Tampering** — deterministic run (fixed seed 42) makes any change to code or
  parameters reproducible and diff-visible; the physics-consistency suite
  (`src/validation/validate_physics.py`, 9 tests) fails closed if governing-equation
  behaviour drifts (grip peak, SOC/temperature bounds, wear monotonicity, lap-time range).
- **Information disclosure** — no proprietary data and no ML; all inputs are the
  published `CarParams`/`TrackParams` defaults.
- **Denial of service** — bounded work (`N` spatial nodes, single stint) and clipped
  state variables prevent runaway integration.

Repudiation, Spoofing, and Elevation-of-Privilege are out of scope for a local,
single-user, no-I/O tool.

## Anti-Hallucination Note

Every relation in this package is standard classical mechanics or an explicitly
labelled modelling choice — nothing is attributed to a standard or paper it does not
come from. The tiers below are applied honestly in `docs/equations.html`:

- **T1 (SOURCE)** — textbook physics reproduced as-is: Newton's second law
  (longitudinal EOM), the aerodynamic force law `F = ½ρC A v²`, the Gaussian grip
  window, and the explicit space-marched integrator.
- **T2 (derived)** — quantities computed from T1 relations and the model parameters
  (traction ceiling from friction × downforce, slip-energy, lap time as the sum of
  local time steps).
- **T3 (ASSUMED / heuristic)** — model-specific proxies and tuned coefficients: the
  first-order aero-mode smoothing length, the curvature-based slip-angle and brake
  proxies, and the tyre heating/cooling/wear coefficients. These are engineering
  choices, not measured or standard values, and are tagged `[ASSUMED]` in the rendered
  equations. No external standard, DOI, or literature citation is claimed anywhere in
  this repository.

## Disclaimer

Research tool only. Not for design, fitness-for-service, or safety-critical decisions without site-specific calibration and independent PE review.

This simulation is a physics-grounded teaching and exploration model. It uses no
proprietary data, no measured tyre/aero maps, and no external standard; several
coefficients are engineering assumptions (see the Anti-Hallucination Note). It is not a
substitute for validated vehicle-dynamics tools.

## License

MIT — Felipe Rocha. See [LICENSE](LICENSE). Usage restrictions, if any, are covered by
the Disclaimer above; the software itself is released under the MIT License.

## How to Cite

If this software contributes to your work, please cite it:

> Rocha, F. (2026). *Integrity Code Series — Week 3 — F1 Lap Simulation (Six Coupled ODEs)* [Computer software]. GitHub. https://github.com/felipearocha/Integrity-code-series-3

**BibTeX:**

```bibtex
@software{rocha_2026_ics_week3_f1_lap,
  author    = {Rocha, Felipe},
  title     = {{Integrity Code Series --- Week 3 --- F1 Lap Simulation
               (Six Coupled ODEs)}},
  year      = 2026,
  publisher = {GitHub},
  url       = {https://github.com/felipearocha/Integrity-code-series-3}
}
```

No archival DOI (e.g. Zenodo) has been minted for this repository yet; cite the GitHub
URL above. When a DOI is issued it will be added here as concept (latest) and version
(pinned) identifiers.

## INTEGRITY CODE SERIES

Physics-first engineering.
Secure digital integrity systems.
Operationally defensible decisions.
Verification over visibility.