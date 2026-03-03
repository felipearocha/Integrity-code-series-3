#!/usr/bin/env python3
"""
run_all.py
==========
ICS2 Week 3  |  F1 Lap Simulation  |  INTEGRITY CODE SERIES

Master execution script. Runs in this order:
  1. Physics validation
  2. Lap simulation
  3. All visualizations
  4. GIF generation

Usage:
  python run_all.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print('=' * 60)
print('INTEGRITY CODE SERIES  |  ICS2 Week 3  |  F1 Lap Simulation')
print('=' * 60)

# Step 1: validation
print('\nSTEP 1: Physics Validation')
print('-' * 40)
from validation.validate_physics import run_all as validate
ok = validate()
if not ok:
    print('Validation failed. Aborting.')
    sys.exit(1)

# Step 2: run simulation
print('\nSTEP 2: Run Lap Simulation')
print('-' * 40)
from simulation.physics_model import CarParams, TrackParams, run_lap
cp = CarParams()
tp = TrackParams()
res = run_lap(cp, tp, verbose=True)
print('Lap time: %.2f s' % res['lap_time'])
print('Max speed: %.1f km/h' % res['v_kmh'].max())
print('Final SOC: %.1f %%' % (res['SOC'][-1] * 100))
print('Final tyre temp: %.1f C' % res['T_tire'][-1])
print('Final wear: %.2f %%' % res['wear'][-1])

# Step 3: visualizations
print('\nSTEP 3: Generate Visualizations')
print('-' * 40)
from visualization.plot_lap import (
    hero_telemetry, thermal_grip_surface,
    sensitivity_lap_time, ers_energy_audit, track_speed_heatmap, OUT_DIR
)
import os
hero_telemetry(res,          os.path.join(OUT_DIR, 'lap_telemetry.png'))
thermal_grip_surface(cp,     os.path.join(OUT_DIR, 'thermal_grip_map.png'))
sensitivity_lap_time(cp, tp, os.path.join(OUT_DIR, 'sensitivity_v_mu.png'))
ers_energy_audit(res,        os.path.join(OUT_DIR, 'residual_energy.png'))
track_speed_heatmap(res,     os.path.join(OUT_DIR, 'track_heatmap.png'))

# Step 4: GIF
print('\nSTEP 4: Generate GIF')
print('-' * 40)
from visualization.generate_gif import make_gif
make_gif(res, os.path.join(OUT_DIR, 'tyre_thermal_evolution.gif'))

print('\n' + '=' * 60)
print('All outputs saved to assets/outputs/')
print('=' * 60)
