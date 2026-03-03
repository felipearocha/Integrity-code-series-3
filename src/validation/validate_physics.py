import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from simulation.physics_model import CarParams, TrackParams, run_lap, grip_window

def test_grip_window_peak():
    cp = CarParams()
    fT_peak = grip_window(cp.T_opt, cp)
    assert abs(fT_peak - 1.0) < 1e-10, 'Gaussian grip window peak must be 1.0 at T_opt'
    print('PASS: grip_window peak = 1.0 at T_opt')

def test_grip_window_falloff():
    cp = CarParams()
    fT_cold = grip_window(cp.T_opt - 2*cp.sigma_T, cp)
    assert fT_cold < 0.15, 'Grip must fall significantly outside 2-sigma window'
    print('PASS: grip_window falloff outside 2-sigma: %.4f' % fT_cold)

def test_traction_ceiling_positive():
    cp = CarParams(); tp = TrackParams()
    res = run_lap(cp, tp)
    assert (res['F_trac_max'] > 0).all(), 'Traction ceiling must be positive everywhere'
    print('PASS: F_trac_max > 0 everywhere')

def test_soc_bounds():
    cp = CarParams(); tp = TrackParams()
    res = run_lap(cp, tp)
    assert res['SOC'].min() >= 0.19, 'SOC must not fall below lower bound'
    assert res['SOC'].max() <= 0.96, 'SOC must not exceed upper bound'
    print('PASS: SOC bounds respected [%.3f, %.3f]' % (res['SOC'].min(), res['SOC'].max()))

def test_tyre_temp_bounds():
    cp = CarParams(); tp = TrackParams()
    res = run_lap(cp, tp)
    assert res['T_tire'].min() >= 39.0, 'Tyre temp must not fall below 40C floor'
    assert res['T_tire'].max() <= 151.0, 'Tyre temp must not exceed 150C ceiling'
    print('PASS: T_tire bounds respected [%.1fC, %.1fC]' % (res['T_tire'].min(), res['T_tire'].max()))

def test_velocity_positive():
    cp = CarParams(); tp = TrackParams()
    res = run_lap(cp, tp)
    assert (res['v'] > 0).all(), 'Velocity must remain positive throughout lap'
    print('PASS: v > 0 throughout lap')

def test_wear_monotonic():
    cp = CarParams(); tp = TrackParams()
    res = run_lap(cp, tp)
    diff = np.diff(res['wear'])
    assert (diff >= -1e-8).all(), 'Tyre wear must be monotonically non-decreasing'
    print('PASS: tyre wear monotonically non-decreasing')

def test_lap_time_physical():
    cp = CarParams(); tp = TrackParams()
    res = run_lap(cp, tp)
    lt = res['lap_time']
    assert 55.0 < lt < 150.0, 'Lap time %.2f s outside physical range [55, 150] s' % lt
    print('PASS: lap time physical: %.2f s' % lt)

def test_energy_conservation():
    cp = CarParams(); tp = TrackParams()
    res = run_lap(cp, tp)
    dSOC = res['SOC'][0] - res['SOC'][-1]
    E_used = dSOC * cp.E_bat / 1e6
    assert E_used >= 0.0, 'Net energy consumed must be non-negative'
    print('PASS: net battery energy consumed = %.3f MJ' % E_used)

def run_all():
    tests = [
        test_grip_window_peak,
        test_grip_window_falloff,
        test_traction_ceiling_positive,
        test_soc_bounds,
        test_tyre_temp_bounds,
        test_velocity_positive,
        test_wear_monotonic,
        test_lap_time_physical,
        test_energy_conservation,
    ]
    passed = 0; failed = 0
    for t in tests:
        try:
            t(); passed += 1
        except AssertionError as e:
            print('FAIL: ' + str(e)); failed += 1
        except Exception as e:
            print('ERROR: ' + str(e)); failed += 1
    print('\nValidation complete: %d passed  %d failed' % (passed, failed))
    return failed == 0

if __name__ == '__main__':
    ok = run_all()
    sys.exit(0 if ok else 1)
