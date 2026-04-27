"""
Diagnostic Pneumotachometer Engine (Electronic Spirometry)
Core physics engine simulating Hagen-Poiseuille transduction,
BTPS correction, GLI/NHANES III demographics, and Trapezoidal integration.
"""

import numpy as np
from scipy.signal import butter, filtfilt

class PneumotachEngine:
    def __init__(self):
        # System Constants
        self.fs = 1000.0             # Sampling frequency (1 kHz)
        self.dt = 1.0 / self.fs      # Time step (1 ms)
        self.t_end = 6.0             # Total simulation time (6 seconds)

        # Fleisch Device Constants (from Report)
        self.peak_flow_ref = 8.0     # Reference peak flow in L/s
        self.peak_dP_ref = 177.0     # Reference peak differential pressure in Pa
        self.K = self.peak_flow_ref / self.peak_dP_ref  # Conductance constant (Q = K * dP)

        # State Management
        self.is_running = False
        self.current_index = 0
        self.data = None
        self.current_profile = "Normal"

    def _calculate_predicted(self, age, height_cm, sex):
        """Calculate predicted FVC and FEV1 using simplified NHANES III equations"""
        if sex == "Male":
            pred_fvc = (0.0600 * height_cm) - (0.0214 * age) - 4.650
            pred_fev1 = (0.0414 * height_cm) - (0.0244 * age) - 2.190
        else: # Female
            pred_fvc = (0.0491 * height_cm) - (0.0216 * age) - 3.590
            pred_fev1 = (0.0342 * height_cm) - (0.0255 * age) - 1.578

        return max(pred_fvc, 1.0), max(pred_fev1, 1.0)

    def start_maneuver(self, patient_profile="Normal", age=25, height_cm=175, sex="Male"):
        """Start a 6-second maneuver with demographic data"""
        self.current_profile = patient_profile
        self.is_running = True
        self.current_index = 0
        self._generate_waveform(patient_profile, age, height_cm, sex)

    def stop_maneuver(self):
        self.is_running = False

    def _generate_waveform(self, profile, age, height_cm, sex):
        """Mathematical synthesis of the waveform, BTPS, filtering, and integration."""
        t = np.arange(0, self.t_end, self.dt)
        btps_factor = 1.11  # Standard Body Temp Pressure Saturated correction

        # Select Physiological Parameters
        if profile == "Normal":
            peak_dP, tau, noise_std, dc_offset = 150.0, 0.50, 3.5, 0.0
        elif profile == "Obstructive (COPD)":
            peak_dP, tau, noise_std, dc_offset = 95.0, 1.50, 3.5, 0.0
        elif profile == "Restrictive":
            peak_dP, tau, noise_std, dc_offset = 120.0, 0.30, 3.5, 0.0
        elif profile == "Sensor Zero-Drift":
            peak_dP, tau, noise_std, dc_offset = 150.0, 0.50, 3.5, 3.0 # 3 Pa drift ruins integration
        elif profile == "3L Syringe Calibration":
            peak_dP, tau, noise_std, dc_offset = 0.0, 0.0, 0.5, 0.0 # Handled manually below
            btps_factor = 1.00 # Syringes use room air (ATPD), no BTPS needed
        else:
            peak_dP, tau, noise_std, dc_offset = 150.0, 0.50, 3.5, 0.0

        dP_ideal = np.zeros_like(t)

        if profile == "3L Syringe Calibration":
            # Generate a perfect 3.0 L stroke over 3 seconds
            mask = t <= 3.0
            Q_ideal = np.zeros_like(t)
            Q_ideal[mask] = (np.pi / 2.0) * np.sin(np.pi * t[mask] / 3.0)
            dP_ideal = Q_ideal / self.K
        else:
            # Phase 1: Sinusoidal Rise
            t_rise = 0.35
            rise_mask = t <= t_rise
            dP_ideal[rise_mask] = peak_dP * np.sin((np.pi * t[rise_mask]) / (2 * t_rise))

            # Phase 2: Exponential Decay
            decay_mask = t > t_rise
            dP_ideal[decay_mask] = peak_dP * np.exp(-(t[decay_mask] - t_rise) / tau)

        # Transduction Stage: Add Sensor Noise and Drift
        dP_noisy = dP_ideal + np.random.normal(0, noise_std, len(t)) + dc_offset

        # Electronic Stage: 2nd-Order Anti-Aliasing Filter (Zero-Phase)
        fc = 50.0
        b, a = butter(2, fc / (0.5 * self.fs), btype='low', analog=False)
        dP_filtered = filtfilt(b, a, dP_noisy)

        # Pneumatic Conversion & Thermodynamics: Hagen-Poiseuille Law * BTPS
        Q = (self.K * dP_filtered) * btps_factor

        if profile != "Sensor Zero-Drift":
            Q = np.maximum(Q, 0)

        # Processing Stage: Trapezoidal Numerical Integration
        V = np.zeros_like(t)
        for i in range(1, len(t)):
            V[i] = V[i-1] + 0.5 * (Q[i] + Q[i-1]) * self.dt

        # Algorithmic Feature Extraction
        fev1_idx = int(1.0 / self.dt)
        fev1 = V[fev1_idx]

        fvc = V[-1]
        end_test_indices = np.where((t > 1.0) & (Q < 0.025))[0]
        if len(end_test_indices) > 0:
            fvc = V[end_test_indices[0]]

        ratio = (fev1 / fvc) * 100.0 if fvc > 0 else 0.0

        # Demographics & Predicted Values
        pred_fvc, pred_fev1 = self._calculate_predicted(age, height_cm, sex)
        pct_fvc = (fvc / pred_fvc) * 100.0
        pct_fev1 = (fev1 / pred_fev1) * 100.0

        # Store generated data
        self.data = {
            't': t, 'dP_noisy': dP_noisy, 'dP_filtered': dP_filtered, 'Q': Q, 'V': V,
            'fev1': fev1, 'fvc': fvc, 'ratio': ratio, 'total_samples': len(t),
            'pct_fvc': pct_fvc, 'pct_fev1': pct_fev1
        }

    def get_current_state(self, advance_by_ms=50):
        """Simulates real-time data streaming"""
        if not self.is_running or self.data is None:
            return {'is_running': False, 'finished': False, 'Q_current': 0.0, 'V_current': 0.0}

        self.current_index += advance_by_ms
        if self.current_index >= self.data['total_samples']:
            self.current_index = self.data['total_samples'] - 1
            self.is_running = False

        idx = self.current_index

        return {
            'is_running': self.is_running, 'finished': not self.is_running,
            'profile': self.current_profile,

            # Clinical Features
            'fev1': self.data['fev1'], 'fvc': self.data['fvc'], 'ratio': self.data['ratio'],
            'pct_fvc': self.data['pct_fvc'], 'pct_fev1': self.data['pct_fev1'],

            # Arrays for plotting
            't_array': self.data['t'][:idx+1],
            'dP_noisy_array': self.data['dP_noisy'][:idx+1],
            'dP_filtered_array': self.data['dP_filtered'][:idx+1],
            'Q_array': self.data['Q'][:idx+1],
            'V_array': self.data['V'][:idx+1]
        }