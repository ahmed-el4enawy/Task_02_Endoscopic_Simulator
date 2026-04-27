"""
cv_engine.py
Advanced Computer Vision engine for the Endoscopic Simulator.
Implements Fisheye distortion, NBI simulation, Parametric Bilateral Denoising,
CLAHE, and custom Sobel edge overlays.
"""

import cv2
import numpy as np
import os

class EndoscopicEngine:
    def __init__(self, image_path=None):
        self.base_image = None
        self.h, self.w = 720, 1280

        # Cached physics masks
        self._vignette_mask = None
        self._map_x = None
        self._map_y = None

        # Parametric Hardware States
        self.illumination_intensity = 1.0
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0

        # Parametric Processing States
        self.denoise_strength = 0          # 0 to 150
        self.clahe_limit = 0.0             # 0.0 to 5.0
        self.texture_opacity = 0.0         # 0.0 to 1.0
        self.anomaly_min_area = 5000       # 5000 down to 100

        # Toggles
        self.enable_nbi = False
        self.enable_fisheye = False
        self.enable_pip = False
        self.enable_hud = True

        self.load_image(image_path)

    def load_image(self, filepath):
        """Loads source data and pre-calculates physics matrices."""
        if filepath and os.path.exists(filepath):
            self.base_image = cv2.imread(filepath)
        else:
            self.base_image = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(self.base_image, "NO SENSOR DATA.", (450, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        self.h, self.w = self.base_image.shape[:2]
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0

        self._generate_vignette_mask()
        self._generate_fisheye_maps()

    def _generate_vignette_mask(self):
        """Calculates a radial gradient to simulate a fiber-optic spotlight."""
        X = np.linspace(-1, 1, self.w)
        Y = np.linspace(-1, 1, self.h)
        x, y = np.meshgrid(X, Y)
        radius = np.sqrt(x**2 + y**2)

        # Gaussian falloff
        mask = np.exp(-(radius**2) / 0.8)
        self._vignette_mask = np.stack([mask]*3, axis=-1)

    def _generate_fisheye_maps(self):
        """Pre-calculates pixel maps for optical Barrel Distortion."""
        x, y = np.meshgrid(np.arange(self.w), np.arange(self.h))

        # Normalize to center [-1, 1]
        x_c = 2.0 * x / self.w - 1.0
        y_c = 2.0 * y / self.h - 1.0
        r = np.sqrt(x_c**2 + y_c**2)
        r[r == 0] = 1 # Prevent division by zero

        # Distortion physics (theta)
        theta = r + 0.15 * r**3

        x_distorted = x_c * (theta / r)
        y_distorted = y_c * (theta / r)

        # Denormalize
        self._map_x = ((x_distorted + 1.0) * self.w / 2.0).astype(np.float32)
        self._map_y = ((y_distorted + 1.0) * self.h / 2.0).astype(np.float32)

    def get_frame(self):
        """Master render pipeline."""
        raw_frame = self.base_image.copy()
        frame = raw_frame.copy()

        # 1. Physics: NBI Optical Filter
        if self.enable_nbi:
            frame = self._apply_nbi(frame)

        # 2. Physics: Fisheye Lens Distortion
        if self.enable_fisheye:
            frame = cv2.remap(frame, self._map_x, self._map_y, cv2.INTER_LINEAR)

        # 3. Hardware: Navigation & Illumination
        frame = self._apply_navigation(frame)
        frame = self._apply_illumination(frame)

        # 4. DSP: Parametric Signal Processing
        frame = self._apply_dsp(frame)

        # 5. Overlays: PiP and HUD
        if self.enable_pip:
            frame = self._apply_pip(frame, raw_frame)

        if self.enable_hud:
            frame = self._draw_hud(frame)

        return True, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _apply_nbi(self, frame):
        """Simulates Narrow Band Imaging (415nm & 540nm wavelengths)."""
        b, g, r = cv2.split(frame)
        # Red light penetrates deep and obscures surface vessels, so it is filtered out
        r_new = np.zeros_like(r)
        # Enhance Green and Blue to highlight capillaries
        g_new = cv2.addWeighted(g, 1.2, b, 0.2, 0)
        b_new = cv2.addWeighted(b, 1.2, g, 0.2, 0)
        return cv2.merge((b_new, g_new, r_new))

    def _apply_navigation(self, frame):
        if self.zoom_level <= 1.0:
            return frame

        new_h, new_w = int(self.h / self.zoom_level), int(self.w / self.zoom_level)
        center_y, center_x = self.h // 2 + self.pan_y, self.w // 2 + self.pan_x

        y1, y2 = max(0, center_y - new_h // 2), min(self.h, center_y - new_h // 2 + new_h)
        x1, x2 = max(0, center_x - new_w // 2), min(self.w, center_x - new_w // 2 + new_w)

        cropped = frame[y1:y2, x1:x2]
        return cv2.resize(cropped, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

    def _apply_illumination(self, frame):
        frame_float = frame.astype(np.float32)
        frame_float = frame_float * self._vignette_mask * self.illumination_intensity
        return np.clip(frame_float, 0, 255).astype(np.uint8)

    def _apply_dsp(self, frame):
        # Bilateral Denoising
        if self.denoise_strength > 0:
            sigma = self.denoise_strength
            frame = cv2.bilateralFilter(frame, d=9, sigmaColor=sigma, sigmaSpace=sigma)

        # Contrast Enhancement (CLAHE)
        if self.clahe_limit > 0.1:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clahe_limit, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            frame = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        # Sobel Texture Extraction
        if self.texture_opacity > 0.05:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            Ky = np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]], dtype=np.float32)

            Gx = cv2.filter2D(gray, cv2.CV_32F, Kx)
            Gy = cv2.filter2D(gray, cv2.CV_32F, Ky)
            magnitude = cv2.normalize(cv2.magnitude(Gx, Gy), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Blend the edge map over the original color frame
            edges_colored = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)
            frame = cv2.addWeighted(frame, 1.0 - self.texture_opacity, edges_colored, self.texture_opacity, 0)

        # Smart Anomaly Detection
        if self.anomaly_min_area < 4900: # Only run if slider is actively lowered
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.bitwise_or(cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])),
                                  cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255])))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > self.anomaly_min_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    cv2.putText(frame, f"ANOMALY ({int(area)}px)", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return frame

    def _apply_pip(self, frame, raw_frame):
        """Renders the raw hardware feed in the bottom right corner."""
        pip_h, pip_w = int(self.h * 0.25), int(self.w * 0.25)
        pip_resized = cv2.resize(raw_frame, (pip_w, pip_h))

        # Add white border
        cv2.rectangle(pip_resized, (0,0), (pip_w-1, pip_h-1), (255,255,255), 2)

        margin = 20
        frame[self.h-pip_h-margin : self.h-margin, self.w-pip_w-margin : self.w-margin] = pip_resized
        return frame

    def _draw_hud(self, frame):
        cx, cy = self.w // 2, self.h // 2
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)

        cv2.putText(frame, "REC", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.circle(frame, (100, 42), 8, (0, 0, 255), -1)
        cv2.putText(frame, f"ZOOM: {self.zoom_level:.1f}x", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"LIGHT: {int(self.illumination_intensity * 100)}%", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame