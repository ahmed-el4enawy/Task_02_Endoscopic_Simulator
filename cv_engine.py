"""
cv_engine.py
Advanced Computer Vision engine for the Endoscopic Simulator.
Implements Fisheye distortion, Radial Illumination, Bilateral Noise Reduction,
and multi-feature anomaly detection (Bounding Boxes).
"""

import cv2
import numpy as np
import os

class EndoscopicEngine:
    def __init__(self, image_path=None):
        self.base_image = None
        self.h, self.w = 720, 1280 # Default dimensions

        # Cached physics masks for performance
        self._vignette_mask = None

        self.load_image(image_path)

        # Hardware Simulation States
        self.illumination_intensity = 1.0
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0

        # Smart Features Flags
        self.enable_noise_reduction = False
        self.enable_clahe = False
        self.enable_texture = False
        self.enable_anomaly_detection = False
        self.enable_hud = True

    def load_image(self, filepath):
        """Loads a new image and resets navigation states."""
        if filepath and os.path.exists(filepath):
            self.base_image = cv2.imread(filepath)
        else:
            # Fallback blank screen
            self.base_image = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(self.base_image, "NO VIDEO SOURCE.",
                        (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(self.base_image, "Drag & Drop an image, or use the Load button.",
                        (300, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        self.h, self.w = self.base_image.shape[:2]
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._generate_vignette_mask()

    def _generate_vignette_mask(self):
        """Pre-calculates a radial gradient mask to simulate fiber-optic spotlight."""
        X = np.linspace(-1, 1, self.w)
        Y = np.linspace(-1, 1, self.h)
        x, y = np.meshgrid(X, Y)
        radius = np.sqrt(x**2 + y**2)

        # Gaussian falloff for smooth lighting drop at the edges
        self._vignette_mask = np.exp(-(radius**2) / 0.8)
        # Expand to 3 channels for fast OpenCV multiplication
        self._vignette_mask = np.stack([self._vignette_mask]*3, axis=-1)

    def get_frame(self):
        """Pipeline: Navigation -> Physics -> Processing -> HUD."""
        frame = self.base_image.copy()

        # 1. Navigation (Digital Zoom & Pan)
        frame = self._apply_navigation(frame)

        # 2. Physics: Illumination & Spotlight Simulation
        frame = self._apply_illumination(frame)

        # 3. Medical Processing
        if self.enable_noise_reduction:
            # Bilateral Filter: Preserves blood vessel edges while smoothing tissue noise
            frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

        if self.enable_clahe:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            frame = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        if self.enable_texture:
            frame = self._extract_texture_feature(frame)

        if self.enable_anomaly_detection:
            frame = self._detect_anomalies(frame)

        # 4. Heads-Up Display (HUD)
        if self.enable_hud:
            frame = self._draw_hud(frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return True, frame_rgb

    def _apply_navigation(self, frame):
        if self.zoom_level <= 1.0:
            return frame

        new_h, new_w = int(self.h / self.zoom_level), int(self.w / self.zoom_level)
        center_y, center_x = self.h // 2 + self.pan_y, self.w // 2 + self.pan_x

        y1 = max(0, center_y - new_h // 2)
        y2 = min(self.h, y1 + new_h)
        x1 = max(0, center_x - new_w // 2)
        x2 = min(self.w, x1 + new_w)

        cropped = frame[y1:y2, x1:x2]
        return cv2.resize(cropped, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

    def _apply_illumination(self, frame):
        """Applies radial light falloff and intensity multipliers."""
        # Convert to float for safe multiplication
        frame_float = frame.astype(np.float32)

        # Apply fiber-optic radial spotlight
        frame_float = frame_float * self._vignette_mask

        # Apply overall hardware intensity
        frame_float = frame_float * self.illumination_intensity

        return np.clip(frame_float, 0, 255).astype(np.uint8)

    def _extract_texture_feature(self, frame):
        """From-scratch Sobel implementation for topological texture."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        Ky = np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]], dtype=np.float32)

        Gx = cv2.filter2D(gray, cv2.CV_32F, Kx)
        Gy = cv2.filter2D(gray, cv2.CV_32F, Ky)
        magnitude = cv2.magnitude(Gx, Gy)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)

    def _detect_anomalies(self, frame):
        """Smart Fusion: Detects abnormal redness and draws bounding boxes with area stats."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                              cv2.inRange(hsv, lower_red2, upper_red2))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 800: # Filter small artifacts
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

                label = f"ANOMALY DETECTED ({int(area)}px)"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame

    def _draw_hud(self, frame):
        """Draws commercial-grade medical overlays."""
        # Crosshair
        cx, cy = self.w // 2, self.h // 2
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)
        cv2.circle(frame, (cx, cy), 15, (0, 255, 0), 1)

        # Telemetry Data
        cv2.putText(frame, "REC", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.circle(frame, (100, 42), 8, (0, 0, 255), -1)

        cv2.putText(frame, f"ZOOM: {self.zoom_level:.1f}x", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"LIGHT: {int(self.illumination_intensity * 100)}%", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame