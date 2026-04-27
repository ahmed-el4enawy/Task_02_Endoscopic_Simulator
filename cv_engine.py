"""
cv_engine.py
Core Computer Vision engine for the Endoscopic Simulator.
Handles static image loading, illumination simulation, navigation (digital pan/zoom),
and smart processing (CLAHE, Noise Reduction, Feature Extraction).
"""

import cv2
import numpy as np
import os

class EndoscopicEngine:
    def __init__(self, image_path="assets/sample_image.jpg"):
        # Load the static image
        self.base_image = cv2.imread(image_path)

        # Fallback if the image isn't placed in the folder yet
        if self.base_image is None:
            self.base_image = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(self.base_image, "Image not found. Place a sample in the assets folder.",
                        (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Hardware Simulation States
        self.illumination_intensity = 1.0
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0

        # Smart Features Flags (Bonus)
        self.enable_noise_reduction = False
        self.enable_clahe = False
        self.enable_feature_extraction = False

    def get_frame(self):
        """Processes the static image based on current GUI parameters."""
        # Work on a fresh copy of the base image every tick
        frame = self.base_image.copy()

        # 1. Simulate Insertion Tube Navigation (Digital Zoom & Pan)
        frame = self._apply_navigation(frame)

        # 2. Simulate Illumination System
        frame = self._apply_illumination(frame)

        # 3. Apply Smart Processing (Bonus)
        if self.enable_noise_reduction:
            # Gaussian blur for sensor noise reduction
            frame = cv2.GaussianBlur(frame, (5, 5), 0)

        if self.enable_clahe:
            # Contrast Limited Adaptive Histogram Equalization
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            limg = cv2.merge((cl, a, b))
            frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        if self.enable_feature_extraction:
            # Extract Shape/Texture Feature
            frame = self._extract_texture_feature(frame)

        # Convert BGR (OpenCV) to RGB (Tkinter/Pillow requirement)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return True, frame_rgb

    def _apply_navigation(self, frame):
        """Simulates physical movement using digital cropping and resizing."""
        if self.zoom_level <= 1.0:
            return frame

        h, w = frame.shape[:2]
        new_h, new_w = int(h / self.zoom_level), int(w / self.zoom_level)

        # Calculate boundaries with pan offsets
        center_y, center_x = h // 2 + self.pan_y, w // 2 + self.pan_x

        # Clamp boundaries to prevent crashing out of bounds
        y1 = max(0, center_y - new_h // 2)
        y2 = min(h, y1 + new_h)
        x1 = max(0, center_x - new_w // 2)
        x2 = min(w, x1 + new_w)

        cropped = frame[y1:y2, x1:x2]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    def _apply_illumination(self, frame):
        """Simulates light source intensity control."""
        if self.illumination_intensity == 1.0:
            return frame

        # Convert to HSV to safely adjust the Value (Brightness) channel
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Apply intensity multiplier and clip to 255 max
        v = np.clip(v * self.illumination_intensity, 0, 255).astype(np.uint8)

        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    def _extract_texture_feature(self, frame):
        """Texture/Shape Feature Extraction: Implemented from scratch using explicit Sobel kernels."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Define explicit Sobel kernels mathematically from scratch
        Kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)

        Ky = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=np.float32)

        # 2. Convolve kernels with the image
        Gx = cv2.filter2D(gray, cv2.CV_32F, Kx)
        Gy = cv2.filter2D(gray, cv2.CV_32F, Ky)

        # 3. Calculate gradient magnitude
        magnitude = cv2.magnitude(Gx, Gy)

        # Normalize back to 8-bit image range for display
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Convert back to 3-channel so it renders properly in the GUI pipeline
        return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)