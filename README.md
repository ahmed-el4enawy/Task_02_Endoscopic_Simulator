# Intelligent Endoscopic Assistance System

## Executive Summary

This project presents an advanced, high-fidelity software simulation of an **Intelligent Endoscopic Console**. Developed for the Medical Equipment II (SBE 3220) course, the system demonstrates the intersection of medical optics, hardware control, and intelligent computer vision. By simulating the physical properties of a fiber-optic endoscope—including radial illumination falloff, wide-angle lens distortion, and Narrow Band Imaging (NBI)—this "Digital Twin" provides a comprehensive platform for clinical feature extraction and pathological anomaly detection.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Installation](#installation)
5. [Usage Instructions](#usage-instructions)
6. [Technical Specifications](#technical-specifications)
7. [File Descriptions](#file-descriptions)
8. [Advanced DSP Logic](#advanced-dsp-logic)

---

## Project Overview

### Background
Endoscopy is a critical diagnostic procedure used to visualize internal cavities. Modern "Smart" endoscopes go beyond simple imaging; they incorporate real-time image processing to enhance tissue contrast and automatically highlight potential lesions, polyps, or bleeding areas.

### Problem Statement
Understanding the interaction between endoscopic hardware (light source intensity, lens curvature) and the resulting diagnostic image requires a controlled environment. Standard imaging software lacks the specialized medical physics—such as radial light falloff and specific wavelength filtering (NBI)—necessary to simulate a real clinical setting.

### Solution
This project implements a professional **Medical Console** built with Python and CustomTkinter. It simulates the four core subsystems of an endoscope (Illumination, Imaging, Navigation, and Processing) using high-resolution patient data. The system features a custom-built Computer Vision engine that processes data with medical-grade algorithms, including from-scratch matrix convolutions for texture analysis.

---

## Key Features

### Advanced Optical Physics Simulation
- **Radial Spotlight (Vignette)**: Models the inverse-square law of light falloff from fiber-optic bundles, creating a realistic central illumination with peripheral shadowing.
- **Fisheye Lens Distortion**: Mathematically warps the viewing field using barrel distortion meshgrids to simulate the wide-angle lenses used in clinical scopes. 
- **Narrow Band Imaging (NBI)**: Simulates the filtering of red light (long wavelengths) to enhance the visualization of surface capillaries and mucosal patterns using blue and green wavelengths. 

### Intelligent Clinical Assistance (Bonus)
- **Edge-Preserving Denoising**: Implements Bilateral Filtering to smooth sensor noise while strictly preserving the sharp boundaries of blood vessels.
- **Topological Texture Overlay**: Uses a custom **Sobel Matrix Convolution** ($3 \times 3$ kernels) to extract and overlay tissue texture on the live feed.
- **Anomaly Detection**: A real-time computer vision hunt for abnormal redness (bleeding/inflammation) and polyps, utilizing contour area analysis and bounding box overlays.

### High-End Medical GUI
- **Premium Teal-Green Theme**: Implements a high-contrast `#004F52` medical palette designed for long-duration surgical focus.
- **Parametric Sliders**: Advanced controls with real-time numerical readouts for precise tuning of DSP algorithms.
- **Fluid Mouse Navigation**: Click-and-drag steering with scroll-wheel advancement to simulate the physical manipulation of the insertion tube.

---

## System Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│              Premium Medical Console (CustomTkinter)            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  Parametric  │  │   Optical    │  │   Live Diagnostic    │   │
│  │   Sliders    │  │    Modes     │  │    Visual Canvas     │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                Endoscopic Computer Vision Engine                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Optical Physics Layer                                  │    │
│  │  - Radial Illumination Mask (Gaussian Falloff)          │    │
│  │  - Barrel Distortion Remapping (Fisheye)                │    │
│  │  - NBI Wavelength Transformation (R-channel Suppression) │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Intelligent Processing Layer                           │    │
│  │  - Bilateral Filtering (Edge-Preserving Denoise)        │    │
│  │  - From-Scratch Sobel Convolution (Texture Extraction)  │    │
│  │  - HSV Contour Analysis (Anomaly Detection)             │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
```

---

## Installation

### Prerequisites
* Python 3.11 or higher
* pip package manager

### Required Libraries
```bash
pip install numpy>=1.19.0
pip install opencv-python>=4.5.0
pip install Pillow>=9.0.0
pip install customtkinter>=5.2.0
```

---

## Usage Instructions

### Starting the Console
```bash
python endoscope_gui.py
```

### Simulation Procedures
1. **Load Data**: Use the "📁 Load Patient Data" button to select an endoscopic image from the `assets/` folder.
2. **Navigate**: 
   - **Scroll Mouse**: Simulates pushing the insertion tube forward (Zoom) or pulling back.
   - **Click & Drag**: Simulates steering the tip of the scope to scan the tissue.
3. **Enhance**: 
   - Toggle **NBI** to highlight superficial blood vessels.
   - Adjust the **CLAHE** slider to improve visibility in dark areas.
   - Use the **Sobel Overlay** to analyze tissue topology.
4. **Diagnostics**:
   - Lower the **Anomaly Sensitivity** slider to trigger automated bounding boxes around potential pathological areas.
   - Click **Capture** to save a timestamped clinical report image.
5. **Reset**: Click **↺ Reset Parameters** to instantly return all hardware and software states to baseline.

---

## Technical Specifications

### Physics & Optical Parameters

| Parameter | Symbol | Mechanism |
| --- | --- | --- |
| Illumination Falloff | $I(r)$ | Gaussian Radial Gradient |
| Lens Curvature | $\theta$ | $r + k \cdot r^3$ (Barrel Model) |
| NBI Green Gain | $G_{nbi}$ | $1.2 \times G + 0.2 \times B$ |
| NBI Blue Gain | $B_{nbi}$ | $1.2 \times B + 0.2 \times G$ |
| Denoising | $Bilateral$ | Spatial $\sigma$ & Range $\sigma$ fusion |

### From-Scratch Sobel Kernels
The system performs manual convolution using the following discrete differentiation operators:

$$K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad K_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

---

## File Descriptions

* **`cv_engine.py`**: The mathematical core. Contains the pre-calculated meshgrids for distortion, the radial mask generator, and the clinical processing pipeline.
* **`endoscope_gui.py`**: The CustomTkinter interface. Manages the modern card layout, parametric slider synchronization, and mouse-event navigation logic.
* **`requirements.txt`**: List of dependencies.
* **`assets/`**: Repository for simulated patient images.

---

## Acknowledgments

**Institution**: Cairo University, Faculty of Engineering
**Department**: Systems & Biomedical Engineering
**Course**: Medical Equipment II (SBE 3220)
**Supervisor**: Dr. Sherif H. El-Gohary

**Team 7 Members**:
* Ahmed Salah Geoshy Elshenawy
* Ahmed Ahmed Mokhtar
* Osama Magdy Ali Khalifa
* Mohamed Hamdy Abdelhamed
* Mennat Allah Khalifa

---
**Document Last Updated**: April 2026