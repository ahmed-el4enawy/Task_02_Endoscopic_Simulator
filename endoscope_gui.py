"""
endoscope_gui.py
Modern Medical UI using CustomTkinter. Provides parametric control
over the physics engine and fluid mouse navigation for the simulation.
"""

import customtkinter as ctk
import cv2
from PIL import Image
import datetime
from tkinter import filedialog, messagebox

from cv_engine import EndoscopicEngine

# Set modern theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class EndoscopeGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Intelligent Endoscopic Console - Team 7")
        self.geometry("1500x900")

        self.engine = EndoscopicEngine(image_path=None)
        self._drag_data = {"x": 0, "y": 0}

        self.create_ui()
        self.bind_mouse_controls()
        self.update_video_stream()

    def create_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- LEFT PANEL (Controls) ---
        self.sidebar = ctk.CTkFrame(self, width=380, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        # Title
        ctk.CTkLabel(self.sidebar, text="SMART ENDOSCOPE", font=ctk.CTkFont(size=22, weight="bold")).pack(pady=(20, 5), padx=20, anchor="w")
        ctk.CTkLabel(self.sidebar, text="SBE 3220 System Console", text_color="gray").pack(pady=(0, 20), padx=20, anchor="w")

        # Source Button
        self.btn_load = ctk.CTkButton(self.sidebar, text="Load Patient Data", command=self.load_image, height=40)
        self.btn_load.pack(pady=10, padx=20, fill="x")

        self.btn_capture = ctk.CTkButton(self.sidebar, text="Capture Image", command=self.capture_image, height=40, fg_color="#238636", hover_color="#2ea043")
        self.btn_capture.pack(pady=(0, 20), padx=20, fill="x")

        # Optical Physics Toggles
        ctk.CTkLabel(self.sidebar, text="OPTICAL MODES", font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(10, 5), padx=20, anchor="w")

        self.sw_nbi = ctk.CTkSwitch(self.sidebar, text="Narrow Band Imaging (NBI)", command=self.update_toggles)
        self.sw_nbi.pack(pady=5, padx=20, anchor="w")

        self.sw_fisheye = ctk.CTkSwitch(self.sidebar, text="Lens Distortion (Fisheye)", command=self.update_toggles)
        self.sw_fisheye.pack(pady=5, padx=20, anchor="w")

        self.sw_pip = ctk.CTkSwitch(self.sidebar, text="Raw Hardware Feed (PiP)", command=self.update_toggles)
        self.sw_pip.pack(pady=5, padx=20, anchor="w")

        # Parametric Sliders
        ctk.CTkLabel(self.sidebar, text="PARAMETRIC DSP CONTROL", font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(25, 5), padx=20, anchor="w")

        self.create_slider("Light Intensity", 0.1, 3.0, 1.0, self.update_illumination)
        self.create_slider("Edge-Preserving Denoise", 0, 150, 0, self.update_denoise)
        self.create_slider("CLAHE Contrast Limit", 0.0, 5.0, 0.0, self.update_clahe)
        self.create_slider("Sobel Texture Overlay", 0.0, 1.0, 0.0, self.update_texture)
        self.create_slider("Anomaly Sensitivity", 100, 5000, 5000, self.update_anomaly, reverse=True)

        # Instructions
        nav_text = "Navigation:\n• Click & Drag to Steer\n• Mouse Wheel to Zoom"
        ctk.CTkLabel(self.sidebar, text=nav_text, text_color="gray", justify="left").pack(side="bottom", pady=20, padx=20, anchor="w")

        # --- RIGHT PANEL (Video Canvas) ---
        self.canvas_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="black")
        self.canvas_frame.grid(row=0, column=1, sticky="nsew")

        self.video_label = ctk.CTkLabel(self.canvas_frame, text="", cursor="fleur")
        self.video_label.pack(expand=True, fill="both")

    def create_slider(self, text, from_, to, default, command, reverse=False):
        frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        frame.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(frame, text=text, font=ctk.CTkFont(size=12)).pack(anchor="w")
        slider = ctk.CTkSlider(frame, from_=from_, to=to, command=command)
        if reverse:
            slider.set(to) # High value means low sensitivity
        else:
            slider.set(default)
        slider.pack(fill="x", pady=(5,0))

    def bind_mouse_controls(self):
        self.video_label.bind("<ButtonPress-1>", self.start_drag)
        self.video_label.bind("<B1-Motion>", self.do_drag)
        self.video_label.bind("<MouseWheel>", self.do_zoom)
        self.video_label.bind("<Button-4>", self.do_zoom)
        self.video_label.bind("<Button-5>", self.do_zoom)

    def start_drag(self, event):
        self._drag_data["x"], self._drag_data["y"] = event.x, event.y

    def do_drag(self, event):
        dx, dy = self._drag_data["x"] - event.x, self._drag_data["y"] - event.y
        self.engine.pan_x += int(dx * 0.4)
        self.engine.pan_y += int(dy * 0.4)
        self._drag_data["x"], self._drag_data["y"] = event.x, event.y

    def do_zoom(self, event):
        dz = 0.2 if (event.num == 4 or getattr(event, 'delta', 0) > 0) else -0.2
        self.engine.zoom_level = max(1.0, min(8.0, self.engine.zoom_level + dz))
        if self.engine.zoom_level == 1.0:
            self.engine.pan_x, self.engine.pan_y = 0, 0

    def load_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.bmp")])
        if filepath:
            self.engine.load_image(filepath)

    def update_toggles(self):
        self.engine.enable_nbi = self.sw_nbi.get()
        self.engine.enable_fisheye = self.sw_fisheye.get()
        self.engine.enable_pip = self.sw_pip.get()

    def update_illumination(self, val): self.engine.illumination_intensity = float(val)
    def update_denoise(self, val): self.engine.denoise_strength = int(val)
    def update_clahe(self, val): self.engine.clahe_limit = float(val)
    def update_texture(self, val): self.engine.texture_opacity = float(val)
    def update_anomaly(self, val): self.engine.anomaly_min_area = int(val)

    def update_video_stream(self):
        ret, frame_rgb = self.engine.get_frame()
        if ret:
            # Fit frame to window
            target_h = self.canvas_frame.winfo_height()
            if target_h > 100: # Ensure window is initialized
                target_w = int((target_h / frame_rgb.shape[0]) * frame_rgb.shape[1])
                frame_resized = cv2.resize(frame_rgb, (target_w, target_h))

                # Convert to CTkImage format
                pil_image = Image.fromarray(frame_resized)
                ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(target_w, target_h))

                self.video_label.configure(image=ctk_image)
                self.video_label.image = ctk_image

        self.after(30, self.update_video_stream)

    def capture_image(self):
        ret, frame_rgb = self.engine.get_frame()
        if ret:
            filename = f"capture_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(filename, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            messagebox.showinfo("Saved", f"High-Res frame saved to {filename}")

if __name__ == "__main__":
    app = EndoscopeGUI()
    app.mainloop()