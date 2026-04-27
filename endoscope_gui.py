"""
endoscope_gui.py
Premium Medical UI using CustomTkinter. Features a custom teal-green
accent palette, advanced dynamic sliders, and quick-action controls.
"""

import customtkinter as ctk
import cv2
from PIL import Image
import datetime
from tkinter import filedialog, messagebox

from cv_engine import EndoscopicEngine

ctk.set_appearance_mode("dark")

class EndoscopeGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Intelligent Endoscopic Console - Team 7")
        self.geometry("1600x950")

        # Premium Medical Palette
        self.theme = {
            "bg_main": "#0D1117",
            "bg_card": "#161B22",
            "accent": "#004F52",
            "accent_hover": "#006F73",
            "border": "#30363D",
            "text_main": "#F0F6FC",
            "text_dim": "#8B949E",
            "danger": "#DA3633"
        }
        self.configure(fg_color=self.theme["bg_main"])

        self.engine = EndoscopicEngine(image_path=None)
        self._drag_data = {"x": 0, "y": 0}

        self.create_ui()
        self.bind_mouse_controls()
        self.update_video_stream()

    def create_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- LEFT PANEL (Control Sidebar) ---
        self.sidebar = ctk.CTkFrame(self, width=420, corner_radius=0, fg_color=self.theme["bg_main"], border_width=1, border_color=self.theme["border"])
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        # Header Area
        header_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        header_frame.pack(pady=(25, 15), padx=20, fill="x")
        ctk.CTkLabel(header_frame, text="SMART ENDOSCOPE", font=ctk.CTkFont(size=24, weight="bold"), text_color=self.theme["text_main"]).pack(anchor="w")
        ctk.CTkLabel(header_frame, text="SBE 3220 Clinical Simulator", font=ctk.CTkFont(size=14), text_color=self.theme["accent_hover"]).pack(anchor="w")

        # Top Buttons Card
        btn_card = self.create_ui_card(self.sidebar)
        self.btn_load = ctk.CTkButton(btn_card, text="📁 Load Patient Data", command=self.load_image, height=45,
                                      fg_color=self.theme["accent"], hover_color=self.theme["accent_hover"], font=ctk.CTkFont(weight="bold"))
        self.btn_load.pack(pady=(0, 10), fill="x")

        self.btn_capture = ctk.CTkButton(btn_card, text="📷 Capture High-Res Image", command=self.capture_image, height=45,
                                         fg_color="transparent", border_width=2, border_color=self.theme["accent"], hover_color="#1b222b", text_color=self.theme["text_main"])
        self.btn_capture.pack(pady=(0, 10), fill="x")

        # Quick Actions Row
        quick_actions = ctk.CTkFrame(btn_card, fg_color="transparent")
        quick_actions.pack(fill="x")

        self.btn_reset = ctk.CTkButton(quick_actions, text="↺ Reset Parameters", command=self.reset_features, height=35,
                                       fg_color=self.theme["border"], hover_color=self.theme["danger"])
        self.btn_reset.pack(side="left", expand=True, padx=(0, 5), fill="x")

        self.btn_hud = ctk.CTkButton(quick_actions, text="👁 Hide HUD", command=self.toggle_hud, height=35,
                                     fg_color=self.theme["border"], hover_color=self.theme["accent_hover"])
        self.btn_hud.pack(side="right", expand=True, padx=(5, 0), fill="x")

        # Optical Modes Card
        optics_card = self.create_ui_card(self.sidebar, title="OPTICAL PHYSICS MODES")
        self.sw_nbi = ctk.CTkSwitch(optics_card, text="Narrow Band Imaging (NBI)", command=self.update_toggles, progress_color=self.theme["accent"])
        self.sw_nbi.pack(pady=(0, 10), anchor="w")

        self.sw_fisheye = ctk.CTkSwitch(optics_card, text="Lens Distortion (Fisheye)", command=self.update_toggles, progress_color=self.theme["accent"])
        self.sw_fisheye.pack(pady=(0, 10), anchor="w")

        self.sw_pip = ctk.CTkSwitch(optics_card, text="Raw Hardware Feed (PiP)", command=self.update_toggles, progress_color=self.theme["accent"])
        self.sw_pip.pack(anchor="w")

        # Parametric DSP Card
        dsp_card = self.create_ui_card(self.sidebar, title="PARAMETRIC DSP CONTROL")
        self.sl_light, self.lbl_light = self.create_advanced_slider(dsp_card, "Light Intensity", 0.1, 3.0, 1.0, self.update_illumination, "{:.1f}x")
        self.sl_denoise, self.lbl_denoise = self.create_advanced_slider(dsp_card, "Edge-Preserving Denoise", 0, 150, 0, self.update_denoise, "{:.0f}")
        self.sl_clahe, self.lbl_clahe = self.create_advanced_slider(dsp_card, "CLAHE Contrast Limit", 0.0, 5.0, 0.0, self.update_clahe, "{:.1f}")
        self.sl_texture, self.lbl_texture = self.create_advanced_slider(dsp_card, "Sobel Texture Overlay", 0.0, 1.0, 0.0, self.update_texture, "{:.2f}")
        self.sl_anomaly, self.lbl_anomaly = self.create_advanced_slider(dsp_card, "Anomaly Sensitivity (px)", 100, 5000, 5000, self.update_anomaly, "{:.0f}", reverse=True)

        # Instructions Footer
        nav_text = "Navigation Instructions:\n• Click & Drag to steer the insertion tube.\n• Mouse Wheel to advance/retract (Zoom)."
        ctk.CTkLabel(self.sidebar, text=nav_text, text_color=self.theme["text_dim"], justify="left").pack(side="bottom", pady=20, padx=20, anchor="w")

        # --- RIGHT PANEL (Video Canvas) ---
        self.canvas_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="black")
        self.canvas_frame.grid(row=0, column=1, sticky="nsew")

        self.video_label = ctk.CTkLabel(self.canvas_frame, text="", cursor="crosshair")
        self.video_label.pack(expand=True, fill="both")

    def create_ui_card(self, parent, title=None):
        card = ctk.CTkFrame(parent, fg_color=self.theme["bg_card"], corner_radius=8)
        card.pack(fill="x", padx=20, pady=(0, 15))

        inner_frame = ctk.CTkFrame(card, fg_color="transparent")
        inner_frame.pack(fill="both", expand=True, padx=15, pady=15)

        if title:
            ctk.CTkLabel(inner_frame, text=title, font=ctk.CTkFont(size=12, weight="bold"), text_color=self.theme["text_dim"]).pack(anchor="w", pady=(0, 10))

        return inner_frame

    def create_advanced_slider(self, parent, text, from_, to, default, command, format_str="{:.2f}", reverse=False):
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(fill="x", pady=(0, 10))

        header = ctk.CTkFrame(container, fg_color="transparent")
        header.pack(fill="x")
        ctk.CTkLabel(header, text=text, font=ctk.CTkFont(size=13)).pack(side="left")

        value_label = ctk.CTkLabel(header, text=format_str.format(default), font=ctk.CTkFont(size=13, weight="bold"), text_color=self.theme["accent"])
        value_label.pack(side="right")

        def slider_callback(val):
            value_label.configure(text=format_str.format(val))
            command(val)

        slider = ctk.CTkSlider(container, from_=from_, to=to, command=slider_callback,
                               button_color=self.theme["accent"], button_hover_color=self.theme["accent_hover"], progress_color=self.theme["accent"])
        slider.set(to if reverse else default)
        slider.pack(fill="x", pady=(5,0))

        return slider, value_label

    def reset_features(self):
        """Clears all engine parameters and resets the UI sliders & switches."""
        self.engine.reset_parameters()

        self.sw_nbi.deselect()
        self.sw_fisheye.deselect()
        self.sw_pip.deselect()

        self.sl_light.set(1.0); self.lbl_light.configure(text="{:.1f}x".format(1.0))
        self.sl_denoise.set(0); self.lbl_denoise.configure(text="{:.0f}".format(0))
        self.sl_clahe.set(0.0); self.lbl_clahe.configure(text="{:.1f}".format(0.0))
        self.sl_texture.set(0.0); self.lbl_texture.configure(text="{:.2f}".format(0.0))
        self.sl_anomaly.set(5000); self.lbl_anomaly.configure(text="{:.0f}".format(5000))

    def toggle_hud(self):
        """Toggles the telemetry overlay and updates button text."""
        self.engine.enable_hud = not self.engine.enable_hud
        if self.engine.enable_hud:
            self.btn_hud.configure(text="👁 Hide HUD")
        else:
            self.btn_hud.configure(text="👁 Show HUD")

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
            target_h = self.canvas_frame.winfo_height()
            if target_h > 100:
                target_w = int((target_h / frame_rgb.shape[0]) * frame_rgb.shape[1])
                frame_resized = cv2.resize(frame_rgb, (target_w, target_h))

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