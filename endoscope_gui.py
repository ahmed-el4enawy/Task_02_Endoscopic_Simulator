"""
endoscope_gui.py
The main Tkinter interface for the Smart Endoscopic Assistance System.
Features drag-and-drop file support and fluid mouse tracking for navigation.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import datetime
from tkinterdnd2 import DND_FILES, TkinterDnD

from cv_engine import EndoscopicEngine

class EndoscopeGUI(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Intelligent Endoscopic Assistance System - Team 7")
        self.geometry("1500x850")

        self.colors = {
            'bg_main': '#0b0f19', 'bg_panel': '#161b22', 'bg_card': '#21262d',
            'accent_blue': '#58a6ff', 'accent_green': '#238636', 'accent_red': '#da3633',
            'text_primary': '#f0f6fc', 'text_secondary': '#8b949e', 'border': '#30363d'
        }
        self.configure(bg=self.colors['bg_main'])

        self.engine = EndoscopicEngine(image_path=None)

        # Mouse Tracking State
        self._drag_data = {"x": 0, "y": 0}

        self.create_ui()
        self.setup_drag_and_drop()
        self.bind_mouse_controls()
        self.update_video_stream()

    def create_ui(self):
        header = tk.Frame(self, bg=self.colors['bg_card'], height=70)
        header.pack(fill=tk.X, padx=20, pady=(20, 10))
        tk.Label(header, text="SMART ENDOSCOPE CONSOLE", font=('Segoe UI', 20, 'bold'),
                 bg=self.colors['bg_card'], fg=self.colors['text_primary']).pack(side=tk.LEFT, padx=20, pady=15)

        capture_btn = tk.Button(header, text="📷 CAPTURE IMAGE", command=self.capture_image,
                                bg=self.colors['accent_blue'], fg='white', font=('Segoe UI', 10, 'bold'),
                                relief=tk.FLAT, cursor='hand2', padx=15)
        capture_btn.pack(side=tk.RIGHT, padx=20, pady=15)

        main_container = tk.Frame(self, bg=self.colors['bg_main'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # --- LEFT PANEL ---
        left_panel = tk.Frame(main_container, bg=self.colors['bg_panel'], width=380, highlightbackground=self.colors['border'], highlightthickness=1)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # Load Source
        tk.Label(left_panel, text="INPUT SOURCE", font=('Segoe UI', 11, 'bold'), bg=self.colors['bg_panel'], fg=self.colors['text_secondary']).pack(anchor='w', padx=20, pady=(20, 5))
        load_btn = tk.Button(left_panel, text="📁 Browse or Drag & Drop Image", command=self.load_custom_image,
                             bg=self.colors['bg_card'], fg=self.colors['accent_blue'], font=('Segoe UI', 10, 'bold'), relief=tk.SOLID, bd=1, pady=8)
        load_btn.pack(fill=tk.X, padx=20, pady=5)
        tk.Frame(left_panel, height=1, bg=self.colors['border']).pack(fill=tk.X, padx=20, pady=15)

        # Hardware Control
        tk.Label(left_panel, text="HARDWARE ILLUMINATION", font=('Segoe UI', 11, 'bold'), bg=self.colors['bg_panel'], fg=self.colors['text_secondary']).pack(anchor='w', padx=20, pady=(0, 10))
        self.light_slider = ttk.Scale(left_panel, from_=0.0, to=3.0, value=1.0, orient=tk.HORIZONTAL, command=self.update_illumination)
        self.light_slider.pack(fill=tk.X, padx=20, pady=5)
        tk.Frame(left_panel, height=1, bg=self.colors['border']).pack(fill=tk.X, padx=20, pady=15)

        # Smart Processing Features
        tk.Label(left_panel, text="SMART PROCESSING", font=('Segoe UI', 11, 'bold'), bg=self.colors['bg_panel'], fg=self.colors['text_secondary']).pack(anchor='w', padx=20, pady=(0, 10))

        self.var_noise = tk.BooleanVar()
        tk.Checkbutton(left_panel, text="Edge-Preserving Denoise (Bilateral)", variable=self.var_noise, command=self.toggle_features, bg=self.colors['bg_panel'], fg=self.colors['text_primary'], selectcolor=self.colors['bg_card']).pack(anchor='w', padx=20, pady=5)

        self.var_clahe = tk.BooleanVar()
        tk.Checkbutton(left_panel, text="Medical Contrast Enhancement (CLAHE)", variable=self.var_clahe, command=self.toggle_features, bg=self.colors['bg_panel'], fg=self.colors['text_primary'], selectcolor=self.colors['bg_card']).pack(anchor='w', padx=20, pady=5)

        self.var_texture = tk.BooleanVar()
        tk.Checkbutton(left_panel, text="Extract Topological Texture (Sobel)", variable=self.var_texture, command=self.toggle_features, bg=self.colors['bg_panel'], fg=self.colors['text_primary'], selectcolor=self.colors['bg_card']).pack(anchor='w', padx=20, pady=5)

        self.var_anomaly = tk.BooleanVar()
        tk.Checkbutton(left_panel, text="Detect Anomalies (Bounding Boxes)", variable=self.var_anomaly, command=self.toggle_features, bg=self.colors['bg_panel'], fg=self.colors['accent_red'], selectcolor=self.colors['bg_card']).pack(anchor='w', padx=20, pady=5)
        tk.Frame(left_panel, height=1, bg=self.colors['border']).pack(fill=tk.X, padx=20, pady=15)

        # Navigation Instructions
        tk.Label(left_panel, text="TUBE NAVIGATION", font=('Segoe UI', 11, 'bold'), bg=self.colors['bg_panel'], fg=self.colors['text_secondary']).pack(anchor='w', padx=20, pady=(0, 10))
        tk.Label(left_panel, text="• Click & Drag to Steer\n• Mouse Scroll to Advance/Retract", font=('Segoe UI', 10), bg=self.colors['bg_panel'], fg=self.colors['text_primary'], justify=tk.LEFT).pack(anchor='w', padx=20)

        # --- RIGHT PANEL (Video Canvas) ---
        right_panel = tk.Frame(main_container, bg="black", highlightbackground=self.colors['border'], highlightthickness=1)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(right_panel, bg="black", cursor="fleur")
        self.video_label.pack(fill=tk.BOTH, expand=True)

    def setup_drag_and_drop(self):
        """Registers the main window to accept dropped files."""
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.handle_file_drop)

    def handle_file_drop(self, event):
        filepath = event.data
        if filepath.startswith('{') and filepath.endswith('}'):
            filepath = filepath[1:-1] # Clean up Windows paths
        self.engine.load_image(filepath)

    def bind_mouse_controls(self):
        self.video_label.bind("<ButtonPress-1>", self.start_drag)
        self.video_label.bind("<B1-Motion>", self.do_drag)

        # Mouse wheel support across platforms
        self.video_label.bind("<MouseWheel>", self.do_zoom)     # Windows/Mac
        self.video_label.bind("<Button-4>", self.do_zoom)       # Linux up
        self.video_label.bind("<Button-5>", self.do_zoom)       # Linux down

    def start_drag(self, event):
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def do_drag(self, event):
        # Calculate delta and invert it so dragging the mouse pulls the image
        dx = self._drag_data["x"] - event.x
        dy = self._drag_data["y"] - event.y
        self.engine.pan_x += int(dx * 0.5)
        self.engine.pan_y += int(dy * 0.5)

        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def do_zoom(self, event):
        # Handle Windows (+/- 120) and Linux (+/- 1)
        if event.num == 4 or getattr(event, 'delta', 0) > 0:
            dz = 0.2
        elif event.num == 5 or getattr(event, 'delta', 0) < 0:
            dz = -0.2
        else:
            dz = 0

        self.engine.zoom_level = max(1.0, min(8.0, self.engine.zoom_level + dz))
        if self.engine.zoom_level == 1.0:
            self.engine.pan_x, self.engine.pan_y = 0, 0

    def load_custom_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if filepath:
            self.engine.load_image(filepath)

    def update_illumination(self, val):
        self.engine.illumination_intensity = float(val)

    def toggle_features(self):
        self.engine.enable_noise_reduction = self.var_noise.get()
        self.engine.enable_clahe = self.var_clahe.get()
        self.engine.enable_texture = self.var_texture.get()
        self.engine.enable_anomaly_detection = self.var_anomaly.get()

    def update_video_stream(self):
        ret, frame_rgb = self.engine.get_frame()
        if ret:
            h, w = frame_rgb.shape[:2]
            target_h = 750
            target_w = int((target_h / h) * w)
            frame_resized = cv2.resize(frame_rgb, (target_w, target_h))

            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.after(30, self.update_video_stream)

    def capture_image(self):
        ret, frame_rgb = self.engine.get_frame()
        if ret:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            filename = f"capture_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(filename, frame_bgr)
            messagebox.showinfo("Image Captured", f"Successfully saved to {filename}")

if __name__ == "__main__":
    app = EndoscopeGUI()
    app.mainloop()