"""
endoscope_gui.py
The main Tkinter interface for the Smart Endoscopic Assistance System.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import datetime
from cv_engine import EndoscopicEngine

class EndoscopeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent Endoscopic Assistance System - Team 7")
        self.root.geometry("1400x850")

        # Premium Dark Theme Palette
        self.colors = {
            'bg_main': '#0b0f19',
            'bg_panel': '#161b22',
            'bg_card': '#21262d',
            'accent_blue': '#58a6ff',
            'accent_green': '#238636',
            'text_primary': '#f0f6fc',
            'text_secondary': '#8b949e',
            'border': '#30363d'
        }
        self.root.configure(bg=self.colors['bg_main'])

        # Initialize the Engine (Using 0 for webcam by default; swap for video file path)
        self.engine = EndoscopicEngine(image_path="assets/sample_image.jpg")

        self.create_ui()
        self.update_video_stream()

        # Bind keyboard events for Navigation (Insertion Tube simulation)
        self.root.bind("<Up>", lambda e: self.navigate(0, -10))
        self.root.bind("<Down>", lambda e: self.navigate(0, 10))
        self.root.bind("<Left>", lambda e: self.navigate(-10, 0))
        self.root.bind("<Right>", lambda e: self.navigate(10, 0))
        self.root.bind("+", lambda e: self.zoom(0.1))
        self.root.bind("-", lambda e: self.zoom(-0.1))

    def create_ui(self):
        # --- HEADER ---
        header = tk.Frame(self.root, bg=self.colors['bg_card'], height=70)
        header.pack(fill=tk.X, padx=20, pady=(20, 10))
        tk.Label(header, text="SMART ENDOSCOPE CONSOLE", font=('Segoe UI', 20, 'bold'),
                 bg=self.colors['bg_card'], fg=self.colors['text_primary']).pack(side=tk.LEFT, padx=20, pady=15)

        # Capture Button
        capture_btn = tk.Button(header, text="📷 CAPTURE IMAGE", command=self.capture_image,
                                bg=self.colors['accent_blue'], fg='white', font=('Segoe UI', 10, 'bold'),
                                relief=tk.FLAT, cursor='hand2', padx=15)
        capture_btn.pack(side=tk.RIGHT, padx=20, pady=15)

        # --- MAIN CONTAINER ---
        main_container = tk.Frame(self.root, bg=self.colors['bg_main'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # --- LEFT PANEL (Controls) ---
        left_panel = tk.Frame(main_container, bg=self.colors['bg_panel'], width=350, highlightbackground=self.colors['border'], highlightthickness=1)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # 1. Illumination System
        tk.Label(left_panel, text="ILLUMINATION SYSTEM", font=('Segoe UI', 11, 'bold'), bg=self.colors['bg_panel'], fg=self.colors['text_secondary']).pack(anchor='w', padx=20, pady=(20, 10))
        self.light_slider = ttk.Scale(left_panel, from_=0.2, to=3.0, value=1.0, orient=tk.HORIZONTAL, command=self.update_illumination)
        self.light_slider.pack(fill=tk.X, padx=20, pady=5)

        # Separator
        tk.Frame(left_panel, height=1, bg=self.colors['border']).pack(fill=tk.X, padx=20, pady=20)

        # 2. Smart Features (Bonus)
        tk.Label(left_panel, text="SMART PROCESSING", font=('Segoe UI', 11, 'bold'), bg=self.colors['bg_panel'], fg=self.colors['text_secondary']).pack(anchor='w', padx=20, pady=(0, 10))

        self.var_noise = tk.BooleanVar()
        tk.Checkbutton(left_panel, text="Enable Noise Reduction", variable=self.var_noise, command=self.toggle_features,
                       bg=self.colors['bg_panel'], fg=self.colors['text_primary'], selectcolor=self.colors['bg_card']).pack(anchor='w', padx=20, pady=5)

        self.var_clahe = tk.BooleanVar()
        tk.Checkbutton(left_panel, text="Enable Contrast Enhancement (CLAHE)", variable=self.var_clahe, command=self.toggle_features,
                       bg=self.colors['bg_panel'], fg=self.colors['text_primary'], selectcolor=self.colors['bg_card']).pack(anchor='w', padx=20, pady=5)

        self.var_feature = tk.BooleanVar()
        tk.Checkbutton(left_panel, text="Detect Abnormalities (Color Feature)", variable=self.var_feature, command=self.toggle_features,
                       bg=self.colors['bg_panel'], fg=self.colors['text_primary'], selectcolor=self.colors['bg_card']).pack(anchor='w', padx=20, pady=5)

        # Separator
        tk.Frame(left_panel, height=1, bg=self.colors['border']).pack(fill=tk.X, padx=20, pady=20)

        # Navigation Instructions
        tk.Label(left_panel, text="NAVIGATION CONTROLS", font=('Segoe UI', 11, 'bold'), bg=self.colors['bg_panel'], fg=self.colors['text_secondary']).pack(anchor='w', padx=20, pady=(0, 10))
        instructions = "• Press [+] to Zoom In (Advance)\n• Press [-] to Zoom Out (Retract)\n• Arrow Keys to Pan"
        tk.Label(left_panel, text=instructions, font=('Segoe UI', 10), bg=self.colors['bg_panel'], fg=self.colors['text_primary'], justify=tk.LEFT).pack(anchor='w', padx=20)


        # --- RIGHT PANEL (Video Feed) ---
        right_panel = tk.Frame(main_container, bg="black", highlightbackground=self.colors['border'], highlightthickness=1)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(right_panel, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)

    def update_illumination(self, val):
        self.engine.illumination_intensity = float(val)

    def toggle_features(self):
        self.engine.enable_noise_reduction = self.var_noise.get()
        self.engine.enable_clahe = self.var_clahe.get()
        self.engine.enable_feature_extraction = self.var_feature.get()

    def navigate(self, dx, dy):
        """Simulate joystick/insertion tube movement"""
        self.engine.pan_x += dx
        self.engine.pan_y += dy

    def zoom(self, dz):
        """Simulate advancing/retracting the tube"""
        self.engine.zoom_level = max(1.0, min(5.0, self.engine.zoom_level + dz))
        if self.engine.zoom_level == 1.0:
            self.engine.pan_x, self.engine.pan_y = 0, 0 # Reset pan when fully zoomed out

    def update_video_stream(self):
        """Main loop to update the video frame on the Tkinter canvas"""
        ret, frame_rgb = self.engine.get_frame()
        if ret:
            # Resize frame to fit panel while maintaining aspect ratio
            h, w = frame_rgb.shape[:2]
            target_h = 700 # Approx panel height
            target_w = int((target_h / h) * w)
            frame_resized = cv2.resize(frame_rgb, (target_w, target_h))

            # Convert to PIL and then ImageTk
            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(30, self.update_video_stream) # ~33fps

    def capture_image(self):
        """Snaps a frame from the live feed and saves it to disk."""
        ret, frame_rgb = self.engine.get_frame()
        if ret:
            # Convert back to BGR for saving
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            filename = f"endoscopy_capture_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(filename, frame_bgr)
            messagebox.showinfo("Image Captured", f"Successfully saved to {filename}")

if __name__ == "__main__":
    root = tk.Tk()
    app = EndoscopeGUI(root)
    root.mainloop()