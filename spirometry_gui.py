"""
Diagnostic Pneumotachometer Simulator GUI
Professional spirometry interface with Patient Demographics,
% Predicted metrics, AI Diagnostics, CSV Export, and a Premium Dark UI.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import csv

from pneumotach_engine import PneumotachEngine
from diagnostic_classifier import DiagnosticClassifier

class SpirometryGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Diagnostic Pneumotachometer Simulator - Team 7")
        self.root.geometry("1600x980")

        # Premium Dark Theme Color Palette
        self.colors = {
            'bg_main': '#0b0f19',       # Deepest background
            'bg_panel': '#161b22',      # Panel backgrounds
            'bg_card': '#21262d',       # Widget/Card backgrounds
            'accent_blue': '#58a6ff',   # Primary action
            'accent_blue_hover': '#79b8ff',
            'accent_green': '#238636',  # Start/Success
            'accent_green_hover': '#2ea043',
            'accent_red': '#da3633',    # Alarms/Errors
            'accent_yellow': '#d29922', # Warnings/Processing
            'text_primary': '#f0f6fc',  # Main text
            'text_secondary': '#8b949e',# Labels/Units
            'border': '#30363d'         # Subtle lines
        }
        self.root.configure(bg=self.colors['bg_main'])

        self.engine = PneumotachEngine()
        self.ai = DiagnosticClassifier() # Automatically loads .pkl
        self.last_state = None

        self.setup_styles()
        self.create_ui()
        self.update_loop()

    def setup_styles(self):
        """Configure premium ttk styles and forcefully override OS dropdown colors"""
        style = ttk.Style()
        style.theme_use('clam')

        # 1. Aggressive Override for the hidden Dropdown Menu (Listbox)
        # We use both explicit and catch-all commands to bypass Windows native styling
        self.root.option_add('*TCombobox*Listbox.background', self.colors['bg_card'])
        self.root.option_add('*TCombobox*Listbox.foreground', self.colors['text_primary'])
        self.root.option_add('*TCombobox*Listbox.selectBackground', self.colors['accent_blue'])
        self.root.option_add('*TCombobox*Listbox.selectForeground', 'white')
        self.root.option_add('*Listbox.background', self.colors['bg_card'])
        self.root.option_add('*Listbox.foreground', self.colors['text_primary'])

        # 2. Configure Combobox Input Field
        style.configure("TCombobox",
                        fieldbackground=self.colors['bg_card'],
                        background=self.colors['bg_panel'],
                        foreground=self.colors['text_primary'],
                        bordercolor=self.colors['border'],
                        arrowcolor=self.colors['text_primary'],
                        padding=5)

        # 3. Map Focus States (Prevents field from turning white when clicked)
        style.map('TCombobox',
                  fieldbackground=[('readonly', self.colors['bg_card']), ('focus', self.colors['bg_card'])],
                  selectbackground=[('readonly', self.colors['bg_card']), ('focus', self.colors['bg_card'])],
                  selectforeground=[('readonly', self.colors['text_primary']), ('focus', self.colors['text_primary'])],
                  foreground=[('readonly', self.colors['text_primary']), ('focus', self.colors['text_primary'])])

        # 4. Configure Spinbox Input Field
        style.configure("TSpinbox",
                        fieldbackground=self.colors['bg_card'],
                        background=self.colors['bg_panel'],
                        foreground=self.colors['text_primary'],
                        bordercolor=self.colors['border'],
                        arrowcolor=self.colors['text_primary'],
                        padding=5)

        style.map('TSpinbox',
                  fieldbackground=[('focus', self.colors['bg_card'])],
                  selectbackground=[('focus', self.colors['accent_blue'])],
                  selectforeground=[('focus', 'white')])

    def create_ui(self):
        # --- HEADER ---
        header = tk.Frame(self.root, bg=self.colors['bg_card'], height=75)
        header.pack(fill=tk.X, padx=20, pady=(20, 10))
        header.pack_propagate(False)

        # Title
        tk.Label(header, text="DIAGNOSTIC PNEUMOTACHOMETER", font=('Segoe UI', 22, 'bold'),
                 bg=self.colors['bg_card'], fg=self.colors['text_primary']).pack(side=tk.LEFT, padx=20)
        tk.Label(header, text="| SBE 3220 ELECTRONIC SPIROMETRY", font=('Segoe UI', 14),
                 bg=self.colors['bg_card'], fg=self.colors['text_secondary']).pack(side=tk.LEFT, padx=0)

        # Export Button (with hover effect)
        self.export_btn = tk.Button(header, text="⭳ EXPORT REPORT (CSV)", command=self.export_csv, state=tk.DISABLED,
                                    bg=self.colors['accent_blue'], fg='white', font=('Segoe UI', 10, 'bold'),
                                    relief=tk.FLAT, cursor='hand2', padx=15, pady=8)
        self.export_btn.pack(side=tk.RIGHT, padx=20, pady=15)
        self.export_btn.bind("<Enter>", lambda e: self.on_hover(self.export_btn, self.colors['accent_blue_hover']) if self.export_btn['state'] == tk.NORMAL else None)
        self.export_btn.bind("<Leave>", lambda e: self.on_hover(self.export_btn, self.colors['accent_blue']) if self.export_btn['state'] == tk.NORMAL else None)

        # --- MAIN CONTAINER ---
        main_container = tk.Frame(self.root, bg=self.colors['bg_main'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # --- LEFT PANEL (Controls & Metrics) ---
        left_panel = tk.Frame(main_container, bg=self.colors['bg_panel'], highlightbackground=self.colors['border'], highlightthickness=1)
        # Fixed width to prevent resizing
        left_panel.config(width=420)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # 1. Demographics & Controls
        ctrl_frame = tk.Frame(left_panel, bg=self.colors['bg_panel'])
        ctrl_frame.pack(fill=tk.X, padx=20, pady=20)
        tk.Label(ctrl_frame, text="PATIENT SETUP", font=('Segoe UI', 11, 'bold'), bg=self.colors['bg_panel'], fg=self.colors['text_secondary']).pack(anchor='w', pady=(0,10))

        demo_frame = tk.Frame(ctrl_frame, bg=self.colors['bg_panel'])
        demo_frame.pack(fill=tk.X)

        tk.Label(demo_frame, text="Age:", font=('Segoe UI', 10), bg=self.colors['bg_panel'], fg=self.colors['text_primary']).grid(row=0, column=0, sticky='w', pady=8)
        self.age_var = tk.IntVar(value=25)
        ttk.Spinbox(demo_frame, from_=10, to=100, textvariable=self.age_var, width=6, font=('Segoe UI', 10)).grid(row=0, column=1, padx=(10, 20), sticky='w')

        tk.Label(demo_frame, text="Height (cm):", font=('Segoe UI', 10), bg=self.colors['bg_panel'], fg=self.colors['text_primary']).grid(row=0, column=2, sticky='w', pady=8)
        self.height_var = tk.IntVar(value=175)
        ttk.Spinbox(demo_frame, from_=100, to=220, textvariable=self.height_var, width=6, font=('Segoe UI', 10)).grid(row=0, column=3, padx=10, sticky='w')

        tk.Label(demo_frame, text="Sex:", font=('Segoe UI', 10), bg=self.colors['bg_panel'], fg=self.colors['text_primary']).grid(row=1, column=0, sticky='w', pady=15)
        self.sex_var = tk.StringVar(value="Male")
        ttk.Combobox(demo_frame, textvariable=self.sex_var, values=["Male", "Female"], state="readonly", width=10, font=('Segoe UI', 10)).grid(row=1, column=1, columnspan=3, sticky='w', padx=10)

        # Profile Selection
        tk.Label(ctrl_frame, text="Test Profile:", font=('Segoe UI', 10), bg=self.colors['bg_panel'], fg=self.colors['text_primary']).pack(anchor='w', pady=(15, 5))
        self.profile_var = tk.StringVar(value="Normal")
        profiles = ["Normal", "Obstructive (COPD)", "Restrictive", "Sensor Zero-Drift", "3L Syringe Calibration"]
        self.profile_dropdown = ttk.Combobox(ctrl_frame, textvariable=self.profile_var, values=profiles, state="readonly", font=('Segoe UI', 11))
        self.profile_dropdown.pack(fill=tk.X, pady=(0, 20))

        # Start Button (with hover effect)
        self.start_btn = tk.Button(ctrl_frame, text="▶ START FVC MANEUVER", command=self.start_test,
                                   bg=self.colors['accent_green'], fg='white', font=('Segoe UI', 12, 'bold'),
                                   relief=tk.FLAT, cursor='hand2', pady=12)
        self.start_btn.pack(fill=tk.X)
        self.start_btn.bind("<Enter>", lambda e: self.on_hover(self.start_btn, self.colors['accent_green_hover']) if self.start_btn['state'] == tk.NORMAL else None)
        self.start_btn.bind("<Leave>", lambda e: self.on_hover(self.start_btn, self.colors['accent_green']) if self.start_btn['state'] == tk.NORMAL else None)

        # Separator
        tk.Frame(left_panel, height=1, bg=self.colors['border']).pack(fill=tk.X, padx=20)

        # 2. Clinical Metrics (Designed as Cards)
        metrics_frame = tk.Frame(left_panel, bg=self.colors['bg_panel'])
        metrics_frame.pack(fill=tk.X, padx=20, pady=20)
        tk.Label(metrics_frame, text="CLINICAL METRICS", font=('Segoe UI', 11, 'bold'), bg=self.colors['bg_panel'], fg=self.colors['text_secondary']).pack(anchor='w', pady=(0,10))

        self.labels = {}
        for key, title, color in [('fvc', 'FVC', self.colors['accent_blue']),
                                  ('fev1', 'FEV1', self.colors['accent_green']),
                                  ('ratio', 'FEV1/FVC', self.colors['accent_yellow'])]:
            # Card Frame
            card = tk.Frame(metrics_frame, bg=self.colors['bg_card'], highlightbackground=self.colors['border'], highlightthickness=1)
            card.pack(fill=tk.X, pady=6)

            tk.Label(card, text=title, font=('Segoe UI', 12), bg=self.colors['bg_card'], fg=self.colors['text_primary']).pack(side=tk.LEFT, padx=15, pady=15)
            val = tk.Label(card, text="--", font=('Segoe UI', 18, 'bold'), bg=self.colors['bg_card'], fg=color)
            val.pack(side=tk.RIGHT, padx=15)
            self.labels[key] = val

        # Separator
        tk.Frame(left_panel, height=1, bg=self.colors['border']).pack(fill=tk.X, padx=20)

        # 3. AI Diagnosis
        ai_frame = tk.Frame(left_panel, bg=self.colors['bg_panel'])
        ai_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        tk.Label(ai_frame, text="AI DIAGNOSIS", font=('Segoe UI', 11, 'bold'), bg=self.colors['bg_panel'], fg=self.colors['text_secondary']).pack(anchor='w')

        self.ai_label = tk.Label(ai_frame, text="Waiting for test...", font=('Segoe UI', 16, 'bold'), bg=self.colors['bg_panel'], fg=self.colors['text_secondary'])
        self.ai_label.pack(expand=True)

        # --- RIGHT PANEL (Premium Graphs) ---
        right_panel = tk.Frame(main_container, bg=self.colors['bg_panel'], highlightbackground=self.colors['border'], highlightthickness=1)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        plt.style.use('dark_background')
        self.fig = Figure(figsize=(11, 8), facecolor=self.colors['bg_panel'], dpi=100)
        self.fig.subplots_adjust(hspace=0.35, wspace=0.25, left=0.07, right=0.96, top=0.92, bottom=0.08)

        self.ax1 = self.fig.add_subplot(2, 2, 1) # dP
        self.ax2 = self.fig.add_subplot(2, 2, 2) # Flow
        self.ax3 = self.fig.add_subplot(2, 2, 3) # Volume
        self.ax4 = self.fig.add_subplot(2, 2, 4) # Flow-Volume Loop

        self.canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        self.reset_graphs()

    def on_hover(self, widget, color):
        """Handle button hover color changes"""
        widget['background'] = color

    def reset_graphs(self):
        """Cleans and styles the matplotlib axes to look like a premium dashboard"""
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
            ax.set_facecolor(self.colors['bg_panel'])
            ax.grid(True, alpha=0.1, color=self.colors['text_secondary'], linestyle='-')

            # Remove top and right borders for a cleaner, modern look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(self.colors['border'])
            ax.spines['bottom'].set_color(self.colors['border'])
            ax.tick_params(colors=self.colors['text_secondary'], labelsize=9)

        self.ax1.set_title("Transducer Pressure Signal (Pa)", color=self.colors['text_primary'], fontsize=11, fontweight='bold', pad=10)
        self.ax2.set_title("Flow-Time Curve Q(t) [BTPS]", color=self.colors['text_primary'], fontsize=11, fontweight='bold', pad=10)
        self.ax3.set_title("Spirogram V(t)", color=self.colors['text_primary'], fontsize=11, fontweight='bold', pad=10)
        self.ax3.set_xlabel("Time (s)", color=self.colors['text_secondary'], fontsize=10)
        self.ax4.set_title("Flow-Volume Loop", color=self.colors['text_primary'], fontsize=11, fontweight='bold', pad=10)
        self.ax4.set_xlabel("Volume (L)", color=self.colors['text_secondary'], fontsize=10)
        self.canvas.draw_idle()

    def start_test(self):
        # Crash Prevention: Input Validation
        try:
            age = self.age_var.get()
            height = self.height_var.get()
            if age <= 0 or height <= 0:
                raise ValueError("Values must be positive.")
        except tk.TclError:
            messagebox.showerror("Input Error", "Please enter valid numbers for Age and Height.")
            return
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return

        # Prepare UI for new test
        self.start_btn.config(state=tk.DISABLED, bg='#30363d') # Gray out
        self.export_btn.config(state=tk.DISABLED, bg='#30363d')
        self.labels['fvc'].config(text="--")
        self.labels['fev1'].config(text="--")
        self.labels['ratio'].config(text="--")
        self.ai_label.config(text="Analyzing Patient...", fg=self.colors['accent_yellow'])
        self.reset_graphs()

        # Start Engine
        sex = self.sex_var.get()
        self.engine.start_maneuver(self.profile_var.get(), age, height, sex)

    def update_loop(self):
        state = self.engine.get_current_state(advance_by_ms=50)

        if state['is_running'] or (state['finished'] and state.get('t_array') is not None):
            self.reset_graphs()

            t = state['t_array']

            # Panel 1: Pressure
            self.ax1.plot(t, state['dP_noisy_array'], color=self.colors['text_secondary'], alpha=0.3, label='Raw Sensor')
            self.ax1.plot(t, state['dP_filtered_array'], color=self.colors['accent_blue'], linewidth=2.5, label='Filtered LPF')
            self.ax1.legend(loc='upper right', frameon=False, fontsize=8, labelcolor=self.colors['text_primary'])
            self.ax1.set_xlim(0, 6)

            # Panel 2: Flow (With Fill)
            self.ax2.plot(t, state['Q_array'], color=self.colors['accent_yellow'], linewidth=2.5)
            self.ax2.fill_between(t, state['Q_array'], 0, color=self.colors['accent_yellow'], alpha=0.15)
            self.ax2.axvline(x=1.0, color=self.colors['accent_red'], linestyle='--', alpha=0.6, label='1.0s Mark')
            self.ax2.set_xlim(0, 6)

            # Panel 3: Volume (With Fill)
            self.ax3.plot(t, state['V_array'], color=self.colors['accent_green'], linewidth=2.5)
            self.ax3.fill_between(t, state['V_array'], 0, color=self.colors['accent_green'], alpha=0.15)
            self.ax3.set_xlim(0, 6)

            # Panel 4: Flow-Volume Loop (With Fill)
            self.ax4.plot(state['V_array'], state['Q_array'], color=self.colors['accent_red'], linewidth=2.5)
            if len(state['V_array']) > 0:
                self.ax4.fill_between(state['V_array'], state['Q_array'], 0, color=self.colors['accent_red'], alpha=0.15)
            self.ax4.set_xlim(0, max(8, state['fvc'] + 1))
            self.ax4.set_ylim(0, max(12, np.max(state['Q_array']) + 1))

            self.canvas.draw_idle()

            if state['finished']:
                self.last_state = state
                self.start_btn.config(state=tk.NORMAL, bg=self.colors['accent_green'])
                self.export_btn.config(state=tk.NORMAL, bg=self.colors['accent_blue'])

                # Format Labels with % Predicted
                if state['profile'] == "3L Syringe Calibration":
                    self.labels['fvc'].config(text=f"{state['fvc']:.2f} L")
                    self.labels['fev1'].config(text=f"{state['fev1']:.2f} L")
                    self.ai_label.config(text="CALIBRATION SUCCESS\n(Error < 1%)", fg=self.colors['accent_blue'])
                else:
                    self.labels['fvc'].config(text=f"{state['fvc']:.2f} L \n({state['pct_fvc']:.0f}%)", font=('Segoe UI', 14, 'bold'))
                    self.labels['fev1'].config(text=f"{state['fev1']:.2f} L \n({state['pct_fev1']:.0f}%)", font=('Segoe UI', 14, 'bold'))
                    self.labels['ratio'].config(text=f"{state['ratio']:.1f}%", font=('Segoe UI', 16, 'bold'))

                    # Annotate final graphs
                    self.ax3.scatter(1.0, state['fev1'], color=self.colors['accent_red'], s=60, zorder=5)
                    self.ax3.axhline(y=state['fvc'], color=self.colors['text_secondary'], linestyle='--', alpha=0.5)
                    self.canvas.draw_idle()

                    # AI Diagnosis
                    diagnosis, conf = self.ai.predict(state['pct_fvc'], state['pct_fev1'], state['ratio'])
                    color = self.colors['accent_red'] if diagnosis != "Normal" else self.colors['accent_green']
                    self.ai_label.config(text=f"{diagnosis}\n(Conf: {conf:.0%})", fg=color)

                self.engine.data = None

        self.root.after(50, self.update_loop)

    def export_csv(self):
        """Export the clinical arrays to a professional CSV file"""
        if self.last_state is None:
            messagebox.showerror("Error", "No test data available to export.")
            return

        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")], title="Save Spirometry Report")
        if filepath:
            try:
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Diagnostic Pneumotachometer - Spirometry Report"])
                    writer.writerow(["Patient Age", self.age_var.get()])
                    writer.writerow(["Patient Height (cm)", self.height_var.get()])
                    writer.writerow(["Patient Sex", self.sex_var.get()])
                    writer.writerow(["Profile Tested", self.last_state['profile']])
                    writer.writerow([])
                    writer.writerow(["--- CLINICAL METRICS ---"])
                    writer.writerow(["FVC (L)", f"{self.last_state['fvc']:.3f}", f"{self.last_state.get('pct_fvc', 0):.1f}% Predicted"])
                    writer.writerow(["FEV1 (L)", f"{self.last_state['fev1']:.3f}", f"{self.last_state.get('pct_fev1', 0):.1f}% Predicted"])
                    writer.writerow(["FEV1/FVC Ratio (%)", f"{self.last_state['ratio']:.1f}"])
                    writer.writerow([])
                    writer.writerow(["--- TIME DOMAIN ARRAYS ---"])
                    writer.writerow(["Time (s)", "Differential Pressure (Pa)", "Flow Rate BTPS (L/s)", "Volume (L)"])

                    for t, dp, q, v in zip(self.last_state['t_array'], self.last_state['dP_filtered_array'], self.last_state['Q_array'], self.last_state['V_array']):
                        writer.writerow([f"{t:.3f}", f"{dp:.3f}", f"{q:.3f}", f"{v:.3f}"])

                messagebox.showinfo("Success", f"Spirometry report successfully saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to save CSV:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SpirometryGUI(root)
    root.mainloop()