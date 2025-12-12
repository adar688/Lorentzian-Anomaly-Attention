import os
import sys
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

try:
    from PIL import Image, ImageTk  # For displaying plot images
except ImportError:
    Image = None
    ImageTk = None
    # You should install pillow: pip install pillow


# -----------------------
# Configuration constants
# -----------------------

# Path to the Python executable (use sys.executable so it will work in venvs)
PYTHON_EXE = sys.executable

# Script names (adjust if needed)
PREPARE_SCRIPT = "prepareData.py"
RUN_SCRIPT = "simple_main.py"

# Folder that contains generated plots (update according to your project)
PLOTS_FOLDER = "plots"

# Supported image extensions for plot viewer
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")


class ToolTip:
    """Simple tooltip for Tkinter widgets."""

    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tooltip_window = None

        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        """Show tooltip near the widget."""
        if self.tooltip_window:
            return
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)

        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            self.tooltip_window,
            text=self.text,
            background="white",
            relief="solid",
            borderwidth=1,
            wraplength=300,
            anchor="w",
            justify="left",
        )
        label.pack()

    def hide_tooltip(self, event=None):
        """Hide tooltip when mouse leaves the widget."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class DynhatUI(tk.Tk):
    """Small UI for Dynhat project."""

    def __init__(self):
        super().__init__()
        self.title("Dynhat UI")
        self.geometry("1000x700")

        # State for plot viewer
        self.plot_files = []
        self.current_plot_index = 0
        self.current_image = None  # Keep reference to prevent GC

        # Build UI
        self._create_main_layout()

        # Load plots on startup (if exist)
        self.load_plot_files()
        self.update_plot_viewer()

    # -----------------------
    # Layout creation
    # -----------------------

    def _create_main_layout(self):
        """Create top-level layout with buttons, console, and plot viewer."""
        # Top frame for control buttons
        top_frame = tk.Frame(self, padx=10, pady=10)
        top_frame.pack(fill="x")

        # Generate Data button
        self.btn_generate = ttk.Button(
            top_frame,
            text="Generate Data",
            command=self.on_generate_data_click
        )
        self.btn_generate.grid(row=0, column=0, padx=5, pady=5)

        ToolTip(
            self.btn_generate,
            "Run prepareData.py to generate all required data for Dynhat."
        )

        # Run button (disabled at start)
        self.btn_run = ttk.Button(
            top_frame,
            text="Run",
            command=self.on_run_click,
            state="disabled"
        )
        self.btn_run.grid(row=0, column=1, padx=5, pady=5)

        ToolTip(
            self.btn_run,
            "Run simpleMain.py after data generation is completed successfully."
        )

        # Reload plots button
        self.btn_reload_plots = ttk.Button(
            top_frame,
            text="Reload Plots",
            command=self.on_reload_plots
        )
        self.btn_reload_plots.grid(row=0, column=2, padx=5, pady=5)
        ToolTip(
            self.btn_reload_plots,
            "Scan the plots folder again and reload the images."
        )

        # Separator between control row and content
        separator = ttk.Separator(self, orient="horizontal")
        separator.pack(fill="x", padx=10, pady=5)

        # Middle frame: left = console, right = plots
        middle_frame = tk.Frame(self, padx=10, pady=10)
        middle_frame.pack(fill="both", expand=True)

        # Console frame (left)
        console_frame = tk.LabelFrame(middle_frame, text="Console Output")
        console_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.console_text = tk.Text(console_frame, wrap="word", state="disabled")
        self.console_text.pack(fill="both", expand=True)

        # Add scrollbar for console
        console_scrollbar = ttk.Scrollbar(
            console_frame,
            command=self.console_text.yview
        )
        console_scrollbar.pack(side="right", fill="y")
        self.console_text.configure(yscrollcommand=console_scrollbar.set)

        # Plot viewer frame (right)
        plot_frame = tk.LabelFrame(middle_frame, text="Plots Viewer")
        plot_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))

        # Canvas/Label for image
        self.plot_label = tk.Label(plot_frame, text="No plots found")
        self.plot_label.pack(fill="both", expand=True)

        # Navigation buttons
        nav_frame = tk.Frame(plot_frame)
        nav_frame.pack(fill="x", pady=5)

        self.btn_prev = ttk.Button(
            nav_frame,
            text="◀ Previous",
            command=self.show_prev_plot
        )
        self.btn_prev.pack(side="left", padx=5)

        self.lbl_index = tk.Label(nav_frame, text="0 / 0")
        self.lbl_index.pack(side="left", padx=5)

        self.btn_next = ttk.Button(
            nav_frame,
            text="Next ▶",
            command=self.show_next_plot
        )
        self.btn_next.pack(side="left", padx=5)

        # Download button
        self.btn_download = ttk.Button(
            plot_frame,
            text="Download current plot...",
            command=self.download_current_plot
        )
        self.btn_download.pack(pady=5)

    # -----------------------
    # Console helpers
    # -----------------------

    def log(self, message: str):
        """Append a line to the console output text widget."""
        self.console_text.configure(state="normal")
        self.console_text.insert(tk.END, message + "\n")
        self.console_text.see(tk.END)
        self.console_text.configure(state="disabled")

    # -----------------------
    # Plot viewer helpers
    # -----------------------

    def load_plot_files(self):
        """Scan PLOTS_FOLDER for image files and update internal list."""
        if not os.path.isdir(PLOTS_FOLDER):
            self.plot_files = []
            return

        files = []
        for fname in os.listdir(PLOTS_FOLDER):
            if fname.lower().endswith(IMAGE_EXTENSIONS):
                files.append(os.path.join(PLOTS_FOLDER, fname))

        files.sort()
        self.plot_files = files
        self.current_plot_index = 0

    def update_plot_viewer(self):
        """Display the current plot image or placeholder text."""
        total = len(self.plot_files)
        if total == 0:
            self.plot_label.config(text="No plots found in folder: " + PLOTS_FOLDER)
            self.lbl_index.config(text="0 / 0")
            self.current_image = None
            return

        # Clamp index
        if self.current_plot_index < 0:
            self.current_plot_index = 0
        if self.current_plot_index >= total:
            self.current_plot_index = total - 1

        # Update index label
        self.lbl_index.config(text=f"{self.current_plot_index + 1} / {total}")

        # Load image if Pillow is available
        if Image is None or ImageTk is None:
            self.plot_label.config(text="PIL (pillow) is not installed.\nCannot display images.")
            self.current_image = None
            return

        img_path = self.plot_files[self.current_plot_index]
        try:
            img = Image.open(img_path)

            # Optionally resize image to fit into widget
            # Get current label size
            label_width = self.plot_label.winfo_width() or 400
            label_height = self.plot_label.winfo_height() or 300

            img.thumbnail((label_width, label_height), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(img)
            self.current_image = tk_img  # keep reference
            self.plot_label.config(image=tk_img, text="")
        except Exception as e:
            self.plot_label.config(text=f"Failed to load image:\n{img_path}\n{e}")
            self.current_image = None

    def show_prev_plot(self):
        """Show previous plot in the list."""
        if not self.plot_files:
            return
        self.current_plot_index -= 1
        if self.current_plot_index < 0:
            self.current_plot_index = len(self.plot_files) - 1
        self.update_plot_viewer()

    def show_next_plot(self):
        """Show next plot in the list."""
        if not self.plot_files:
            return
        self.current_plot_index += 1
        if self.current_plot_index >= len(self.plot_files):
            self.current_plot_index = 0
        self.update_plot_viewer()

    def download_current_plot(self):
        """Open a save dialog to copy the current plot to another location."""
        if not self.plot_files:
            messagebox.showinfo("Download", "No plots available to download.")
            return

        current_path = self.plot_files[self.current_plot_index]
        initial_name = os.path.basename(current_path)

        save_path = filedialog.asksaveasfilename(
            title="Save current plot as...",
            initialfile=initial_name,
            defaultextension=os.path.splitext(initial_name)[1],
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files", "*.*")]
        )

        if not save_path:
            return

        try:
            # Simple copy of the file
            with open(current_path, "rb") as src, open(save_path, "wb") as dst:
                dst.write(src.read())
            messagebox.showinfo("Download", f"Plot saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Download error", f"Failed to save plot:\n{e}")

    def on_reload_plots(self):
        """Handler for 'Reload Plots' button."""
        self.load_plot_files()
        self.update_plot_viewer()
        self.log("🔄 Plots reloaded from folder: " + PLOTS_FOLDER)

    # -----------------------
    # Script execution logic
    # -----------------------

    def on_generate_data_click(self):
        """Handler for Generate Data button."""
        script_path = os.path.join(os.getcwd(), PREPARE_SCRIPT)
        if not os.path.isfile(script_path):
            messagebox.showerror("Error", f"Script not found:\n{script_path}")
            return

        # Disable buttons while running
        self.btn_generate.config(state="disabled")
        self.btn_run.config(state="disabled")
        self.log(f"▶ Running data generation: {PREPARE_SCRIPT}")

        # Run in a separate thread so UI will not freeze
        threading.Thread(
            target=self.run_script,
            args=(script_path, [], self.on_generate_success, self.on_generate_fail),
            daemon=True
        ).start()

    def on_run_click(self):
        """Handler for Run button."""
        script_path = os.path.join(os.getcwd(), RUN_SCRIPT)
        if not os.path.isfile(script_path):
            messagebox.showerror("Error", f"Script not found:\n{script_path}")
            return

        self.btn_generate.config(state="disabled")
        self.btn_run.config(state="disabled")
        self.log(f"▶ Running main script: {RUN_SCRIPT}")

        threading.Thread(
            target=self.run_script,
            args=(script_path, [], self.on_run_success, self.on_run_fail),
            daemon=True
        ).start()

    def run_script(self, script_path: str, args, on_success, on_fail):
        """
        Run a Python script with real-time logging into the console widget.

        This function runs in a background thread and uses Popen to stream
        stdout/stderr line by line into the UI.
        """
        cmd = [PYTHON_EXE, script_path] + list(args)
        self.log(f"$ {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            # Must switch back to main thread to update UI
            self.after(0, lambda: on_fail(f"Failed to start process: {e}"))
            return

        # Read output line by line
        if process.stdout:
            for line in process.stdout:
                # Use after to safely update UI from background thread
                self.after(0, lambda l=line: self.log(l.rstrip("\n")))

        process.wait()
        ret = process.returncode

        if ret == 0:
            self.after(0, lambda: on_success())
        else:
            self.after(0, lambda: on_fail(f"Process exited with code {ret}"))

    # -----------------------
    # Callbacks after scripts
    # -----------------------

    def on_generate_success(self):
        """Callback when prepareData.py finished successfully."""
        self.log("✅ Data generation completed successfully.")
        # Allow user to run main now
        self.btn_generate.config(state="normal")
        self.btn_run.config(state="normal")
        # Optionally reload plots (in case your prepare step already creates plots)
        self.on_reload_plots()

    def on_generate_fail(self, message: str):
        """Callback when prepareData.py failed."""
        self.log("❌ Data generation failed.")
        self.log(message)
        # Allow to try again
        self.btn_generate.config(state="normal")
        self.btn_run.config(state="disabled")

    def on_run_success(self):
        """Callback when simpleMain.py finished successfully."""
        self.log("✅ Main script completed successfully.")
        self.btn_generate.config(state="normal")
        self.btn_run.config(state="normal")
        # After main script, reload plots to show new ones
        self.on_reload_plots()

    def on_run_fail(self, message: str):
        """Callback when simpleMain.py failed."""
        self.log("❌ Main script failed.")
        self.log(message)
        self.btn_generate.config(state="normal")
        # You might still want to allow Run, but safer to disable until data is okay
        self.btn_run.config(state="disabled")


if __name__ == "__main__":
    try:
        app = DynhatUI()
        app.mainloop()
    except Exception as e:
        print(f"❌ Failed to start Dynhat UI: {e}")
