import os
import sys
import subprocess
from typing import List, Tuple, Optional

import gradio as gr


# -----------------------
# Configuration constants
# -----------------------

# Script names: adjust to your project structure if needed
PREPARE_SCRIPT = "prepareData.py"
RUN_SCRIPT = "simple_main.py"

# Folder that contains generated plots
PLOTS_DIR = "plots"

# Supported image extensions
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")


# -----------------------
# Helper functions
# -----------------------

def get_plot_files() -> List[str]:
    """Return a sorted list of plot image file paths in PLOTS_DIR."""
    if not os.path.isdir(PLOTS_DIR):
        return []
    files = []
    for fname in os.listdir(PLOTS_DIR):
        if fname.lower().endswith(IMAGE_EXTENSIONS):
            files.append(os.path.join(PLOTS_DIR, fname))
    files.sort()
    return files


def run_script(script_name: str) -> Tuple[str, bool]:
    """
    Run a Python script by name using the current Python interpreter.

    Returns:
        logs (str): Combined stdout + stderr.
        success (bool): True if returncode == 0.
    """
    script_path = os.path.join(os.getcwd(), script_name)
    if not os.path.isfile(script_path):
        return f"❌ Script not found: {script_path}", False

    cmd = [sys.executable, script_path]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += "\n" + result.stderr

        if result.returncode == 0:
            header = f"✅ {script_name} finished successfully.\n"
            return header + output, True
        else:
            header = f"❌ {script_name} exited with code {result.returncode}.\n"
            return header + output, False
    except Exception as e:
        return f"❌ Failed to run {script_name}: {e}", False


# -----------------------
# Gradio callback functions
# -----------------------

def generate_data(current_logs: str) -> Tuple[str, gr.Update]:
    """
    Callback for 'Generate Data' button.
    Runs PREPARE_SCRIPT and appends logs.
    Enables 'Run' button only on success.
    """
    logs, success = run_script(PREPARE_SCRIPT)
    current_logs = current_logs or ""
    new_logs = current_logs + "\n\n=== Generate Data ===\n" + logs
    run_button_update = gr.update(interactive=success)
    return new_logs, run_button_update


def run_main(current_logs: str, idx_state: int):
    """
    Callback for 'Run' button.
    Runs RUN_SCRIPT, appends logs, and reloads plots.
    """
    logs, success = run_script(RUN_SCRIPT)
    current_logs = current_logs or ""
    new_logs = current_logs + "\n\n=== Run Main ===\n" + logs

    files = get_plot_files()
    if files:
        # Reset index to first plot
        idx_state = 0
        slider_update = gr.update(
            minimum=1,
            maximum=len(files),
            step=1,
            value=1,
            visible=True
        )
        img = files[0]
        download_file = files[0]
    else:
        idx_state = 0
        slider_update = gr.update(
            minimum=0,
            maximum=0,
            step=1,
            value=0,
            visible=False
        )
        img = None
        download_file = None

    # After a successful run, keep 'Run' enabled (even if script failed you might change this logic)
    return new_logs, idx_state, slider_update, img, download_file


def reload_plots(idx_state: int):
    """
    Callback for 'Reload plots' button.
    Reloads plot list without running any script.
    """
    files = get_plot_files()
    if not files:
        idx_state = 0
        slider_update = gr.update(
            minimum=0,
            maximum=0,
            step=1,
            value=0,
            visible=False
        )
        img = None
        download_file = None
    else:
        # Ensure index is in range
        idx_state = max(0, min(idx_state, len(files) - 1))
        slider_update = gr.update(
            minimum=1,
            maximum=len(files),
            step=1,
            value=idx_state + 1,
            visible=True
        )
        img = files[idx_state]
        download_file = files[idx_state]

    return idx_state, slider_update, img, download_file


def change_plot_by_slider(slider_value: float, idx_state: int):
    """
    Callback when slider changes.
    Updates current plot according to slider position.
    """
    files = get_plot_files()
    if not files:
        return 0, None, None

    idx = int(slider_value) - 1
    idx = max(0, min(idx, len(files) - 1))

    img = files[idx]
    download_file = files[idx]
    return idx, img, download_file


def prev_plot(idx_state: int):
    """
    Show previous plot in the list (cyclic).
    Also updates the slider position.
    """
    files = get_plot_files()
    if not files:
        slider_update = gr.update(
            minimum=0, maximum=0, step=1, value=0, visible=False
        )
        return 0, slider_update, None, None

    idx_state = (idx_state - 1) % len(files)
    slider_update = gr.update(
        minimum=1,
        maximum=len(files),
        step=1,
        value=idx_state + 1,
        visible=True
    )
    img = files[idx_state]
    download_file = files[idx_state]
    return idx_state, slider_update, img, download_file


def next_plot(idx_state: int):
    """
    Show next plot in the list (cyclic).
    Also updates the slider position.
    """
    files = get_plot_files()
    if not files:
        slider_update = gr.update(
            minimum=0, maximum=0, step=1, value=0, visible=False
        )
        return 0, slider_update, None, None

    idx_state = (idx_state + 1) % len(files)
    slider_update = gr.update(
        minimum=1,
        maximum=len(files),
        step=1,
        value=idx_state + 1,
        visible=True
    )
    img = files[idx_state]
    download_file = files[idx_state]
    return idx_state, slider_update, img, download_file


# -----------------------
# Build Gradio UI
# -----------------------

with gr.Blocks() as demo:
    gr.Markdown("# Dynhat UI (Generate Data, Run, Logs & Plots Viewer)")

    with gr.Row():
        btn_generate = gr.Button("Generate Data")
        btn_run = gr.Button("Run", interactive=False)
        btn_reload = gr.Button("Reload plots")

    logs_box = gr.Textbox(
        label="Logs",
        lines=20,
        value="",
        interactive=False
    )

    # State for current plot index (0-based)
    idx_state = gr.State(0)

    with gr.Row():
        plot_image = gr.Image(
            label="Current plot",
            interactive=False,
            type="filepath"  # so Gradio knows it's a path
        )
        download_file = gr.File(
            label="Download current plot"
        )

    with gr.Row():
        btn_prev = gr.Button("◀ Previous")
        btn_next = gr.Button("Next ▶")
        plot_slider = gr.Slider(
            label="Plot index",
            minimum=0,
            maximum=0,
            step=1,
            value=0,
            visible=False
        )

    # Wire buttons and slider to callbacks
    btn_generate.click(
        fn=generate_data,
        inputs=[logs_box],
        outputs=[logs_box, btn_run]
    )

    btn_run.click(
        fn=run_main,
        inputs=[logs_box, idx_state],
        outputs=[logs_box, idx_state, plot_slider, plot_image, download_file]
    )

    btn_reload.click(
        fn=reload_plots,
        inputs=[idx_state],
        outputs=[idx_state, plot_slider, plot_image, download_file]
    )

    plot_slider.change(
        fn=change_plot_by_slider,
        inputs=[plot_slider, idx_state],
        outputs=[idx_state, plot_image, download_file]
    )

    btn_prev.click(
        fn=prev_plot,
        inputs=[idx_state],
        outputs=[idx_state, plot_slider, plot_image, download_file]
    )

    btn_next.click(
        fn=next_plot,
        inputs=[idx_state],
        outputs=[idx_state, plot_slider, plot_image, download_file]
    )

if __name__ == "__main__":
    # Set share=True if you want a public URL as well (useful for Colab).
    demo.launch()
