import os
import sys
import subprocess
from typing import List, Tuple, Optional, Any, Generator

import gradio as gr


# -----------------------
# Configuration constants
# -----------------------

# Script names: change if your files are in another folder or have other names
PREPARE_SCRIPT = "script/utils/prepareData.py"
RUN_SCRIPT = "script/simple_main.py"

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


def get_plot_state(idx_state: int) -> Tuple[int, Any, Optional[str], Optional[str]]:
    """
    Compute slider + image state based on current index and available plot files.

    Returns:
        idx_state (int): possibly adjusted index
        slider_update (gr.Update): update object for the slider
        img (str | None): filepath of current image
        download_file (str | None): filepath for download component
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
        return idx_state, slider_update, None, None

    # Clamp index
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


# -----------------------
# Streaming script runners
# -----------------------

def generate_data(current_logs: str) -> Generator[Tuple[str, Any], None, None]:
    """
    Streaming callback for 'Generate Data' button.

    Runs PREPARE_SCRIPT and yields logs as they arrive.
    Enables 'Run' button only on success.
    """
    current_logs = current_logs or ""
    script_path = os.path.join(os.getcwd(), PREPARE_SCRIPT)

    header = "\n\n=== Generate Data ===\n"
    if not os.path.isfile(script_path):
        msg = f"❌ Script not found: {script_path}"
        current_logs += header + msg
        # Run button remains disabled
        yield current_logs, gr.update(interactive=False)
        return

    cmd = [sys.executable, script_path]
    current_logs += header + f"$ {' '.join(cmd)}\n"
    # While running, keep Run button disabled
    yield current_logs, gr.update(interactive=False)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        current_logs += f"\n❌ Failed to start {PREPARE_SCRIPT}: {e}"
        yield current_logs, gr.update(interactive=False)
        return

    # Stream stdout line by line
    if process.stdout:
        for line in process.stdout:
            current_logs += line
            # Keep Run button disabled during execution
            yield current_logs, gr.update(interactive=False)

    process.wait()
    if process.returncode == 0:
        current_logs += "\n✅ Generate Data finished successfully.\n"
        # Now enable Run button
        yield current_logs, gr.update(interactive=True)
    else:
        current_logs += f"\n❌ Generate Data exited with code {process.returncode}.\n"
        # Keep Run disabled on failure
        yield current_logs, gr.update(interactive=False)


def run_main(current_logs: str, idx_state: int) -> Generator[Tuple[str, int, Any, Optional[str], Optional[str]], None, None]:
    """
    Streaming callback for 'Run' button.

    Runs RUN_SCRIPT, streams logs, and at the end reloads plots.
    """
    current_logs = current_logs or ""
    script_path = os.path.join(os.getcwd(), RUN_SCRIPT)

    header = "\n\n=== Run Main ===\n"
    if not os.path.isfile(script_path):
        msg = f"❌ Script not found: {script_path}"
        current_logs += header + msg
        # Return current plot state unchanged
        idx_state, slider_update, img, download_file = get_plot_state(idx_state)
        yield current_logs, idx_state, slider_update, img, download_file
        return

    cmd = [sys.executable, script_path]
    current_logs += header + f"$ {' '.join(cmd)}\n"

    # Before starting, show current plot state (if any)
    idx_state, slider_update, img, download_file = get_plot_state(idx_state)
    yield current_logs, idx_state, slider_update, img, download_file

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        current_logs += f"\n❌ Failed to start {RUN_SCRIPT}: {e}"
        idx_state, slider_update, img, download_file = get_plot_state(idx_state)
        yield current_logs, idx_state, slider_update, img, download_file
        return

    # Stream stdout lines
    if process.stdout:
        for line in process.stdout:
            current_logs += line
            # While running, keep the same plot state
            idx_state, slider_update, img, download_file = get_plot_state(idx_state)
            yield current_logs, idx_state, slider_update, img, download_file

    process.wait()
    if process.returncode == 0:
        current_logs += "\n✅ Main script finished successfully.\n"
        # After successful run, reload plots and show from first plot
        idx_state, slider_update, img, download_file = get_plot_state(0)
        yield current_logs, idx_state, slider_update, img, download_file
    else:
        current_logs += f"\n❌ Main script exited with code {process.returncode}.\n"
        # Reload plots anyway (in case partial output exists)
        idx_state, slider_update, img, download_file = get_plot_state(idx_state)
        yield current_logs, idx_state, slider_update, img, download_file


# -----------------------
# Non-streaming helpers (reload plots, navigation)
# -----------------------

def reload_plots(idx_state: int) -> Tuple[int, Any, Optional[str], Optional[str]]:
    """
    Callback for 'Reload plots' button.
    Reloads plot list without running any script.
    """
    idx_state, slider_update, img, download_file = get_plot_state(idx_state)
    return idx_state, slider_update, img, download_file


def change_plot_by_slider(slider_value: float, idx_state: int) -> Tuple[int, Optional[str], Optional[str]]:
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


def prev_plot(idx_state: int) -> Tuple[int, Any, Optional[str], Optional[str]]:
    """
    Show previous plot in the list (cyclic).
    Also updates the slider position.
    """
    files = get_plot_files()
    if not files:
        return get_plot_state(0)

    idx_state = (idx_state - 1) % len(files)
    return get_plot_state(idx_state)


def next_plot(idx_state: int) -> Tuple[int, Any, Optional[str], Optional[str]]:
    """
    Show next plot in the list (cyclic).
    Also updates the slider position.
    """
    files = get_plot_files()
    if not files:
        return get_plot_state(0)

    idx_state = (idx_state + 1) % len(files)
    return get_plot_state(idx_state)


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
            type="filepath"  # path to image file
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

    # Wire callbacks
    btn_generate.click(
        fn=generate_data,
        inputs=[logs_box],
        outputs=[logs_box, btn_run],
        
    )

    btn_run.click(
        fn=run_main,
        inputs=[logs_box, idx_state],
        outputs=[logs_box, idx_state, plot_slider, plot_image, download_file],
    
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
    # share=True is needed in Colab to get a public URL.
    demo.launch(share=True)
