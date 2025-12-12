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

def generate_data(current_logs: str) -> Generator[Tuple[str, Any, Any], None, None]:
    """
    Streaming callback for 'Generate Data' button.

    While the script is running:
      - Disable the Generate button and show a loading label.
      - Disable the Run button.
    After completion:
      - Re-enable Generate.
      - Enable Run only if script succeeded.
    Also filters noisy duplicate lines (e.g., from progress bars) and batches updates.
    """
    current_logs = current_logs or ""
    script_path = os.path.join(os.getcwd(), PREPARE_SCRIPT)

    header = "\n\n=== Generate Data ===\n"

    # Initial state: set buttons to "running" for this action
    btn_generate_update = gr.update(value="⏳ Generating...", interactive=False)
    btn_run_update = gr.update(interactive=False)

    if not os.path.isfile(script_path):
        msg = f"❌ Script not found: {script_path}"
        current_logs += header + msg
        # Keep Generate enabled so user can fix and retry, Run stays disabled
        btn_generate_update = gr.update(value="Generate Data", interactive=True)
        btn_run_update = gr.update(interactive=False)
        yield current_logs, btn_generate_update, btn_run_update
        return

    cmd = [sys.executable, script_path]
    current_logs += header + f"$ {' '.join(cmd)}\n"

    # Show starting state immediately
    yield current_logs, btn_generate_update, btn_run_update

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
        # Allow user to retry, Run stays disabled
        btn_generate_update = gr.update(value="Generate Data", interactive=True)
        btn_run_update = gr.update(interactive=False)
        yield current_logs, btn_generate_update, btn_run_update
        return

    # Stream stdout line by line, skipping noisy duplicates and batching updates
    last_line: Optional[str] = None
    buffer_lines: list[str] = []
    flush_every = 5  # update UI every N lines

    if process.stdout:
        for raw_line in process.stdout:
            # Strip trailing newline
            line = raw_line.rstrip("\n")

            # If there are carriage returns (e.g., tqdm progress), take last part
            if "\r" in line:
                line = line.split("\r")[-1]

            # Skip empty lines
            if not line.strip():
                continue

            # Skip consecutive identical lines (common for progress updates)
            if line == last_line:
                continue

            last_line = line
            buffer_lines.append(line)

            # Flush buffer to UI every few lines
            if len(buffer_lines) >= flush_every:
                current_logs += "\n".join(buffer_lines) + "\n"
                buffer_lines.clear()
                yield current_logs, btn_generate_update, btn_run_update

    # Flush any remaining buffered lines
    if buffer_lines:
        current_logs += "\n".join(buffer_lines) + "\n"
        buffer_lines.clear()
        yield current_logs, btn_generate_update, btn_run_update

    process.wait()
    if process.returncode == 0:
        current_logs += "\n✅ Generate Data finished successfully.\n"
        # Generate can be pressed again, Run is now enabled
        btn_generate_update = gr.update(value="Generate Data", interactive=True)
        btn_run_update = gr.update(value="Run", interactive=True)
        yield current_logs, btn_generate_update, btn_run_update
    else:
        current_logs += f"\n❌ Generate Data exited with code {process.returncode}.\n"
        # Allow retry of Generate, keep Run disabled
        btn_generate_update = gr.update(value="Generate Data", interactive=True)
        btn_run_update = gr.update(interactive=False)
        yield current_logs, btn_generate_update, btn_run_update


def run_main(
    current_logs: str,
    idx_state: int
) -> Generator[Tuple[str, int, Any, Optional[str], Optional[str], Any, Any], None, None]:
    """
    Streaming callback for 'Run' button.

    While the script is running:
      - Disable the Run button and show a loading label.
      - Disable the Generate button.
    After completion:
      - Re-enable both buttons.
      - Reload plots and show current plot state (starting from first plot on success).
    Also filters noisy duplicate lines and batches updates, without rescanning plots each line.
    """
    current_logs = current_logs or ""
    script_path = os.path.join(os.getcwd(), RUN_SCRIPT)

    header = "\n\n=== Run Main ===\n"

    # While running this script, lock both buttons
    btn_generate_update = gr.update(interactive=False)
    btn_run_update = gr.update(value="⏳ Running...", interactive=False)

    if not os.path.isfile(script_path):
        msg = f"❌ Script not found: {script_path}"
        current_logs += header + msg
        # Restore buttons to idle state (Generate enabled, Run disabled)
        btn_generate_update = gr.update(value="Generate Data", interactive=True)
        btn_run_update = gr.update(value="Run", interactive=False)
        idx_state, slider_update, img, download_file = get_plot_state(idx_state)
        yield current_logs, idx_state, slider_update, img, download_file, btn_generate_update, btn_run_update
        return

    cmd = [sys.executable, script_path]
    current_logs += header + f"$ {' '.join(cmd)}\n"

    # Initial plots state before running (do not refresh them on every log line)
    idx_state, slider_update, img, download_file = get_plot_state(idx_state)
    yield current_logs, idx_state, slider_update, img, download_file, btn_generate_update, btn_run_update

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
        # Unlock Generate, keep Run disabled
        btn_generate_update = gr.update(value="Generate Data", interactive=True)
        btn_run_update = gr.update(value="Run", interactive=False)
        idx_state, slider_update, img, download_file = get_plot_state(idx_state)
        yield current_logs, idx_state, slider_update, img, download_file, btn_generate_update, btn_run_update
        return

    # Stream stdout lines, skipping noisy duplicates and batching updates
    last_line: Optional[str] = None
    buffer_lines: list[str] = []
    flush_every = 5  # update UI every N lines

    if process.stdout:
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")

            if "\r" in line:
                line = line.split("\r")[-1]

            if not line.strip():
                continue

            if line == last_line:
                continue

            last_line = line
            buffer_lines.append(line)

            if len(buffer_lines) >= flush_every:
                current_logs += "\n".join(buffer_lines) + "\n"
                buffer_lines.clear()
                # While running, keep the same plots and button state
                yield current_logs, idx_state, slider_update, img, download_file, btn_generate_update, btn_run_update

    # Flush remaining buffered lines
    if buffer_lines:
        current_logs += "\n".join(buffer_lines) + "\n"
        buffer_lines.clear()
        yield current_logs, idx_state, slider_update, img, download_file, btn_generate_update, btn_run_update

    process.wait()
    if process.returncode == 0:
        current_logs += "\n✅ Main script finished successfully.\n"
        # Reload plots from first index and unlock both buttons
        idx_state, slider_update, img, download_file = get_plot_state(0)
        btn_generate_update = gr.update(value="Generate Data", interactive=True)
        btn_run_update = gr.update(value="Run", interactive=True)
        yield current_logs, idx_state, slider_update, img, download_file, btn_generate_update, btn_run_update
    else:
        current_logs += f"\n❌ Main script exited with code {process.returncode}.\n"
        # Reload plots (if any) and unlock buttons (Run stays enabled so user can retry)
        idx_state, slider_update, img, download_file = get_plot_state(idx_state)
        btn_generate_update = gr.update(value="Generate Data", interactive=True)
        btn_run_update = gr.update(value="Run", interactive=True)
        yield current_logs, idx_state, slider_update, img, download_file, btn_generate_update, btn_run_update


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
    gr.Markdown("Lorentzian Anomaly Attention (LAA): A Self-Attentive Approach to Citation Networks")

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
        outputs=[logs_box, btn_generate, btn_run],
    )

    btn_run.click(
        fn=run_main,
        inputs=[logs_box, idx_state],
        outputs=[logs_box, idx_state, plot_slider, plot_image, download_file, btn_generate, btn_run],
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
    # queue() is recommended when using generator functions (streaming).
    demo.queue()
    demo.launch(share=True)
