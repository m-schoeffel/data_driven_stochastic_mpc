import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_simulation(save_dir=None):
    """Run the simulation as a subprocess and yield stdout lines as they arrive.

    Running in a subprocess keeps the Streamlit process itself from blocking
    on the simulation loop or on matplotlib's `plt.show()`.
    """
    cmd = [sys.executable, "-m", "simulation.main", "--no-show"]
    if save_dir is not None:
        cmd += ["--save-dir", save_dir]

    process = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        yield line.rstrip("\n")

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(
            f"Simulation exited with code {process.returncode}")
