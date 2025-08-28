import subprocess
from pathlib import Path


def test_cfg_train_anchor() -> None:
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        "grep -R '^\\[CFG-TRAIN\\]' -n src/",
        cwd=root,
        shell=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0 and proc.stdout.strip()
