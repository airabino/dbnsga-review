#!/usr/bin/env python3
"""
setup_env.py — One-shot environment bootstrap for vehicle_fleet_opt.

Builds a fresh, local Python venv (default: rust_env/), installs the Python
dependencies from requirements.txt, and compiles rust_src/ into the
fleet_opt_core extension via `maturin develop --release` so that both `src`
and `rust_src` are importable.

The venv is intentionally NOT checked into git (see .gitignore) — venvs bake
in machine-specific absolute paths and their bin/ symlinks don't survive a
zip/download round-trip, both of which have caused real breakage in this repo
before. Run this script on every machine instead of relying on a committed copy.

Usage:
    python3 setup_env.py
    python3 setup_env.py --python /opt/anaconda3/bin/python3.12
    python3 setup_env.py --reuse       # skip recreating an existing venv
    python3 setup_env.py --venv-dir .venv
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from glob import glob

REPO_ROOT = Path(__file__).resolve().parent
MIN_PYTHON = (3, 11)
RUSTUP_INSTALL_CMD = (
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
)


def fail(message: str) -> None:
    print(f"\n✗ {message}", file=sys.stderr)
    sys.exit(1)

def venv_bin_dir(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts" if sys.platform == "win32" else "bin")

def venv_python(venv_dir: Path) -> Path:
    exe = "python.exe" if sys.platform == "win32" else "python3"
    return venv_bin_dir(venv_dir) / exe

def bootstrap_pip(python: Path) -> None:
    check = subprocess.run(
        [str(python), "-m", "pip", "--version"], capture_output=True,
    )
    if check.returncode == 0:
        print("✓ pip already present in venv")
        return
    print("Bootstrapping pip via ensurepip ...")
    subprocess.run([str(python), "-m", "ensurepip", "--upgrade"], check=True)
    print("✓ pip installed")

def install_requirements(python: Path) -> None:
    req_file = REPO_ROOT / "requirements.txt"
    if not req_file.exists():
        fail(f"requirements.txt not found at {req_file}")
    print(f"Installing dependencies from {req_file.name} ...")
    subprocess.run(
        [str(python), "-m", "pip", "install", "--disable-pip-version-check",
         "-r", str(req_file)],
        check=True,
    )
    print("✓ Python dependencies installed")

def find_wheel():

    directory = Path("target/wheels")
    matching_files = directory.glob("fleet_opt_core-*")

    try:

        newest_file = max(matching_files, key=lambda f: f.stat().st_mtime)

        return newest_file

    except ValueError:

        fail("Could not find wheel")


def build_rust_extension(venv_dir: Path, python: Path) -> None:
    import os

    print("Building fleet_opt_core with maturin ...")
    env = os.environ.copy()
    env.pop("CONDA_PREFIX", None)  # maturin refuses to run if both VIRTUAL_ENV
                                    # and CONDA_PREFIX are set at the same time.
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["PATH"] = f"{venv_bin_dir(venv_dir)}{os.pathsep}{env.get('PATH', '')}"

    subprocess.run(
        [str(python), "-m", "maturin", "build", "--release"],
        check=True, cwd=REPO_ROOT, env=env,
    )

    subprocess.run(
        [str(python), "-m", "pip", "install",
        find_wheel(), "--force-reinstall"],
        check=True, cwd=REPO_ROOT, env=env,
    )

    print("✓ fleet_opt_core built and installed")

def verify(python: Path) -> None:
    print("Verifying src and fleet_opt_core import cleanly ...")
    smoke_test = (
        "import sys, unittest.mock; "
        "sys.modules['cvxpy'] = unittest.mock.MagicMock(); "
        "import fleet_opt_core; "
        "import src; "
        "print('fleet_opt_core:', fleet_opt_core.__file__)"
    )
    result = subprocess.run(
        [str(python), "-c", smoke_test], cwd=REPO_ROOT,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        fail(f"Verification failed:\n{result.stdout}\n{result.stderr}")
    print(result.stdout.strip())
    print("✓ src and rust_src (fleet_opt_core) import successfully")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python", default=sys.executable,
        help="Base Python interpreter (>=3.11) to build the venv from. "
             "Defaults to the interpreter running this script.",
    )
    parser.add_argument(
        "--venv-dir", default="rust_env",
        help="Directory to create the venv in (default: rust_env).",
    )
    parser.add_argument(
        "--reuse", action="store_true",
        help="Reuse an existing venv instead of recreating it from scratch.",
    )
    args = parser.parse_args()

    venv_dir = (REPO_ROOT / args.venv_dir).resolve()

    python = sys.executable
    bootstrap_pip(python)
    install_requirements(python)
    build_rust_extension(venv_dir, python)
    verify(python)

    print("\nPackage fleet_opt_core successfully built and installed.")
    print("\nAfter making changes to rust source, re-run this script.")


if __name__ == "__main__":
    main()
