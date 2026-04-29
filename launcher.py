import sys
import subprocess
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
APP_FILE = APP_DIR / "app.py"

def main():
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP_FILE)], cwd=str(APP_DIR))

if __name__ == "__main__":
    main()
