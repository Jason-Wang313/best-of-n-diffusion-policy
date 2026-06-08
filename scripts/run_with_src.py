from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_with_src.py <script.py> [args...]")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root))
    target = root / sys.argv[1]
    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
