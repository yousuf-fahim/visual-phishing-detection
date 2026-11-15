"""Phase 0: print library versions, check for a GPU, and test offline tldextract.

Run: .venv/Scripts/python -m src.check_env
Writes the same information to results/env.json so later phases (and the
report) can cite the exact environment.
"""
import importlib
import json
import platform
import sys
from pathlib import Path

# Import name -> name to show (pip names differ from import names for some).
LIBS = {
    "numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn",
    "lightgbm": "lightgbm", "matplotlib": "matplotlib", "seaborn": "seaborn",
    "cv2": "opencv-python", "PIL": "pillow", "imagehash": "imagehash",
    "torch": "torch", "torchvision": "torchvision", "timm": "timm",
    "pytorch_grad_cam": "grad-cam", "onnx": "onnx", "onnxruntime": "onnxruntime",
    "tqdm": "tqdm", "tldextract": "tldextract", "langdetect": "langdetect",
    "bs4": "beautifulsoup4",
}


def main() -> None:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "libraries": {}}
    for mod_name, pretty in LIBS.items():
        try:
            mod = importlib.import_module(mod_name)
            info["libraries"][pretty] = getattr(mod, "__version__", "installed")
        except Exception as exc:  # report, don't hide, a broken install
            info["libraries"][pretty] = f"IMPORT FAILED: {exc}"

    import torch
    info["cuda_available"] = torch.cuda.is_available()
    info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    info["cpu_threads"] = torch.get_num_threads()

    # Offline tldextract check: must work with no network access.
    from src.domain_utils import registered_domain
    info["tldextract_offline_test"] = registered_domain("http://login.example.co.uk/path")

    for k, v in info.items():
        if k != "libraries":
            print(f"{k:24s} {v}")
    for lib, ver in info["libraries"].items():
        print(f"  {lib:20s} {ver}")

    out = Path("results") / "env.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(info, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
