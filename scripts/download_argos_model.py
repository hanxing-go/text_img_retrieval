from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys


def download_argos_model(source: str, target: str, out_dir: Path, install: bool) -> Path:
    try:
        import argostranslate.package as argos_package
    except Exception as exc:
        raise RuntimeError("argostranslate is not installed. Run: pip install -r requirements.txt") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    argos_package.update_package_index()
    packages = argos_package.get_available_packages()
    match = next((p for p in packages if p.from_code == source and p.to_code == target), None)
    if match is None:
        raise RuntimeError(f"No Argos package found for {source}->{target}")

    download_path = Path(match.download())
    if download_path.parent != out_dir:
        target_path = out_dir / download_path.name
        try:
            shutil.move(str(download_path), str(target_path))
            download_path = target_path
        except Exception:
            # If move fails (e.g., already exists), keep original path
            pass
    if install:
        argos_package.install_from_path(str(download_path))
    return download_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Argos Translate model package")
    parser.add_argument("--source", default="zh", help="Source language code (default: zh)")
    parser.add_argument("--target", default="en", help="Target language code (default: en)")
    parser.add_argument("--out_dir", default="data/models", help="Output directory for .argosmodel")
    parser.add_argument("--install", action="store_true", help="Install model after download")
    args = parser.parse_args()

    path = download_argos_model(args.source, args.target, Path(args.out_dir), args.install)
    print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
