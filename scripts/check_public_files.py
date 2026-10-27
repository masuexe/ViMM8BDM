"""Check public project files for common local data and excluded artifacts."""
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "home directory": re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/"),
    "mounted machine path": re.compile(r"/mnt/[a-z]/", re.I),
    "temporary machine path": re.compile(r"/(?:tmp|var/tmp)/[A-Za-z0-9_.-]+"),
    "Windows profile path": re.compile(r"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]", re.I),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    "private key": re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    "local artifact link": re.compile(r"\]\([^)]*(?:\.\./)?runs/"),
}
EXCLUDED_PARTS = {".local", "runs", "data", "weights", ".venv", "__pycache__", ".aws", ".codex", ".agents"}
EXCLUDED_SUFFIXES = {".vrec", ".pk3", ".wad", ".pt", ".pth", ".onnx", ".safetensors", ".zip", ".cld", ".lmp"}


def public_paths():
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True)
    if result.returncode == 0 and result.stdout:
        return [Path(p) for p in result.stdout.decode().split("\0") if p], True
    files = [Path(name) for name in ("README.md", "pyproject.toml", ".gitignore")]
    for directory in ("src", "tests", "scripts", "docs"):
        files.extend(p.relative_to(ROOT) for p in (ROOT / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts)
    return files, False


def inspect_text(text):
    return [(number, name) for number, line in enumerate(text.splitlines(), 1)
            for name, pattern in PATTERNS.items() if pattern.search(line)]


def main():
    failures = []
    paths, indexed = public_paths()
    for relative in paths:
        path = ROOT / relative
        if EXCLUDED_PARTS.intersection(relative.parts) or path.suffix.lower() in EXCLUDED_SUFFIXES:
            failures.append(f"{relative}: excluded local file or artifact")
            continue
        if not indexed and not path.is_file():
            failures.append(f"{relative}: indexed file is missing")
            continue
        if path.is_symlink():
            failures.append(f"{relative}: inspect symbolic links separately")
            continue
        try:
            if indexed:
                result = subprocess.run(["git", "show", ":" + relative.as_posix()], cwd=ROOT,
                                        capture_output=True, check=True)
                text = result.stdout.decode("utf-8")
            else:
                text = path.read_text(encoding="utf-8")
        except UnicodeError:
            failures.append(f"{relative}: non-text file requires separate review")
            continue
        failures.extend(f"{relative}:{line}: {rule}" for line, rule in inspect_text(text))
    if failures:
        print("Public file check failed:")
        print("\n".join(failures))
        return 1
    print(f"Public file check passed: {len(paths)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
