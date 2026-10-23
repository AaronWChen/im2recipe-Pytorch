#!/usr/bin/env python
"""Check that this folder matches the release it came from, file by file.

    python scripts/verify_checkout.py                      # compare with MANIFEST.sha256
    python scripts/verify_checkout.py --package skipinstructions   # if you renamed the skipinstr package
    python scripts/verify_checkout.py --write              # (maintainers) write MANIFEST.sha256 for the current tree

Reports files that are missing, files whose content differs (edited, or an older version left behind by a partial update),
and extra files in the source folders. Line endings are ignored. Your data, snapshots, results and runs folders are not checked.
Exit status 0 means every released file is present and identical.
"""
import argparse
import hashlib
import os
import sys
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = "MANIFEST.sha256"
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "data", "snapshots", "results", "runs", "models", ".venv", "venv"}
SKIP_SUFFIXES = (".pyc", ".pt", ".pth", ".tar", ".safetensors", ".npy", ".log")
PACKAGE = "skipinstructions"  # the name the release uses


def walk(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info"))
        for f in sorted(filenames):
            if f != MANIFEST and not f.endswith(SKIP_SUFFIXES) and f != ".DS_Store":
                yield os.path.relpath(os.path.join(dirpath, f), root).replace(os.sep, "/")


def to_release_name(data, alt):
    """Undo a package rename (alt -> PACKAGE) in file content. If alt is a prefix of PACKAGE, occurrences that already
    spell out the full release name are left alone, so applying this twice does not mangle anything."""
    pattern = re.escape(alt.encode())
    if alt != PACKAGE and PACKAGE.startswith(alt):
        pattern += b"(?!" + re.escape(PACKAGE[len(alt):].encode()) + b")"
    return re.sub(pattern, PACKAGE.encode(), data)


def digest(path, rename_back=None):
    with open(path, "rb") as f:
        data = f.read().replace(b"\r\n", b"\n")
    if rename_back:  # undo a package rename so the content can be compared with the release
        data = to_release_name(data, rename_back)
    return hashlib.sha256(data).hexdigest()


def write_manifest(root):
    lines = [f"{digest(os.path.join(root, rel))}  {rel}" for rel in walk(root)]
    with open(os.path.join(root, MANIFEST), "w") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


def verify(root, package=PACKAGE):
    renamed = package != PACKAGE
    with open(os.path.join(root, MANIFEST)) as f:
        expected = dict(line.rstrip("\n").split("  ", 1)[::-1] for line in f if line.strip())
    missing, different, seen = [], [], set()
    for rel, sha in expected.items():
        local = rel.replace(PACKAGE, package) if renamed else rel
        seen.add(local)
        path = os.path.join(root, local)
        if not os.path.exists(path):
            missing.append(local)
        elif digest(path, package if renamed else None) != sha:
            different.append(local)
    extra = [rel for rel in walk(root) if rel not in seen and rel.endswith((".py", ".sh", ".md"))]
    return missing, different, extra


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=ROOT)
    p.add_argument("--package", default=PACKAGE, help="the name your copy of the skipinstructions package has, if you renamed it")
    p.add_argument("--write", action="store_true")
    a = p.parse_args(argv)
    if a.write:
        print(f"wrote {MANIFEST} with {write_manifest(a.root)} files")
        return 0
    if not os.path.exists(os.path.join(a.root, MANIFEST)):
        sys.exit(f"{MANIFEST} is not in {a.root}; it comes with the release zip")
    missing, different, extra = verify(a.root, a.package)
    for title, items in (("MISSING", missing), ("DIFFERENT from the release", different), ("extra (not in the release)", extra)):
        if items:
            print(f"{title}:")
            print("".join(f"  {x}\n" for x in items), end="")
    if missing or different:
        print("\nCopy those files from the release zip. If you applied the .patch file more than once, or on top of "
              "a modified tree, some changes may have been lost or applied in the wrong place; look for *.rej and *.orig files.")
        return 1
    print("OK: every released file is present and identical" + (f" (package renamed to {a.package})" if a.package != PACKAGE else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
