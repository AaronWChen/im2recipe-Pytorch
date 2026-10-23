import os
import shutil

import pytest

import verify_checkout as V

P = V.PACKAGE
ALT = "altpkg"

FILES = {
    "README.md": f"Install {P} first.\n",
    "test.py": "print('hi')\n",
    f"{P}/__init__.py": "",
    f"{P}/encode.py": f"from {P}.data import x\n\ndef load_model():\n    model = 1\n    return model\n",
    f"{P}/tests/test_{P}.py": f"import {P}\n",
    "textenc/pooling.py": "A = 1\nB = 2\n",
}


def write(root, rel, text):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


@pytest.fixture()
def tree(tmp_path):
    root = str(tmp_path / "repo")
    for rel, text in FILES.items():
        write(root, rel, text)
    V.write_manifest(root)
    return root


def test_a_fresh_tree_verifies_and_changes_are_reported(tree):
    assert V.verify(tree) == ([], [], [])
    write(tree, "notes.md", "mine")
    write(tree, "data/anything.py", "user data is not checked")
    write(tree, "snapshots/model.py", "neither are snapshots")
    write(tree, f"{P}/encode.py", FILES[f"{P}/encode.py"] + "\n# edited\n")
    os.remove(os.path.join(tree, "test.py"))
    missing, different, extra = V.verify(tree)
    assert missing == ["test.py"] and different == [f"{P}/encode.py"] and extra == ["notes.md"]
    assert V.main(["--root", tree]) == 1


def test_line_endings_do_not_matter(tree):
    write(tree, "textenc/pooling.py", FILES["textenc/pooling.py"].replace("\n", "\r\n"))
    assert V.verify(tree) == ([], [], [])
    assert V.main(["--root", tree]) == 0


def rename_package(root, new):
    os.rename(os.path.join(root, P), os.path.join(root, new))
    for dirpath, _, files in os.walk(root):
        for f in files:
            path = os.path.join(dirpath, f)
            if f.endswith((".py", ".md")):
                text = open(path, encoding="utf-8").read()
                open(path, "w", encoding="utf-8", newline="").write(text.replace(P, new))
    for dirpath, _, files in os.walk(os.path.join(root, new)):
        for f in files:
            if P in f:
                os.rename(os.path.join(dirpath, f), os.path.join(dirpath, f.replace(P, new)))


def test_a_renamed_package_verifies_with_its_new_name_only(tree):
    rename_package(tree, ALT)
    assert V.verify(tree, package=ALT) == ([], [], [])
    missing, _, _ = V.verify(tree)  # without being told about the rename, nothing is where it should be
    assert any(m.startswith(P + "/") for m in missing)
    assert V.main(["--root", tree, "--package", ALT]) == 0


def test_a_stale_file_among_renamed_ones_is_found(tree):
    rename_package(tree, ALT)
    path = os.path.join(tree, ALT, "encode.py")
    text = open(path, encoding="utf-8").read()
    open(path, "w", encoding="utf-8").write(text.replace("    model = 1", "        model = 1"))  # an indentation slip
    _, different, _ = V.verify(tree, package=ALT)
    assert different == [f"{ALT}/encode.py"]

def test_a_shorter_name_that_is_a_prefix_of_the_release_name_is_handled(tree):
    short = P[:8]
    assert V.to_release_name(f"{short} and {P}".encode(), short) == f"{P} and {P}".encode()  # no double expansion
    rename_package(tree, short)
    assert V.verify(tree, package=short) == ([], [], [])

def test_without_a_manifest_it_says_so(tmp_path):
    with pytest.raises(SystemExit, match="MANIFEST.sha256"):
        V.main(["--root", str(tmp_path)])
