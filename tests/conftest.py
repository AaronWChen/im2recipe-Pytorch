import os
import sys

# The tests check logic on tiny models; on a GPU machine they are slower, not better, and the GPU code path would be tested
# by accident. Set IM2RECIPE_TEST_GPU=1 to leave the GPU visible.
if not os.environ.get("IM2RECIPE_TEST_GPU"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "scripts"), os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)


def _check_checkout():
    """The tests use features added to skipinstructions/ over time"""
    import inspect

    import skipinstructions.tokenize_instructions as tokenizer_module

    if "raw" not in inspect.signature(tokenizer_module.process).parameters:
        pytest.exit(
            "The skipinstr package that Python imports is older than these tests:\n"
            f"  {inspect.getfile(tokenizer_module)}\n"
            "tokenize_instructions.process() has no `raw` option. Replace that folder with the one from "
            "im2recipe-Pytorch-patched.zip (it must be named skipinstructions, as every import and the tests use that name), "
            "and make sure no second, older copy is on the path.", returncode=2)


_check_checkout()


import build_dataset  # noqa: E402
from recipe_store import RecipeStore  # noqa: E402
from skipinstructions import encode, train  # noqa: E402
from skipinstructions.tokenize_instructions import process  # noqa: E402
from synthetic import DIM, INGREDIENTS, make_dataset, make_images, write_w2v  # noqa: E402


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("recipe1M")
    make_dataset(str(root))
    skip_text = root / "text"
    process(str(root / "layer1.json"), str(root / "det_ingrs.json"), str(skip_text), raw=True)
    run = root / "run"
    train.run(train.parse_args([
        "--train-file", str(skip_text / "instructions_train.txt"), "--train-index", str(skip_text / "instructions_train.index.tsv"),
        "--out-dir", str(run), "--maxlen", "10", "--word-size", "16", "--thought-size", str(DIM),
        "--batch-size", "16", "--iters", "30", "--log-every", "1000", "--eval-every", "1000",
        "--save-every", "30", "--device", "cpu"]))
    for part in ("train", "val", "test"):
        encode.main(["--checkpoint", str(run / "skipinstructions-last.pt"),
                     "--sentences", str(skip_text / f"instructions_{part}.txt"),
                     "--index", str(skip_text / f"instructions_{part}.index.tsv"),
                     "--out-prefix", str(root / "skipinstructions" / part), "--batch-size", "8", "--device", "cpu"])
    build_dataset.main(["--dataset", str(root), "--vocab", str(root / "vocab.txt"), "--classes", str(root / "classes.pkl"),
                        "--remove", str(root / "remove.txt"), "--skip-dir", str(root / "skipinstructions"),
                        "--out-dir", str(root / "data")])
    make_images(str(root / "images"), [RecipeStore(str(root / "data" / f"{p}_store")) for p in ("train", "val", "test")])
    # ingredient embeddings: one row per vocab.txt line, in the same order (the model adds the two reserved rows)
    write_w2v(str(root / "vocab.bin"), [t.replace(" ", "_") for t in INGREDIENTS], 6)
    return root
