import csv
import json
import os

import pytest

pytest.importorskip("pandas", reason="pip install -r requirements.txt")
import pandas as pd

import merge_recipenlg as M
import sample_recipenlg
from textenc.lmdata import SPECIAL_TOKENS, recipe_to_text

N = 40  # Recipe1M recipes r0..r39: partitions 24 train / 8 val / 8 test


def part(i):
    return "train" if i < 24 else ("val" if i < 32 else "test")


def row(i, title=None, source="Recipes1M", link=None, ingredients=None, directions=None, ner="auto"):
    title = title or f"Recipe number {i}"
    directions = directions or [f"Cook batch {i} until golden brown and fragrant.", f"Serve dish {i} warm with rice."]
    ingredients = ingredients or [f"{i} cups of flour", "1 tsp salt"]
    ner = ["flour", "salt"] if ner == "auto" else ner
    return {"title": title, "ingredients": json.dumps(ingredients), "directions": json.dumps(directions),
            "link": link or f"www.example.com/recipes/{i}", "source": source,
            "NER": json.dumps(ner) if isinstance(ner, list) else ner}


@pytest.fixture()
def fixture(tmp_path):
    layer1 = [{"id": f"r{i}", "partition": part(i), "url": f"http://www.example.com/recipes/{i}"} for i in range(N)]
    layer1_path = tmp_path / "layer1.json"
    layer1_path.write_text(json.dumps(layer1))
    rows = []
    for i in range(N):
        if i in (10, 11):
            continue  # Recipe1M recipes RecipeNLG dropped
        r = row(i)
        if i % 5 == 0:
            r["link"] += "/"  # trailing slash on one side
        rows.append(r)
    rows += [row(100 + j, source="Gathered", link=f"www.other.com/r/{j}") for j in range(30)]
    rows.append(row(200, title="ab", source="Gathered", link="www.other.com/short"))  # title too short
    rows.append(row(201, source="Gathered", link="www.other.com/step",
                    directions=["Step 1: cook the whole thing for a long while."]))  # filtered by 'step'
    rows.append(row(202, source="Gathered", link="www.other.com/nonner", ner=""))  # no NER
    rows.append({**row(203, source="Gathered", link="www.other.com/badjson"), "ingredients": "not json at all"})
    # the same recipe as val recipe r30 but scraped from another site: an exact copy of a held-out recipe
    rows.append(row(30, source="Gathered", link="www.copy-of-val.com/r30"))
    rows.append(row(300, source="Gathered", link="www.other.com/newline",
                    directions=["Mix the flour\nand water.", "Bake it until it is done well."]))
    csv_path = tmp_path / "full_dataset.csv"
    pd.DataFrame(rows).rename_axis("Unnamed: 0").reset_index().to_csv(csv_path, index=False)
    return tmp_path, str(csv_path), str(layer1_path), rows


def lines(path):
    return open(path, encoding="utf-8").read().split("\n")[:-1]


def test_norm_link_and_filter_semantics():
    assert M.norm_link("http://www.a.com/x/") == M.norm_link("https://www.a.com/x") == M.norm_link("www.a.com/x") == "www.a.com/x"
    ok = ("Good title", '["a", "b"]', '["Cook it for a good while until done."]')
    assert M.passes_recipenlg_filter(*ok)
    assert not M.passes_recipenlg_filter("abc", ok[1], ok[2])  # title under 4 characters
    assert not M.passes_recipenlg_filter(ok[0], "", ok[2])
    assert not M.passes_recipenlg_filter(ok[0], ok[1], '["Short."]')  # directions text under 30 characters
    assert not M.passes_recipenlg_filter(ok[0], ok[1], '["Step 1: cook it for a good while until done."]')
    assert not M.passes_recipenlg_filter(ok[0], ok[1], '["Then MIX ALL the things together in a bowl."]')


def test_join_restores_ids_and_partitions(fixture):
    tmp, csv_path, layer1, rows = fixture
    s = M.merge(csv_path, layer1, str(tmp / "out"), val_fraction=0.25, log=lambda *a: None)
    assert s["rows"] == len(rows)
    assert s["recipes1m_source_matched"] == s["recipes1m_source_rows"] == N - 2  # normalisation matches every one
    assert s["matched_by_partition"] == {"train": 24 - 2, "val": 8, "test": 8}
    assert s["recipe1m_not_in_recipenlg"] == {"train": 2, "val": 0, "test": 0}
    # the copy scraped elsewhere is not matched by link; it is caught by content
    assert s["matched_recipe1m"] == N - 2
    links = list(csv.DictReader(open(tmp / "out" / "recipe_links.csv", encoding="utf-8")))
    assert len(links) == N - 2 and links[0].keys() >= {"recipenlg_row", "link", "recipe1m_id", "partition", "source", "title"}
    by_id = {r["recipe1m_id"]: r for r in links}
    assert by_id["r33"]["partition"] == "test" and by_id["r5"]["link"] == "www.example.com/recipes/5"  # slash removed
    assert by_id["r0"]["link"] == "www.example.com/recipes/0"
    assert "r10" not in by_id and "r11" not in by_id


def test_holdout_is_never_trained_on(fixture):
    tmp, csv_path, layer1, rows = fixture
    s = M.merge(csv_path, layer1, str(tmp / "out"), val_fraction=0.25, log=lambda *a: None)
    out = tmp / "out"
    train, val, hold = (lines(out / f"lm_{k}.txt") for k in ("train", "val", "holdout"))
    assert len(hold) == 16 and s["written"] == {"train": len(train), "val": len(val), "holdout": 16}
    assert s["dup_of_holdout"] == 1  # the copy of r30
    held_titles = {f"Recipe number {i}" for i in range(24, N)}
    for text in train + val:
        title = text.split("<TITLE_START> ")[1].split(" <TITLE_END>")[0]
        assert title not in held_titles  # not by link, not by content
    assert all("Recipe number" in t for t in hold)
    # train + val = everything that survived: 22 train-partition + 30 gathered + the newline one
    assert len(train) + len(val) == 22 + 30 + 1
    assert s["filtered_out"] == 2 and s["no_ner"] == 1 and s["bad_json"] == 1
    assert 0 < len(val) < len(train)  # val_fraction 0.25 of a hash of the link


def test_copy_of_a_holdout_recipe_is_kept_only_when_asked(fixture):
    tmp, csv_path, layer1, _ = fixture
    s = M.merge(csv_path, layer1, str(tmp / "out"), dedup_holdout=False, val_fraction=0.0, log=lambda *a: None)
    assert s["dup_of_holdout"] == 0
    assert sum("<TITLE_START> Recipe number 30 <TITLE_END>" in t for t in lines(tmp / "out" / "lm_train.txt")) == 1


def test_split_is_deterministic_and_text_format_is_exact(fixture):
    tmp, csv_path, layer1, rows = fixture
    for name in ("a", "b"):
        M.merge(csv_path, layer1, str(tmp / name), val_fraction=0.3, log=lambda *a: None)
    for k in ("train", "val", "holdout"):
        assert (tmp / "a" / f"lm_{k}.txt").read_bytes() == (tmp / "b" / f"lm_{k}.txt").read_bytes()
    r = rows[2]  # r2, a train-partition recipe
    expected = recipe_to_text(r["title"], json.loads(r["ingredients"]), json.loads(r["directions"]), json.loads(r["NER"]))
    texts = lines(tmp / "a" / "lm_train.txt") + lines(tmp / "a" / "lm_val.txt")
    assert expected in texts
    assert expected.startswith("<RECIPE_START> <INPUT_START> flour <NEXT_INPUT> salt <INPUT_END> <INGR_START> 2 cups of flour <NEXT_INGR>")
    assert expected.endswith("<TITLE_START> Recipe number 2 <TITLE_END> <RECIPE_END>")
    newline_one = [t for t in texts if "Mix the flour and water." in t]  # the newline inside a direction became a space
    assert len(newline_one) == 1 and all(tag in newline_one[0] for tag in SPECIAL_TOKENS)


def test_limit_rows_and_partition_choice(fixture):
    tmp, csv_path, layer1, _ = fixture
    s = M.merge(csv_path, layer1, str(tmp / "out"), holdout_partitions=("test",), limit=20, val_fraction=0.0,
                log=lambda *a: None)
    assert s["rows"] == 20 and s["holdout_partitions"] == ["test"] and s["written"]["holdout"] == 0  # test rows come later


def test_sampler_keeps_columns_and_is_reproducible(fixture, tmp_path):
    _, csv_path, _, rows = fixture
    total, n = sample_recipenlg.sample_csv(csv_path, str(tmp_path / "s1.csv"), rows=20, seed=1, chunksize=25)
    sample_recipenlg.sample_csv(csv_path, str(tmp_path / "s2.csv"), rows=20, seed=1, chunksize=25)
    assert total == len(rows) and 5 <= n <= 40
    df = pd.read_csv(tmp_path / "s1.csv", dtype=str, keep_default_na=False)
    assert list(df.columns) == ["Unnamed: 0", "title", "ingredients", "directions", "link", "source", "NER"]
    assert (tmp_path / "s1.csv").read_bytes() == (tmp_path / "s2.csv").read_bytes()
    assert json.loads(df.loc[0, "ingredients"])  # JSON columns survive
    assert os.path.getsize(tmp_path / "s1.csv") < os.path.getsize(csv_path)
