"""The tagged recipe text format used by RecipeNLG's GPT-2 (generation/dataset2text.ipynb in their code).

    <RECIPE_START> <INPUT_START> ner1 <NEXT_INPUT> ner2 <INPUT_END>
    <INGR_START> line1 <NEXT_INGR> line2 <INGR_END>
    <INSTR_START> step1 <NEXT_INSTR> step2 <INSTR_END>
    <TITLE_START> title <TITLE_END> <RECIPE_END>                       (one recipe per line)

The model is prompted with the ingredient names (INPUT) and writes the rest, title last.
"""
import re

SPECIAL_TOKENS = ["<TITLE_START>", "<TITLE_END>", "<INSTR_START>", "<NEXT_INSTR>", "<INSTR_END>", "<INGR_START>",
                  "<NEXT_INGR>", "<INGR_END>", "<RECIPE_START>", "<RECIPE_END>", "<INPUT_START>", "<INPUT_END>",
                  "<NEXT_INPUT>"]
END_TOKEN = "<RECIPE_END>"

_SPECIAL = re.compile("|".join(re.escape(t) for t in SPECIAL_TOKENS))


def clean(text):
    """One line per recipe: collapse all whitespace (including newlines) and drop look-alikes of our tags."""
    return _SPECIAL.sub(" ", " ".join(str(text).split()))


def recipe_to_text(title, ingredients, directions, ner):
    return ("<RECIPE_START> <INPUT_START> " + " <NEXT_INPUT> ".join(clean(x) for x in ner) +
            " <INPUT_END> <INGR_START> " + " <NEXT_INGR> ".join(clean(x) for x in ingredients) +
            " <INGR_END> <INSTR_START> " + " <NEXT_INSTR> ".join(clean(x) for x in directions) +
            " <INSTR_END> <TITLE_START> " + clean(title) + " <TITLE_END> <RECIPE_END>")


def prompt_for(ingredient_names):
    """Prompt that makes a model trained on this format write ingredients, steps and a title."""
    return ("<RECIPE_START> <INPUT_START> " + " <NEXT_INPUT> ".join(clean(x) for x in ingredient_names) +
            " <INPUT_END> <INGR_START>")
