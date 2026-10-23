"""textenc: per-instruction vectors from any Hugging Face transformer (GPT-2, RecipeNLG, a model you fine-tuned, BERT ...).

Writes the same <prefix>.encs.npy + <prefix>.index.json files as skipinstructions.encode, so scripts/build_dataset.py
takes them via --skip-dir unchanged (use the encoder's hidden size as --stDim).
"""
