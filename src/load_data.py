import pandas as pd
from datasets import load_dataset

# Load verified working dataset with text processing
dataset = load_dataset(
    "universal_dependencies", "grc_perseus", split="train", trust_remote_code=True
)

# Extract work ID from first token and join tokens into text
greek_df = pd.DataFrame(
    {
        "text": [" ".join(tokens) for tokens in dataset["tokens"]],
        "work": [
            tokens[0].split(":")[0] if ":" in tokens[0] else "unknown"
            for tokens in dataset["tokens"]
        ],
    }
)

# Filter target words with work metadata
target_words = ["λόγος", "ψυχή", "ἀρετή"]
filtered = [
    greek_df[greek_df.text.str.contains(word)].head(50).assign(word=word)
    for word in target_words
]

final_df = pd.concat(filtered)[["word", "text", "work"]].reset_index(drop=True)
print(f"Dataset prepared with {len(final_df)} sentences")
print("Works identified:", final_df.work.unique())
