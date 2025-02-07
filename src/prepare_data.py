import json
import os
from datetime import datetime
from typing import Dict, List

import pandas as pd

# Configuration for data collection
GENRES = {
    "philosophy": [
        "Plato/Republic",
        "Aristotle/Metaphysics",
        "Aristotle/Ethics",
        "Seneca/Letters",
        "Cicero/De_Natura_Deorum",
    ],
    "poetry": ["Homer/Iliad", "Virgil/Aeneid", "Lucretius/De_Rerum_Natura"],
    "history": ["Thucydides/History", "Tacitus/Annals", "Herodotus/Histories"],
}


def load_raw_texts(data_dir: str) -> Dict[str, List[str]]:
    """Load raw texts from Perseus Digital Library format"""
    texts = {}
    for genre, works in GENRES.items():
        for work in works:
            filepath = os.path.join(data_dir, f"{work}.txt")
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    texts[work] = f.readlines()
            else:
                print(f"Warning: {filepath} not found")
    return texts


def extract_contexts(
    texts: Dict[str, List[str]], target_words: List[str], window_size: int = 3
) -> List[Dict]:
    """Extract contexts for target words with metadata"""
    contexts = []

    for work, lines in texts.items():
        genre = next(g for g, works in GENRES.items() if work in works)

        for i, line in enumerate(lines):
            for word in target_words:
                if word in line:
                    # Get context window
                    start = max(0, i - window_size)
                    end = min(len(lines), i + window_size + 1)
                    context = " ".join(lines[start:end]).strip()

                    # Store with metadata
                    contexts.append(
                        {
                            "text": context,
                            "word": word,
                            "work": work,
                            "genre": genre,
                            "line_number": i,
                            "source_text": line.strip(),
                        }
                    )

    return contexts


def prepare_dataset(
    data_dir: str = "data/raw",
    output_dir: str = "data/processed",
    min_contexts: int = 50,
) -> pd.DataFrame:
    """Prepare expanded dataset with balanced contexts"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all target words from term pairs
    from run_experiments import TERM_PAIRS

    greek_terms = []
    latin_terms = []

    for pairs in TERM_PAIRS.values():
        for gr, la in pairs:
            greek_terms.append(gr)
            latin_terms.append(la)

    # Load raw texts
    print("Loading raw texts...")
    texts = load_raw_texts(data_dir)

    # Extract contexts
    print("Extracting contexts...")
    greek_contexts = extract_contexts(
        {
            k: v
            for k, v in texts.items()
            if any(
                g in k
                for g in ["Plato", "Aristotle", "Homer", "Thucydides", "Herodotus"]
            )
        },
        greek_terms,
    )

    latin_contexts = extract_contexts(
        {
            k: v
            for k, v in texts.items()
            if any(
                l in k for l in ["Seneca", "Cicero", "Virgil", "Lucretius", "Tacitus"]
            )
        },
        latin_terms,
    )

    # Combine and convert to DataFrame
    all_contexts = greek_contexts + latin_contexts
    df = pd.DataFrame(all_contexts)

    # Balance contexts
    print("Balancing contexts...")
    balanced_contexts = []
    for word in greek_terms + latin_terms:
        word_contexts = df[df["word"] == word]
        if len(word_contexts) < min_contexts:
            print(f"Warning: Only {len(word_contexts)} contexts found for {word}")
        balanced_contexts.append(
            word_contexts.sample(
                n=min(len(word_contexts), min_contexts), random_state=42
            )
        )

    balanced_df = pd.concat(balanced_contexts, ignore_index=True)

    # Save dataset
    output_path = os.path.join(output_dir, "contexts.csv")
    balanced_df.to_csv(output_path, index=False)

    # Save dataset stats
    stats = {
        "total_contexts": len(balanced_df),
        "contexts_per_word": {
            word: len(balanced_df[balanced_df["word"] == word])
            for word in greek_terms + latin_terms
        },
        "genres": {
            genre: len(balanced_df[balanced_df["genre"] == genre])
            for genre in GENRES.keys()
        },
        "timestamp": datetime.now().isoformat(),
    }

    with open(os.path.join(output_dir, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDataset prepared and saved to {output_path}")
    print(f"Total contexts: {len(balanced_df)}")
    print("\nContexts per word:")
    for word, count in stats["contexts_per_word"].items():
        print(f"  {word}: {count}")

    return balanced_df


if __name__ == "__main__":
    df = prepare_dataset()
