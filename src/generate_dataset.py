import json
import os
import random

import pandas as pd

# Define philosophical contexts and templates
CONTEXTS = {
    "philosophy": {
        "templates": [
            "In considering {concept}, {author} argues that {term} is fundamental to understanding reality.",
            "The nature of {term} plays a central role in {author}'s theory of {concept}.",
            "When discussing {concept}, {author} emphasizes how {term} relates to human knowledge.",
            "{author} defines {term} as the principle underlying {concept}.",
            "The relationship between {term} and {concept} is explored in {author}'s work.",
        ],
        "concepts": [
            "knowledge",
            "truth",
            "reality",
            "being",
            "wisdom",
            "virtue",
            "justice",
            "nature",
            "mind",
            "soul",
        ],
        "greek_authors": [
            "Plato",
            "Aristotle",
            "Plotinus",
            "Chrysippus",
            "Epicurus",
            "Zeno",
            "Parmenides",
            "Heraclitus",
            "Democritus",
            "Pythagoras",
        ],
        "latin_authors": [
            "Cicero",
            "Seneca",
            "Lucretius",
            "Marcus Aurelius",
            "Boethius",
            "Augustine",
            "Quintilian",
            "Varro",
            "Lactantius",
            "Macrobius",
        ],
    },
    "metaphysics": {
        "templates": [
            "The essence of {term} manifests in {concept}, as {author} demonstrates.",
            "{author} shows how {term} transcends mere {concept}.",
            "Through {term}, we can understand the true nature of {concept}.",
            "{author}'s analysis reveals {term} as the foundation of {concept}.",
            "The metaphysical status of {term} determines our grasp of {concept}.",
        ],
        "concepts": [
            "form",
            "substance",
            "causation",
            "time",
            "space",
            "matter",
            "essence",
            "existence",
            "unity",
            "plurality",
        ],
    },
    "ethics": {
        "templates": [
            "{author} considers {term} essential to achieving {concept}.",
            "The practice of {term} leads to {concept}, according to {author}.",
            "{author} teaches that {term} is necessary for {concept}.",
            "True {concept} requires understanding of {term}, as {author} explains.",
            "In {author}'s ethical theory, {term} guides us toward {concept}.",
        ],
        "concepts": [
            "happiness",
            "virtue",
            "goodness",
            "justice",
            "courage",
            "temperance",
            "wisdom",
            "friendship",
            "duty",
            "excellence",
        ],
    },
}

# Define term pairs with their domains
TERM_PAIRS = {
    "etymological": [
        {
            "greek": "ἐπιστήμη",
            "latin": "scientia",
            "domains": ["philosophy"],
            "meaning": "knowledge, systematic understanding",
        },
        {
            "greek": "δικαιοσύνη",
            "latin": "iustitia",
            "domains": ["ethics"],
            "meaning": "justice, righteousness",
        },
        {
            "greek": "ἀλήθεια",
            "latin": "veritas",
            "domains": ["philosophy", "metaphysics"],
            "meaning": "truth, reality",
        },
        {
            "greek": "ψυχή",
            "latin": "anima",
            "domains": ["philosophy", "metaphysics"],
            "meaning": "soul, life force",
        },
        {
            "greek": "νοῦς",
            "latin": "intellectus",
            "domains": ["philosophy", "metaphysics"],
            "meaning": "mind, intellect, understanding",
        },
    ],
    "control": [
        {
            "greek": "ἐπιστήμη",
            "latin": "corpus",
            "domains": ["philosophy"],
            "meaning": "knowledge vs body",
        },
        {
            "greek": "δικαιοσύνη",
            "latin": "tempus",
            "domains": ["philosophy"],
            "meaning": "justice vs time",
        },
        {
            "greek": "ἀλήθεια",
            "latin": "mors",
            "domains": ["philosophy"],
            "meaning": "truth vs death",
        },
        {
            "greek": "ψυχή",
            "latin": "res",
            "domains": ["philosophy"],
            "meaning": "soul vs thing",
        },
        {
            "greek": "νοῦς",
            "latin": "vita",
            "domains": ["philosophy"],
            "meaning": "mind vs life",
        },
    ],
}


def generate_context(term: str, author: str, domain: str) -> str:
    """Generate a context for a given term using templates."""
    template = random.choice(CONTEXTS[domain]["templates"])
    concept = random.choice(CONTEXTS[domain]["concepts"])
    return template.format(term=term, author=author, concept=concept)


def generate_examples(
    num_examples: int = 50, output_dir: str = "data/synthetic"
) -> pd.DataFrame:
    """Generate synthetic examples for each term pair."""
    examples = []

    for pair_type, pairs in TERM_PAIRS.items():
        for pair in pairs:
            # Generate Greek examples
            for _ in range(num_examples):
                domain = random.choice(pair["domains"])
                author = random.choice(CONTEXTS["philosophy"]["greek_authors"])
                context = generate_context(pair["greek"], author, domain)

                examples.append(
                    {
                        "text": context,
                        "word": pair["greek"],
                        "language": "greek",
                        "pair_type": pair_type,
                        "domain": domain,
                        "author": author,
                        "meaning": pair["meaning"],
                    }
                )

            # Generate Latin examples
            for _ in range(num_examples):
                domain = random.choice(pair["domains"])
                author = random.choice(CONTEXTS["philosophy"]["latin_authors"])
                context = generate_context(pair["latin"], author, domain)

                examples.append(
                    {
                        "text": context,
                        "word": pair["latin"],
                        "language": "latin",
                        "pair_type": pair_type,
                        "domain": domain,
                        "author": author,
                        "meaning": pair["meaning"],
                    }
                )

    # Convert to DataFrame
    df = pd.DataFrame(examples)

    # Create output directory and save
    os.makedirs(output_dir, exist_ok=True)

    # Save the dataset
    output_path = os.path.join(output_dir, "contexts.csv")
    df.to_csv(output_path, index=False)

    # Save metadata
    metadata = {
        "total_examples": len(df),
        "examples_per_term": num_examples,
        "term_pairs": TERM_PAIRS,
        "domains": list(CONTEXTS.keys()),
        "stats": {
            "by_language": df.language.value_counts().to_dict(),
            "by_pair_type": df.pair_type.value_counts().to_dict(),
            "by_domain": df.domain.value_counts().to_dict(),
        },
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDataset generated and saved to {output_path}")
    print(f"Total examples: {len(df)}")
    print("\nExamples per category:")
    print(df.groupby(["language", "pair_type"]).size().unstack())

    return df


if __name__ == "__main__":
    # Generate 50 examples per term
    df = generate_examples(num_examples=50)
