import json
import os
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import torch

from sphilberta.evaluation import Evaluator
from sphilberta.model import SPHILBERTA

# Extended term pairs for more comprehensive analysis
TERM_PAIRS = {
    "etymological": [
        ("ἐπιστήμη", "scientia"),  # knowledge/science
        ("δικαιοσύνη", "iustitia"),  # justice
        ("ἀλήθεια", "veritas"),  # truth
        ("ψυχή", "anima"),  # soul
        ("νοῦς", "intellectus"),  # mind/intellect
    ],
    "control": [
        ("ἐπιστήμη", "corpus"),  # knowledge vs body
        ("δικαιοσύνη", "tempus"),  # justice vs time
        ("ἀλήθεια", "mors"),  # truth vs death
        ("ψυχή", "res"),  # soul vs thing
        ("νοῦς", "vita"),  # mind vs life
    ],
}


def safe_embed(text: str, model: SPHILBERTA, pooling: str = "cls") -> np.ndarray:
    try:
        inputs = model.tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        outputs = model.bert(**inputs)

        if pooling == "cls":
            return outputs.last_hidden_state[:, 0, :].detach().numpy()[0]
        else:
            return torch.mean(outputs.last_hidden_state, dim=1).detach().numpy()[0]
    except Exception as e:
        print(f"Error embedding text: {str(e)}")
        return np.zeros(768)


def angular_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_sim = dot_product / norms if norms > 0 else 0
    return float(1 - (2 / np.pi) * np.arccos(np.clip(cos_sim, -1.0, 1.0)))


def run_experiments(
    model: SPHILBERTA, contexts_df: pd.DataFrame, output_dir: str = "outputs"
) -> Dict:
    """Run enhanced experiments with more term pairs and analysis"""

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "results"), exist_ok=True)

    # Initialize evaluator
    evaluator = Evaluator(model, None)  # We'll handle data loading separately

    # Calculate similarities for all term pairs
    print("\n1. Calculating cross-lingual similarities...")
    similarity_data = []

    for pair_type, pairs in TERM_PAIRS.items():
        for gr_word, la_word in pairs:
            # Get contexts for both terms
            gr_contexts = contexts_df[contexts_df["word"] == gr_word]
            la_contexts = contexts_df[contexts_df["word"] == la_word]

            if len(gr_contexts) == 0 or len(la_contexts) == 0:
                print(f"Warning: No contexts found for {gr_word}-{la_word}")
                continue

            # Get embeddings
            gr_embeds = np.vstack(
                [safe_embed(text, model) for text in gr_contexts["text"].tolist()]
            )
            la_embeds = np.vstack(
                [safe_embed(text, model) for text in la_contexts["text"].tolist()]
            )

            # Compute mean embeddings
            gr_mean = np.mean(gr_embeds, axis=0)
            la_mean = np.mean(la_embeds, axis=0)

            # Compute similarity
            sim = angular_similarity(gr_mean, la_mean)

            similarity_data.append(
                {
                    "Greek": gr_word,
                    "Latin": la_word,
                    "similarity": sim,
                    "pair_type": pair_type,
                    "greek_contexts": len(gr_contexts),
                    "latin_contexts": len(la_contexts),
                }
            )

    similarities_df = pd.DataFrame(similarity_data)

    # Generate visualizations
    print("\n2. Generating visualizations...")

    # Similarity heatmap
    evaluator.plot_similarity_heatmap(
        similarities_df, os.path.join(run_dir, "figures", "similarity_heatmap.png")
    )

    # Distribution plots
    evaluator.plot_similarity_distribution(
        similarities_df, os.path.join(run_dir, "figures", "similarity_distribution.png")
    )

    # t-SNE visualization
    evaluator.plot_tsne_projection(
        similarities_df, os.path.join(run_dir, "figures", "tsne_projection.png")
    )

    # Additional plots
    evaluator.plot_similarity_histogram(
        similarities_df, os.path.join(run_dir, "figures", "similarity_histogram.png")
    )

    evaluator.plot_similarity_boxplot(
        similarities_df, os.path.join(run_dir, "figures", "similarity_boxplot.png")
    )

    # Generate analysis report
    print("\n3. Generating analysis report...")
    evaluator.generate_analysis_report(
        similarities_df, os.path.join(run_dir, "results", "analysis_report.txt")
    )

    # Save results
    similarities_df.to_csv(
        os.path.join(run_dir, "results", "similarities.csv"), index=False
    )

    # Compute and save statistics
    stats = {
        "etymological_mean": float(
            similarities_df[similarities_df["pair_type"] == "etymological"][
                "similarity"
            ].mean()
        ),
        "control_mean": float(
            similarities_df[similarities_df["pair_type"] == "control"][
                "similarity"
            ].mean()
        ),
        "etymological_std": float(
            similarities_df[similarities_df["pair_type"] == "etymological"][
                "similarity"
            ].std()
        ),
        "control_std": float(
            similarities_df[similarities_df["pair_type"] == "control"][
                "similarity"
            ].std()
        ),
        "total_pairs": len(similarities_df),
        "total_greek_terms": len(similarities_df["Greek"].unique()),
        "total_latin_terms": len(similarities_df["Latin"].unique()),
    }

    with open(os.path.join(run_dir, "results", "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nExperiments complete! Results saved to {run_dir}")
    return stats


if __name__ == "__main__":
    # Load model and data
    model = SPHILBERTA()
    contexts_df = pd.read_csv("data/contexts.csv")  # You'll need to prepare this

    # Run experiments
    stats = run_experiments(model, contexts_df)
    print("\nKey findings:")
    print(
        f"Etymological pairs: {stats['etymological_mean']:.3f} ± {stats['etymological_std']:.3f}"
    )
    print(f"Control pairs: {stats['control_mean']:.3f} ± {stats['control_std']:.3f}")
