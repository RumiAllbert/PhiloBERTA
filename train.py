import argparse
import json
import os
from datetime import datetime

from sphilberta.data import CorpusLoader
from sphilberta.evaluation import Evaluator
from sphilberta.model import SPHILBERTA


def setup_output_dirs(base_dir="outputs"):
    """Create timestamped output directories"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")

    # Create directories
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "results"), exist_ok=True)

    return run_dir


def train(args):
    # Setup output directories
    run_dir = setup_output_dirs()

    # Initialize components
    model = SPHILBERTA()
    data_loader = CorpusLoader()
    evaluator = Evaluator(model, data_loader)

    # Load data
    print("Loading Perseus corpus...")
    corpus_df = data_loader.load_perseus_corpus()
    term_pairs = data_loader.get_term_pairs()

    # Extract contexts
    print("Extracting term contexts...")
    greek_terms = [pair[0] for pair in term_pairs]
    latin_terms = [pair[1] for pair in term_pairs]
    contexts = data_loader.extract_contexts(greek_terms + latin_terms, corpus_df)

    # Save context statistics
    context_stats = {
        "total_contexts": len(contexts),
        "contexts_per_word": contexts.groupby("target_word").size().to_dict(),
        "contexts_per_genre": contexts.groupby("genre").size().to_dict(),
    }

    with open(os.path.join(run_dir, "results", "context_stats.json"), "w") as f:
        json.dump(context_stats, f, indent=2)

    # Compute similarities
    print("Computing cross-lingual similarities...")
    similarities = evaluator.compute_similarities(term_pairs, contexts)

    # Save similarity results
    similarities.to_csv(
        os.path.join(run_dir, "results", "similarities.csv"), index=False
    )

    # Generate visualizations
    print("Generating visualizations...")

    # 1. Similarity heatmap
    print("1. Creating similarity heatmap...")
    evaluator.plot_similarity_heatmap(
        similarities, os.path.join(run_dir, "figures", "similarity_heatmap.png")
    )

    # 2. Similarity distribution
    print("2. Creating similarity distribution plot...")
    evaluator.plot_similarity_distribution(
        similarities, os.path.join(run_dir, "figures", "similarity_distribution.png")
    )

    # 3. t-SNE projection
    print("3. Computing embeddings for t-SNE visualization...")
    all_texts = contexts.text.tolist()
    embeddings = model.get_embeddings(all_texts)

    evaluator.plot_tsne_projection(
        similarities, os.path.join(run_dir, "figures", "tsne_projection.png")
    )

    # 4. Similarity histogram
    print("4. Creating similarity histogram...")
    evaluator.plot_similarity_histogram(
        similarities, os.path.join(run_dir, "figures", "similarity_histogram.png")
    )

    # 5. Similarity boxplot
    print("5. Creating similarity boxplot...")
    evaluator.plot_similarity_boxplot(
        similarities, os.path.join(run_dir, "figures", "similarity_boxplot.png")
    )

    # Generate analysis report
    print("6. Generating detailed analysis report...")
    evaluator.generate_analysis_report(
        similarities, os.path.join(run_dir, "results", "analysis_report.txt")
    )

    # Save run configuration
    config = {
        "corpus_size": len(corpus_df),
        "total_contexts": len(contexts),
        "greek_terms": greek_terms,
        "latin_terms": latin_terms,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    with open(os.path.join(run_dir, "results", "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nAnalysis complete! Results saved to {run_dir}")
    print("\nGenerated files:")
    print("1. Visualizations:")
    print(
        f"   - Similarity Heatmap: {os.path.join('figures', 'similarity_heatmap.png')}"
    )
    print(
        f"   - Similarity Distribution: {os.path.join('figures', 'similarity_distribution.png')}"
    )
    print(f"   - t-SNE Projection: {os.path.join('figures', 'tsne_projection.png')}")
    print(
        f"   - Similarity Histogram: {os.path.join('figures', 'similarity_histogram.png')}"
    )
    print(
        f"   - Similarity Boxplot: {os.path.join('figures', 'similarity_boxplot.png')}"
    )
    print("\n2. Analysis:")
    print(f"   - Full Report: {os.path.join('results', 'analysis_report.txt')}")
    print(f"   - Similarity Data: {os.path.join('results', 'similarities.csv')}")
    print(f"   - Context Statistics: {os.path.join('results', 'context_stats.json')}")
    print(f"   - Run Configuration: {os.path.join('results', 'config.json')}")

    # Print key findings
    print("\nKey Findings:")
    print(
        f"- Etymological pairs mean similarity: {evaluator.analysis_results['etymological_mean']:.3f}"
    )
    print(
        f"- Control pairs mean similarity: {evaluator.analysis_results['control_mean']:.3f}"
    )
    print(
        f"- Statistical significance: p = {evaluator.analysis_results['p_value']:.3e}"
    )
    print(f"- Effect size (Cohen's d): {evaluator.analysis_results['cohens_d']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    train(args)
