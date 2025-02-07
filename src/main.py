import argparse
import os
from datetime import datetime

import pandas as pd

from run_experiments import run_experiments
from sphilberta.model import SPHILBERTA


def main(args):
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("=== Greek-Latin Cross-Lingual Semantic Analysis ===")
    print(f"Output directory: {output_dir}")

    # Step 1: Load synthetic dataset
    print("\nStep 1: Loading dataset...")
    contexts_df = pd.read_csv(os.path.join(args.data_dir, "contexts.csv"))
    print(f"Loaded {len(contexts_df)} examples")
    print("\nDistribution:")
    print(contexts_df.groupby(["language", "pair_type"]).size().unstack())

    # Step 2: Initialize model
    print("\nStep 2: Loading SPHILBERTA model...")
    model = SPHILBERTA(base_model=args.model_base)

    # Step 3: Run experiments
    print("\nStep 3: Running experiments...")
    stats = run_experiments(model=model, contexts_df=contexts_df, output_dir=output_dir)

    print("\n=== Experiment Complete ===")
    print(f"Results saved to: {output_dir}")
    print("\nKey Findings:")
    print(f"- Total term pairs analyzed: {stats['total_pairs']}")
    print(
        f"- Etymological pairs: {stats['etymological_mean']:.3f} ± {stats['etymological_std']:.3f}"
    )
    print(f"- Control pairs: {stats['control_mean']:.3f} ± {stats['control_std']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Greek-Latin semantic analysis experiments"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/synthetic",
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--min_contexts",
        type=int,
        default=50,
        help="Minimum number of contexts per term",
    )
    parser.add_argument(
        "--model_base",
        type=str,
        default="bert-base-multilingual-cased",
        help="Base model to use for SPHILBERTA",
    )

    args = parser.parse_args()
    main(args)
