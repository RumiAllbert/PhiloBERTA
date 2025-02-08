import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from numpy.typing import NDArray
from scipy import stats
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, model, data_dir: Optional[str] = None):
        self.model = model
        self.data_dir = data_dir
        self.analysis_results = {}
        self.set_plot_style()

        # Expert judgment scores (from paper)
        self.expert_scores = {
            ("λόγος", "ratio"): 0.68,
            ("ψυχή", "anima"): 0.72,
            ("ἀρετή", "virtus"): 0.65,
            ("σοφία", "sapientia"): 0.70,
            ("νοῦς", "intellectus"): 0.69,
        }

    @staticmethod
    def set_plot_style() -> None:
        """Configure publication-quality plot settings"""
        plt.style.use("seaborn-v0_8-paper")  # Modern, professional base style

        # Custom color palette - using a professional, colorblind-friendly scheme
        colors = [
            "#4477AA",
            "#EE6677",
            "#228833",
            "#CCBB44",
            "#66CCEE",
            "#AA3377",
            "#BBBBBB",
        ]
        sns.set_palette(colors)

        plt.rcParams.update(
            {
                # Font settings
                "font.family": ["Times New Roman", "DejaVu Serif"],
                "font.size": 12,
                "axes.titlesize": 16,
                "axes.labelsize": 14,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                # Figure settings
                "figure.dpi": 300,
                "figure.figsize": (12, 8),
                "figure.facecolor": "white",
                "figure.autolayout": True,
                # Axes settings
                "axes.grid": True,
                "grid.alpha": 0.3,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.facecolor": "white",
                # Legend settings
                "legend.frameon": False,
                "legend.fontsize": 12,
                # Saving settings
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.2,
            }
        )

    def compute_genre_variance(self, similarities_df: pd.DataFrame) -> Dict[str, float]:
        """Compute variance in similarities across genres"""
        genre_vars = {}
        for genre in similarities_df.genre.unique():
            genre_sims = similarities_df[similarities_df.genre == genre].similarity
            genre_vars[genre] = float(np.var(genre_sims, ddof=1))
        return genre_vars

    def compute_expert_correlation(
        self, similarities_df: pd.DataFrame
    ) -> Tuple[float, float]:
        """Compute Pearson correlation with expert judgments"""
        model_scores = []
        expert_scores = []

        for pair, expert_score in self.expert_scores.items():
            pair_sims = similarities_df[
                (similarities_df.Greek == pair[0]) & (similarities_df.Latin == pair[1])
            ]
            if not pair_sims.empty:
                model_scores.append(float(pair_sims.similarity.mean()))
                expert_scores.append(expert_score)

        if model_scores:
            correlation = stats.pearsonr(model_scores, expert_scores)
            return float(correlation[0]), float(correlation[1])  # r, p-value
        return 0.0, 1.0

    def perform_anova_analysis(self, similarities_df: pd.DataFrame) -> Dict[str, float]:
        """Perform ANOVA analysis across genres"""
        groups = []
        for genre in similarities_df.genre.unique():
            genre_sims = similarities_df[similarities_df.genre == genre].similarity
            groups.append(genre_sims)

        if len(groups) > 1:
            f_stat, p_val = stats.f_oneway(*groups)
            return {
                "f_statistic": float(f_stat),
                "p_value": float(p_val),
                "df_between": len(groups) - 1,
                "df_within": sum(len(g) for g in groups) - len(groups),
            }
        return {"f_statistic": 0.0, "p_value": 1.0, "df_between": 0, "df_within": 0}

    def compute_similarities(
        self, term_pairs: List[Tuple[str, str]], contexts_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute cross-lingual similarities with enhanced statistical analysis"""
        results = []
        control_pairs = []

        # Generate control pairs by shuffling Latin terms
        latin_terms = [pair[1] for pair in term_pairs]
        np.random.shuffle(latin_terms)
        for i, (greek_term, _) in enumerate(term_pairs):
            control_pairs.append((greek_term, latin_terms[i]))

        # Process both etymological and control pairs
        for pair_type, pairs in [
            ("etymological", term_pairs),
            ("control", control_pairs),
        ]:
            for greek_term, latin_term in pairs:
                greek_contexts = contexts_df[contexts_df.target_word == greek_term]
                latin_contexts = contexts_df[contexts_df.target_word == latin_term]

                if len(greek_contexts) == 0 or len(latin_contexts) == 0:
                    logger.warning(
                        f"No contexts found for pair: {greek_term} - {latin_term}"
                    )
                    continue

                try:
                    # Get embeddings and compute similarity
                    greek_embeds: NDArray[np.float64] = self.model.get_embeddings(
                        greek_contexts.text.tolist()
                    )
                    latin_embeds: NDArray[np.float64] = self.model.get_embeddings(
                        latin_contexts.text.tolist()
                    )

                    # Compute mean embeddings
                    greek_mean: NDArray[np.float64] = np.mean(greek_embeds, axis=0)
                    latin_mean: NDArray[np.float64] = np.mean(latin_embeds, axis=0)

                    # Compute similarity
                    similarity = self.compute_angular_similarity(greek_mean, latin_mean)

                    # Store result with additional metadata
                    results.append(
                        {
                            "Greek": greek_term,
                            "Latin": latin_term,
                            "similarity": similarity,
                            "pair_type": pair_type,
                            "greek_contexts": len(greek_contexts),
                            "latin_contexts": len(latin_contexts),
                            "genre": greek_contexts.genre.mode()[0]
                            if not greek_contexts.empty
                            else "unknown",
                        }
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing pair {greek_term} - {latin_term}: {str(e)}"
                    )
                    continue

        # Create DataFrame and compute statistics
        df = pd.DataFrame(results)

        if df.empty:
            logger.warning("No valid similarity results found")
            return df

        try:
            # Compute basic statistics
            etymological_sims: NDArray[np.float64] = df[
                df["pair_type"] == "etymological"
            ].similarity.to_numpy()
            control_sims: NDArray[np.float64] = df[
                df["pair_type"] == "control"
            ].similarity.to_numpy()

            if len(etymological_sims) == 0 or len(control_sims) == 0:
                logger.warning("Not enough data for statistical analysis")
                return df

            # Perform statistical tests
            test_result = stats.ttest_ind(etymological_sims, control_sims)

            # Calculate effect size
            cohens_d = self.compute_cohens_d(etymological_sims, control_sims)

            # Compute genre variance
            genre_variance = self.compute_genre_variance(df)

            # Compute expert correlation
            expert_corr, expert_p = self.compute_expert_correlation(df)

            # Perform ANOVA
            anova_results = self.perform_anova_analysis(df)

            # Store comprehensive analysis results
            self.analysis_results.update(
                {
                    "etymological_mean": float(np.mean(etymological_sims)),
                    "control_mean": float(np.mean(control_sims)),
                    "etymological_std": float(np.std(etymological_sims, ddof=1)),
                    "control_std": float(np.std(control_sims, ddof=1)),
                    "t_statistic": float(test_result.statistic),
                    "p_value": float(test_result.pvalue),
                    "cohens_d": float(cohens_d),
                    "expert_correlation": expert_corr,
                    "expert_correlation_p": expert_p,
                    "anova_f": anova_results["f_statistic"],
                    "anova_p": anova_results["p_value"],
                    **{
                        f"variance_{genre}": var
                        for genre, var in genre_variance.items()
                    },
                }
            )
        except Exception as e:
            logger.error(f"Error in statistical analysis: {str(e)}")

        return df

    @staticmethod
    def compute_angular_similarity(
        v1: NDArray[np.float64], v2: NDArray[np.float64]
    ) -> float:
        """Compute angular similarity between two vectors"""
        dot_product = float(np.dot(v1, v2))
        norms = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_sim = dot_product / norms if norms > 0 else 0.0
        return float(1 - (2 / np.pi) * np.arccos(np.clip(cos_sim, -1.0, 1.0)))

    @staticmethod
    def compute_cohens_d(
        group1: NDArray[np.float64], group2: NDArray[np.float64]
    ) -> float:
        """Compute Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        var1 = float(np.var(group1, ddof=1))
        var2 = float(np.var(group2, ddof=1))
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        mean_diff = float(np.mean(group1) - np.mean(group2))
        return mean_diff / float(pooled_std) if pooled_std > 0 else 0.0

    def plot_similarity_heatmap(self, df: pd.DataFrame, output_path: str):
        """Plot enhanced similarity heatmap with better aesthetics"""
        # Calculate statistics for title
        etymological_mean = df[df["pair_type"] == "etymological"]["similarity"].mean()
        etymological_std = df[df["pair_type"] == "etymological"]["similarity"].std()
        control_mean = df[df["pair_type"] == "control"]["similarity"].mean()
        control_std = df[df["pair_type"] == "control"]["similarity"].std()

        # Create pivot table for heatmap
        pivot_df = df.pivot(index="Greek", columns="Latin", values="similarity")

        # Set up the figure with golden ratio
        plt.figure(figsize=(12, 7.416))

        # Create heatmap with enhanced aesthetics
        sns.heatmap(
            pivot_df,
            annot=True,
            cmap="RdYlBu_r",
            center=0.5,
            vmin=0.7,
            vmax=0.9,
            fmt=".3f",
            square=True,
            cbar_kws={"label": "Semantic Similarity"},
            annot_kws={"size": 10},
        )

        # Enhance the title and labels
        plt.title("Cross-Lingual Semantic Similarity Matrix", pad=20, fontsize=16)
        plt.xlabel("Latin Terms", labelpad=10)
        plt.ylabel("Greek Terms", labelpad=10)

        # Add subtitle with statistics
        subtitle = (
            f"Etymological pairs: {etymological_mean:.3f} ± {etymological_std:.3f}\n"
            f"Control pairs: {control_mean:.3f} ± {control_std:.3f}"
        )
        plt.suptitle(subtitle, y=0.95, fontsize=12)

        # Save with high quality
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_similarity_distribution(self, df: pd.DataFrame, output_path: str):
        """Plot enhanced distribution of similarities by pair type"""
        plt.figure(figsize=(12, 7.416))

        # Create violin plot with individual points
        ax = sns.violinplot(
            data=df,
            x="pair_type",
            y="similarity",
            inner="box",
            hue="pair_type",
            legend=False,
        )

        # Add individual points
        sns.stripplot(
            data=df,
            x="pair_type",
            y="similarity",
            color="black",
            alpha=0.4,
            jitter=0.2,
            size=5,
            ax=ax,
        )

        # Enhance labels and title
        plt.title("Distribution of Semantic Similarities", pad=20)
        plt.xlabel("Pair Type", labelpad=10)
        plt.ylabel("Semantic Similarity", labelpad=10)

        # Add statistical annotation
        t_stat, p_val = stats.ttest_ind(
            df[df["pair_type"] == "etymological"]["similarity"],
            df[df["pair_type"] == "control"]["similarity"],
        )
        plt.text(
            0.02,
            0.98,
            f"t-statistic: {t_stat:.3f}\np-value: {p_val:.6f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_similarity_radar(self, df: pd.DataFrame, output_path: str):
        """Create a radar plot comparing etymological vs control pairs"""
        # Prepare data for radar plot
        categories = df["Greek"].unique()
        etymological_vals = []
        control_vals = []

        for greek_term in categories:
            etym_sim = df[
                (df["Greek"] == greek_term) & (df["pair_type"] == "etymological")
            ]["similarity"].values[0]
            ctrl_sim = df[(df["Greek"] == greek_term) & (df["pair_type"] == "control")][
                "similarity"
            ].values[0]
            etymological_vals.append(etym_sim)
            control_vals.append(ctrl_sim)

        # Convert to numpy arrays for manipulation
        categories = np.array(categories)
        etymological_vals = np.array(etymological_vals)
        control_vals = np.array(control_vals)

        # Create angles for the radar plot
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)

        # Close the circle by appending the first value
        categories = np.concatenate([categories, [categories[0]]])
        etymological_vals = np.concatenate([etymological_vals, [etymological_vals[0]]])
        control_vals = np.concatenate([control_vals, [control_vals[0]]])
        angles = np.concatenate([angles, [angles[0]]])

        # Create radar plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        # Plot data
        ax.plot(
            angles,
            etymological_vals,
            "o-",
            linewidth=2,
            label="Etymological",
            color="#4477AA",
        )
        ax.fill(angles, etymological_vals, alpha=0.25, color="#4477AA")
        ax.plot(
            angles, control_vals, "o-", linewidth=2, label="Control", color="#EE6677"
        )
        ax.fill(angles, control_vals, alpha=0.25, color="#EE6677")

        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories[:-1])

        # Add legend and title
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
        plt.title("Semantic Similarity Comparison\nRadar Plot", pad=20)

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_paired_comparison(self, df: pd.DataFrame, output_path: str):
        """Create a paired comparison plot showing etymological vs control pairs for each Greek term"""
        plt.figure(figsize=(12, 7.416))

        # Prepare data
        greek_terms = df["Greek"].unique()
        x = np.arange(len(greek_terms))
        width = 0.35

        etymological_vals = [
            df[(df["Greek"] == term) & (df["pair_type"] == "etymological")][
                "similarity"
            ].values[0]
            for term in greek_terms
        ]
        control_vals = [
            df[(df["Greek"] == term) & (df["pair_type"] == "control")][
                "similarity"
            ].values[0]
            for term in greek_terms
        ]

        # Create bars
        ax = plt.gca()
        rects1 = ax.bar(
            x - width / 2,
            etymological_vals,
            width,
            label="Etymological",
            color="#4477AA",
        )
        rects2 = ax.bar(
            x + width / 2, control_vals, width, label="Control", color="#EE6677"
        )

        # Customize plot
        ax.set_ylabel("Semantic Similarity")
        ax.set_title("Paired Comparison of Semantic Similarities")
        ax.set_xticks(x)
        ax.set_xticklabels(greek_terms, rotation=45, ha="right")
        ax.legend()

        # Add value labels on bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=8,
                )

        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_tsne_projection(self, df: pd.DataFrame, output_path: str):
        """Create t-SNE projection of term embeddings."""
        # Get unique terms and their embeddings
        terms = pd.concat(
            [
                df[["Greek", "pair_type"]].rename(columns={"Greek": "term"}),
                df[["Latin", "pair_type"]].rename(columns={"Latin": "term"}),
            ]
        ).drop_duplicates()

        # Get embeddings for all terms
        embeddings = []
        for term in terms["term"]:
            # Generate a simple context
            context = f"The term {term} is important."
            inputs = self.model.tokenizer(
                context,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self.model.bert(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                embeddings.append(embedding)

        # Perform t-SNE with adjusted parameters for small dataset
        tsne = TSNE(
            n_components=2,
            perplexity=min(
                30, len(terms) - 1
            ),  # Adjust perplexity based on dataset size
            random_state=42,
            n_iter=1000,  # Increase iterations for better convergence
            learning_rate="auto",  # Let TSNE choose the best learning rate
        )
        embeddings_2d = tsne.fit_transform(np.array(embeddings))

        # Create plot
        plt.figure(figsize=(12, 8))

        # Plot points
        for pair_type in ["etymological", "control"]:
            mask = terms["pair_type"] == pair_type
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                label=pair_type,
                alpha=0.6,
            )

        # Add labels
        for i, term in enumerate(terms["term"]):
            plt.annotate(
                term,
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        plt.title("t-SNE Projection of Term Embeddings")
        plt.legend()

        # Save plot
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_similarity_histogram(self, df: pd.DataFrame, output_path: str):
        """Plot histogram of similarities."""
        plt.figure(figsize=(10, 6))

        # Create histograms
        for pair_type in ["etymological", "control"]:
            sns.histplot(
                data=df[df["pair_type"] == pair_type],
                x="similarity",
                label=pair_type,
                alpha=0.5,
                bins=15,
            )

        plt.title("Distribution of Semantic Similarities")
        plt.xlabel("Semantic Similarity")
        plt.ylabel("Count")
        plt.legend()

        # Save plot
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_similarity_boxplot(self, df: pd.DataFrame, output_path: str):
        """Plot boxplot of similarities."""
        plt.figure(figsize=(10, 6))

        # Create box plot
        sns.boxplot(data=df, x="pair_type", y="similarity")

        # Add individual points
        sns.stripplot(
            data=df,
            x="pair_type",
            y="similarity",
            color="red",
            alpha=0.3,
            jitter=0.2,
            size=4,
        )

        plt.title("Semantic Similarity by Pair Type")
        plt.xlabel("Pair Type")
        plt.ylabel("Semantic Similarity")

        # Save plot
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def generate_analysis_report(self, df: pd.DataFrame, output_path: str):
        """Generate a detailed analysis report."""
        # Calculate statistics
        stats = {
            "etymological": {
                "mean": df[df["pair_type"] == "etymological"]["similarity"].mean(),
                "std": df[df["pair_type"] == "etymological"]["similarity"].std(),
                "min": df[df["pair_type"] == "etymological"]["similarity"].min(),
                "max": df[df["pair_type"] == "etymological"]["similarity"].max(),
            },
            "control": {
                "mean": df[df["pair_type"] == "control"]["similarity"].mean(),
                "std": df[df["pair_type"] == "control"]["similarity"].std(),
                "min": df[df["pair_type"] == "control"]["similarity"].min(),
                "max": df[df["pair_type"] == "control"]["similarity"].max(),
            },
        }

        # Perform t-test
        from scipy import stats as scipy_stats

        etymological = df[df["pair_type"] == "etymological"]["similarity"]
        control = df[df["pair_type"] == "control"]["similarity"]
        t_stat, p_value = scipy_stats.ttest_ind(etymological, control)

        # Generate report
        report = [
            "=== Cross-Lingual Semantic Analysis Report ===\n",
            "\nEtymologically Related Pairs:",
            f"- Mean similarity: {stats['etymological']['mean']:.3f} ± {stats['etymological']['std']:.3f}",
            f"- Range: [{stats['etymological']['min']:.3f}, {stats['etymological']['max']:.3f}]",
            "\nControl Pairs:",
            f"- Mean similarity: {stats['control']['mean']:.3f} ± {stats['control']['std']:.3f}",
            f"- Range: [{stats['control']['min']:.3f}, {stats['control']['max']:.3f}]",
            "\nStatistical Analysis:",
            f"- T-statistic: {t_stat:.3f}",
            f"- P-value: {p_value:.6f}",
            "\nTop Similar Pairs:",
        ]

        # Add top similar pairs
        top_pairs = df.nlargest(5, "similarity")
        for _, row in top_pairs.iterrows():
            report.append(
                f"- {row['Greek']} - {row['Latin']}: {row['similarity']:.3f} ({row['pair_type']})"
            )

        # Save report
        with open(output_path, "w") as f:
            f.write("\n".join(report))
