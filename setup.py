from setuptools import find_packages, setup

setup(
    name="philoberta",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "cltk>=1.1.0",
        "datasets>=2.12.0",
        "tqdm>=4.65.0",
    ],
    author="Agent Laboratory",
    description="Cross-Lingual Semantic Analysis of Ancient Greek Philosophical Terms",
    python_requires=">=3.8",
)
