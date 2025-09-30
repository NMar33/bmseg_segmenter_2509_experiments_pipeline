# setup.py

from setuptools import setup, find_packages
from pathlib import Path

# --- Best Practice: Read requirements from a file ---
# This ensures that `pip install -r requirements.txt` and `pip install .`
# use the same single source of truth for dependencies.
try:
    with open('requirements.txt', encoding="utf-8") as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    print("Warning: requirements.txt not found. Dependencies will not be installed.")
    requirements = []

# --- Setup Configuration ---
setup(
    # The name of the package as it will be listed on PyPI or installed via pip.
    name="segmentation-workbench",
    version="0.2.0", # Bump version to reflect architectural changes
    description="A workbench for conducting segmentation experiments, focusing on domain shift and feature banks.",
    author="<Your Name>",
    author_email="<your.email@example.com>",
    url="<URL to your GitHub repo>",

    # --- Package Discovery ---
    # `find_packages()` will automatically find the `segwork` package.
    # This ensures that all imports must start with `from segwork. ...`,
    # preventing naming conflicts.
    packages=find_packages(),

    # Dependencies that will be installed with the package.
    install_requires=requirements,

    # --- Console Scripts ---
    # This creates command-line executables for our scripts.
    # Using a prefix like `segwork-` is a best practice to avoid conflicts.
    entry_points={
        'console_scripts': [
            # Main experiment orchestrator
            'segwork-run-experiment = scripts.run_experiment:main',
            
            # Individual pipeline steps
            'segwork-preprocess-tiles = scripts.preprocess_tiles:main',
            'segwork-generate-splits = scripts.generate_splits:main',
            'segwork-train = scripts.train:main',
            'segwork-evaluate = scripts.evaluate:main',

            # Reporting and analysis
            'segwork-make-reports = scripts.make_reports:main',
            
            # Visualization tools (assuming they exist)
            'segwork-visualize-filters = scripts.visualize_filters:main',
        ],
    },

    # --- Additional Metadata ---
    python_requires='>=3.9',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License", # Example, choose your license
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)