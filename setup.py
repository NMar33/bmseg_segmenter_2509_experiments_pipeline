from setuptools import setup, find_packages
from pathlib import Path

# --- Best Practice: Read requirements from a file ---
# This ensures that `pip install -r requirements.txt` and `pip install .`
# use the same single source of truth for dependencies.
try:
    with open('requirements.txt', encoding="utf-8") as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    print("Warning: requirements.txt not found. Using a default list of dependencies.")
    # DEV: Это запасной вариант на случай, если requirements.txt отсутствует.
    requirements = [
        "numpy", "pyyaml", "opencv-python-headless", "tqdm", "pandas",
        "scikit-image", "scipy", "torch", "torchvision",
        "segmentation-models-pytorch", "albumentations", "mlflow", "matplotlib"
    ]

# --- Setup Configuration ---
setup(
    # The name of the package as it will be listed on PyPI or installed via pip.
    # It's good practice for this to match the repository name.
    name="segmentation-workbench",
    version="0.1.0",
    description="A workbench for conducting segmentation experiments, focusing on domain shift and feature banks.",
    author="<Your Name>", # <-- Replace with your name
    author_email="<your.email@example.com>", # <-- and your email
    url="<URL to your GitHub repo>", # <-- Add the URL to your project repo

    # --- Package Discovery ---
    # DEV: Это самая важная часть, исправленная согласно нашему обсуждению.
    # Мы НЕ используем `package_dir`.
    # `find_packages()` найдет один пакет верхнего уровня: `segwork`.
    # Это гарантирует, что все импорты должны будут начинаться с `segwork.`,
    # что предотвращает конфликты имен.
    packages=find_packages(),

    # Dependencies that will be installed with the package.
    install_requires=requirements,

    # --- Console Scripts ---
    # This creates command-line executables for our scripts.
    # DEV: Имена команд с префиксом `segwork-` - это лучшая практика,
    # чтобы избежать конфликтов с другими системными утилитами.
    entry_points={
        'console_scripts': [
            'segwork-preprocess = scripts.preprocess:main',
            'segwork-train = scripts.train:main',
            'segwork-evaluate = scripts.evaluate:main',
            'segwork-visualize = scripts.visualize_filters:main',
            'segwork-explore = scripts.explore_dataset:main', # Assuming you add this
            'segwork-run-exp = scripts.run_experiment:main',
            'segwork-make-report = scripts.make_reports:main',
        ],
    },

    # --- Additional Metadata ---
    python_requires='>=3.9',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License", # Example, choose your own license
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)