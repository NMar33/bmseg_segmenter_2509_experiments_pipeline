from setuptools import setup, find_packages

setup(
    name="ml_pipeline",
    version="0.1.0",
    # DEV: find_packages() автоматически найдет все наши модули (data, models, etc.)
    # внутри папки src.
    packages=find_packages(where="src"),
    # DEV: Эта строка важна! Она говорит, что корень пакета ('') находится в 'src'.
    # Это позволяет делать импорты как `from data.dataset import ...` вместо `from src.data.dataset ...`
    package_dir={'': 'src'},

    # DEV: Здесь мы перечисляем зависимости, нужные ТОЛЬКО для ML-экспериментов.
    install_requires=[
        "numpy",
        "pyyaml",
        "opencv-python-headless",
        "tqdm",
        "pandas", # для make_reports.py
        "scikit-image", # для метрик и фильтров
        "scipy", # для метрик
        "torch",
        "torchvision",
        "segmentation-models-pytorch",
        "albumentations",
        "mlflow",
        # DEV: Убедись, что все нужные библиотеки здесь
    ],

    # DEV: Это самая крутая часть! Мы создаем удобные консольные команды
    # для запуска наших скриптов. Вместо `python scripts/train.py` можно будет
    # просто писать `train-model`.
    entry_points={
        'console_scripts': [
            'preprocess-data = scripts.preprocess:main',
            'train-model = scripts.train:main',
            'evaluate-model = scripts.evaluate:main',
            'run-experiment = scripts.run_experiment:main',
            'make-reports = scripts.make_reports:main',
        ],
    },
)