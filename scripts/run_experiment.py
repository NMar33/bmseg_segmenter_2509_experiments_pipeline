# scripts/run_experiment.py

"""
Main orchestrator script for running end-to-end experiments.
This script manages preprocessing, training, and evaluation for different
scenarios like in-domain, cross-domain, and few-shot learning across
multiple models and random seeds.
"""
import argparse
import subprocess
from pathlib import Path
import yaml
import shutil
import random

# DEV: Мы "хардкодим" здесь структуру наших экспериментов.
# Это "план статьи" в виде кода.
# Ключи (M1, M2, M3) соответствуют нашим моделям.
# Значения - это пути к базовым конфигам для каждой модели на каждом датасете.
EXPERIMENT_CONFIGS = {
    "isbi": {
        "M1_Baseline": "configs/exp/isbi_baseline.yaml",
        "M2_Adapter": "configs/exp/isbi_adapter.yaml",
        "M3_FeatureBank": "configs/exp/isbi_feature_bank.yaml",
    },
    "urisc": {
        "M1_Baseline": "configs/exp/urisc_baseline.yaml",
        "M2_Adapter": "configs/exp/urisc_adapter.yaml",
        "M3_FeatureBank": "configs/exp/urisc_feature_bank.yaml",
    }
}

def run_command(command: str):
    """Runs a command in a subprocess and checks for errors."""
    print(f"\n>>> RUNNING: {command}")
    subprocess.run(command, shell=True, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run a full experiment suite.")
    parser.add_argument("--experiment", required=True, choices=["A_InDomain", "B_CrossDomain", "C_FewShot"], help="Which experiment to run.")
    parser.add_argument("--source_domain", default="isbi", choices=["isbi", "urisc"], help="Dataset for initial training.")
    parser.add_argument("--target_domain", default="urisc", choices=["isbi", "urisc"], help="Dataset for transfer/evaluation.")
    parser.add_argument("--seeds", default="42,1337,2023", help="Comma-separated list of random seeds.")
    parser.add_argument("--few_shot_fracs", default="0.01,0.05", help="Comma-separated list of fractions for few-shot learning.")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    # --- Эксперимент A: Обучение и оценка в исходном домене ---
    if args.experiment == "A_InDomain":
        print(f"\n--- Running Experiment A: In-Domain Performance on '{args.source_domain}' ---")
        model_configs = EXPERIMENT_CONFIGS[args.source_domain]
        
        for model_name, base_config_path in model_configs.items():
            for seed in seeds:
                # Создаем временный конфиг с нужным сидом
                cfg = yaml.safe_load(open(base_config_path))
                cfg['seed'] = seed
                temp_config_path = f"configs/tmp/A_{args.source_domain}_{model_name}_seed{seed}.yaml"
                Path(temp_config_path).parent.mkdir(exist_ok=True)
                yaml.dump(cfg, open(temp_config_path, 'w'))

                # 1. Препроцессинг (выполнится только один раз)
                run_command(f"python ml_pipeline/scripts/preprocess.py --config {temp_config_path}")
                # 2. Обучение
                run_command(f"python ml_pipeline/scripts/train.py --config {temp_config_path}")
                # 3. Оценка
                # DEV: Находим последний созданный чекпоинт в MLflow для этого запуска
                run_name = f"{Path(temp_config_path).stem}_seed{seed}"
                # Это упрощенный поиск, в реальности может понадобиться API MLflow
                checkpoint = f"mlruns/1/{run_id}/artifacts/checkpoints/best_model.pt" # TODO: get run_id
                # run_command(f"python ml_pipeline/scripts/evaluate.py --config {temp_config_path} --checkpoint {checkpoint}")

    # --- Эксперимент B: Кросс-доменная оценка (Zero-Shot) ---
    elif args.experiment == "B_CrossDomain":
        print(f"\n--- Running Experiment B: Cross-Domain (Zero-Shot) from '{args.source_domain}' to '{args.target_domain}' ---")
        # TODO: Логика похожа, но evaluate.py запускается с конфигом от target_domain, а чекпоинтом от source_domain
        pass

    # --- Эксперимент C: Few-Shot дообучение ---
    elif args.experiment == "C_FewShot":
        print(f"\n--- Running Experiment C: Few-Shot Fine-tuning on '{args.target_domain}' ---")
        # TODO: 
        # 1. Создать few-shot сплиты (новая утилита или функция)
        # 2. Найти чекпоинты из Эксперимента А
        # 3. Запустить train.py с флагом --init_checkpoint
        # 4. Запустить evaluate.py
        pass

    print("\n--- Experiment suite finished ---")

if __name__ == "__main__":
    # DEV: Этот скрипт пока является высокоуровневым шаблоном.
    # Для полной реализации потребуется более тесная интеграция с MLflow API
    # для поиска нужных чекпоинтов. Но основная структура готова.
    main()