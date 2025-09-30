# scripts/run_experiment.py

"""
Main orchestrator for running a complete experiment suite.

This script manages the entire lifecycle of an experiment defined by a template config:
1. Initializes an MLflow experiment and a progress tracking file.
2. Runs the one-time, heavy preprocessing step (`preprocess-tiles`).
3. Iterates through a list of seeds defined in the config. For each seed:
    a. Generates the specific data splits (`generate-splits`).
    b. Trains the model (`train`).
    c. Finds the run ID and best checkpoint from the training run.
    d. Evaluates the model (`evaluate`), logging results back to the original MLflow run.
4. It is resumable: it tracks completed steps in a `progress.json` file.
"""
import argparse
import subprocess
from pathlib import Path
import json
import mlflow
from typing import Tuple

from segwork.utils import load_config

def run_command(command: str):
    """Prints and runs a command in a subprocess, raising an error on failure."""
    print(f"\n{'='*20} RUNNING COMMAND {'='*20}")
    print(f">>> {command}")
    print(f"{'='*59}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[FATAL ERROR] Command failed with exit code {e.returncode}")
        raise e

def find_mlflow_run(mlflow_client, experiment_id, run_name) -> Tuple[str | None, str | None]:
    """
    Finds the latest MLflow run by name and returns its ID and best checkpoint path.
    """
    runs = mlflow_client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1
    )
    if not runs:
        print(f"Warning: Could not find MLflow run with name '{run_name}'")
        return None, None
    
    run_id = runs[0].info.run_id
    artifacts = mlflow_client.list_artifacts(run_id, "checkpoints")
    if not artifacts:
        print(f"Warning: No checkpoints found in run '{run_id}'")
        return run_id, None
    
    artifact_path = artifacts[0].path
    local_path = mlflow_client.download_artifacts(run_id, artifact_path)
    return run_id, local_path

def main():
    parser = argparse.ArgumentParser(description="Run a full experiment suite from a template config.")
    parser.add_argument("--config", required=True, help="Path to the experiment template config file.")
    args = parser.parse_args()

    # --- 1. Initialization ---
    cfg = load_config(args.config)

    mlflow.set_tracking_uri(f"file://{Path(cfg['logging']['artifact_uri']).resolve()}")

    experiment_name = cfg['logging']['experiment_name']
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Creating new MLflow experiment: '{experiment_name}'")
        mlflow.create_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)

    artifact_path = Path(experiment.artifact_location.replace("file://", ""))
    artifact_path.mkdir(parents=True, exist_ok=True)

    # --- НОВАЯ ЛОГИКА ИМЕНОВАНИЯ ФАЙЛА ПРОГРЕССА ---
    # Имя файла прогресса теперь уникально для каждого шаблона конфига.
    config_stem = Path(args.config).stem
    progress_filename = f"progress_{config_stem}.json"
    progress_file = artifact_path / progress_filename
    progress = {}

    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            print(f"Resuming experiment '{config_stem}', progress loaded from: {progress_file}")
        except json.JSONDecodeError:
            print(f"Warning: Could not read progress file at {progress_file}. Starting fresh for '{config_stem}'.")
                
    # --- 2. Heavy Preprocessing (Tiles) ---
    # DEV: Ключ для отслеживания `preprocess-tiles` теперь тоже уникален,
    # он зависит от `interim_data_root`, так как разные модели могут
    # требовать разные feature banks (и, следовательно, разные interim-папки).
    preprocess_key = f"preprocess_tiles_{Path(cfg['data']['interim_data_root']).name}"

    if not progress.get(preprocess_key, False):
        print("\n--- Running Step 1: Preprocessing Tiles ---")
        run_command(f"segwork-preprocess-tiles --config {args.config}")
        progress[preprocess_key] = True
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=4)
    else:
        print("\n--- Skipping Step 1: Preprocessing Tiles (already completed) ---")
  
    # --- 3. Loop Through Seeds ---
    seeds = cfg.get('run_seeds', [cfg.get('seed')])
    
    for seed in seeds:
        run_key = f"seed_{seed}"
        print(f"\n{'#'*20} PROCESSING SEED: {seed} {'#'*20}")
        
        if progress.get(run_key, {}).get("status") == "completed":
            print(f"Skipping seed {seed}: already marked as completed.")
            continue
            
        progress[run_key] = progress.get(run_key, {})

        # --- 3.1 Generate Splits ---
        if not progress[run_key].get("splits_generated"):
            progress[run_key]["status"] = "generating_splits"
            with open(progress_file, 'w') as f: json.dump(progress, f, indent=4)
            run_command(f"segwork-generate-splits --config {args.config} --seed {seed}")
            progress[run_key]["splits_generated"] = True
        
        # --- 3.2 Train Model ---
        if not progress[run_key].get("training_completed"):
            progress[run_key]["status"] = "training"
            with open(progress_file, 'w') as f: json.dump(progress, f, indent=4)
            run_command(f"segwork-train --config {args.config} --seed {seed}")
            progress[run_key]["training_completed"] = True
        
        # --- 3.3 Evaluate Model ---
        if not progress[run_key].get("evaluation_completed"):
            progress[run_key]["status"] = "evaluating"
            with open(progress_file, 'w') as f: json.dump(progress, f, indent=4)
            
            run_name = f"{Path(args.config).stem}_seed{seed}"
            # --- ИЗМЕНЕНИЕ ЗДЕСЬ: Находим и ID, и чекпоинт ---
            run_id, checkpoint_path = find_mlflow_run(mlflow.tracking.MlflowClient(), experiment.experiment_id, run_name)
            
            if checkpoint_path and run_id:
                # --- ИЗМЕНЕНИЕ ЗДЕСЬ: Передаем `mlflow_run_id` в `evaluate` ---
                eval_command = (
                    f"segwork-evaluate --config {args.config} --seed {seed} "
                    f"--checkpoint {checkpoint_path} --mlflow_run_id {run_id}"
                )
                run_command(eval_command)
                progress[run_key]["evaluation_completed"] = True
            else:
                error_msg = "Could not find checkpoint" if not checkpoint_path else "Could not find run_id"
                print(f"[ERROR] {error_msg} for run '{run_name}'. Skipping evaluation.")
                progress[run_key]["status"] = "error_evaluation_failed"
        
        # --- 3.4 Finalize ---
        if "error" not in progress.get(run_key, {}).get("status", ""):
            progress[run_key]["status"] = "completed"
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=4)
        print(f"--- Finished processing seed: {seed} ---")

    print("\n🎉 Experiment suite finished successfully! 🎉")

if __name__ == "__main__":
    main()