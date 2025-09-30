# scripts/run_experiment.py

"""
Main orchestrator for running a complete experiment suite.

This script manages the entire lifecycle of an experiment defined by a template config:
1. Initializes an MLflow experiment and a progress tracking file.
2. Runs the one-time, heavy preprocessing step (`preprocess-tiles`).
3. Iterates through a list of seeds defined in the config. For each seed:
    a. Generates the specific data splits (`generate-splits`).
    b. Trains the model (`train`).
    c. Finds the best checkpoint from the training run.
    d. Evaluates the model on the test set (`evaluate`).
4. It is resumable: it tracks completed steps in a `progress.json` file and
   skips them on subsequent runs.
"""
import argparse
import subprocess
from pathlib import Path
import json
import mlflow

# DEV: Используем наш кастомный `load_config` из `segwork`
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

def find_best_checkpoint(mlflow_client, experiment_id, run_name) -> str | None:
    """Finds the path to the best checkpoint artifact for a given MLflow run."""
    runs = mlflow_client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1
    )
    if not runs:
        print(f"Warning: Could not find MLflow run with name '{run_name}'")
        return None
    
    run_id = runs[0].info.run_id
    artifacts = mlflow_client.list_artifacts(run_id, "checkpoints")
    if not artifacts:
        print(f"Warning: No checkpoints found in run '{run_id}'")
        return None
    
    artifact_path = artifacts[0].path
    # DEV: Скачиваем артефакт во временную директорию. MLflow сам управляет путем.
    local_path = mlflow_client.download_artifacts(run_id, artifact_path)
    return local_path

def main():
    parser = argparse.ArgumentParser(description="Run a full experiment suite from a template config.")
    parser.add_argument("--config", required=True, help="Path to the experiment template config file.")
    args = parser.parse_args()

    # --- 1. Initialization ---
    cfg = load_config(args.config)
    
    mlflow.set_tracking_uri(f"file://{Path(cfg['logging']['artifact_uri']).resolve()}")
    
    # Ensure the experiment exists in MLflow to get its artifact location
    experiment_name = cfg['logging']['experiment_name']
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Creating new MLflow experiment: '{experiment_name}'")
        mlflow.create_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
    
    # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
    # `experiment.artifact_location` возвращает URI вида 'file:///path/to/artifacts'
    # Мы преобразуем его в объект Path, который корректно работает с файловой системой.
    artifact_path = Path(experiment.artifact_location.replace("file://", ""))
    
    # Явно создаем директорию артефактов эксперимента, если она еще не существует.
    # Это решает проблему `FileNotFoundError`, если MLflow не успел создать папку.
    artifact_path.mkdir(parents=True, exist_ok=True)
    
    progress_file = artifact_path / "progress.json"
    progress = {}
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            print(f"Resuming experiment, progress loaded from: {progress_file}")
        except json.JSONDecodeError:
            print(f"Warning: Could not read progress file at {progress_file}. Starting fresh.")
    
    # --- 2. Heavy Preprocessing (Tiles) ---
    if not progress.get("preprocess_tiles_completed", False):
        print("\n--- Running Step 1: Preprocessing Tiles ---")
        run_command(f"segwork-preprocess-tiles --config {args.config}")
        progress["preprocess_tiles_completed"] = True
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
            checkpoint_path = find_best_checkpoint(mlflow.tracking.MlflowClient(), experiment.experiment_id, run_name)
            
            if checkpoint_path:
                run_command(f"segwork-evaluate --config {args.config} --seed {seed} --checkpoint {checkpoint_path}")
                progress[run_key]["evaluation_completed"] = True
            else:
                print(f"[ERROR] Could not find checkpoint for run '{run_name}'. Skipping evaluation.")
                progress[run_key]["status"] = "error_no_checkpoint"
        
        # --- 3.4 Finalize ---
        if progress.get(run_key, {}).get("status") != "error_no_checkpoint":
            progress[run_key]["status"] = "completed"
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=4)
        print(f"--- Finished processing seed: {seed} ---")

    print("\n🎉 Experiment suite finished successfully! 🎉")

if __name__ == "__main__":
    main()