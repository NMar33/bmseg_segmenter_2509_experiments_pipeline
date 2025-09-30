# scripts/generate_splits.py

"""
Script for the second stage of preprocessing: Split Generation.

This script reads the `master_split.csv` file created by `preprocess_tiles.py`
and generates the final `train.txt`, `val.txt`, and `test.txt` files based on
the splitting strategy defined in the config and a specific random seed.

It creates a dedicated folder for each seed's splits inside `interim_data/splits/`.
This is a fast, lightweight operation that can be run multiple times for
different seeds without re-calculating tiles.
"""
import argparse
import random
from pathlib import Path
from typing import Dict, Set
import pandas as pd

from segwork.utils import load_config

def resolve_data_splits(
    df_master: pd.DataFrame,
    split_gen_config: dict,
    seed: int
) -> Dict[str, Set[str]]:
    """
    Resolves train, validation, and test sets of SOURCE IMAGE IDs based on the config.
    """
    # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
    # `split_gen_config` - это вся секция `split_generation`. Нам нужно
    # сначала обратиться к `source_splits` внутри нее.
    source_splits_config = split_gen_config['source_splits']
    
    source_ids: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}

    # --- Step 1: Populate sets from explicitly defined folders ---
    print("Step 1: Populating splits from explicit folder definitions...")
    for split_name in ["train", "val", "test"]:
        cfg_val = source_splits_config.get(split_name)
        if isinstance(cfg_val, list):
            unique_ids = df_master[df_master['source_split'].isin(cfg_val)]['source_image_id'].unique()
            source_ids[split_name].update(unique_ids)
            print(f"  - Found {len(unique_ids)} source images for '{split_name}' from folders: {cfg_val}")

    # --- Step 2: Check for overlaps from explicit folder definitions ---
    if source_ids['train'].intersection(source_ids['val']):
        raise ValueError("Config Error: `train` and `val` source folders overlap.")
    if source_ids['train'].intersection(source_ids['test']):
        raise ValueError("Config Error: `train` and `test` source folders overlap.")
    if source_ids['val'].intersection(source_ids['test']):
        raise ValueError("Config Error: `val` and `test` source folders overlap.")

    # --- Step 3: Carve out validation set if defined as a fraction ---
    val_cfg = source_splits_config.get('val')
    if isinstance(val_cfg, float):
        print("Step 3: Carving out validation set from train set...")
        if source_ids['val']:
            raise ValueError("Config Error: Cannot specify both folder list and fraction for `val` split.")
        
        val_frac = val_cfg
        train_list = sorted(list(source_ids['train']))
        rng = random.Random(seed)
        rng.shuffle(train_list)
        
        val_size = int(len(train_list) * val_frac)
        if val_size == 0 and len(train_list) > 0:
            raise ValueError(f"val_frac {val_frac} is too small, results in 0 validation samples.")
        
        val_ids_to_move = set(train_list[:val_size])
        source_ids['val'].update(val_ids_to_move)
        source_ids['train'].difference_update(val_ids_to_move)
        print(f"  - Moved {len(source_ids['val'])} source images to validation set.")

    # --- Step 4: Carve out test set if defined as a fraction ---
    test_cfg = source_splits_config.get('test')
    if isinstance(test_cfg, float):
        print("Step 4: Carving out test set...")
        if source_ids['test']:
            raise ValueError("Config Error: Cannot specify both folder list and fraction for `test` split.")
        
        test_frac = test_cfg
        
        if not source_ids['val']:
            raise ValueError("Config Error: Cannot create 'test' set from a fraction because 'val' set is not defined. Please define 'val' using a folder or a fraction.")
        
        source_list = sorted(list(source_ids['val']))
        rng = random.Random(seed)
        rng.shuffle(source_list)

        test_size = int(len(source_list) * test_frac)
        if test_size == 0 and len(source_list) > 0:
            raise ValueError(f"test_frac {test_frac} is too small, results in 0 test samples.")
            
        test_ids_to_move = set(source_list[:test_size])
        source_ids['test'].update(test_ids_to_move)
        source_ids['val'].difference_update(test_ids_to_move)
        print(f"  - Moved {len(source_ids['test'])} source images from validation to test set.")

    return source_ids

def main():
    parser = argparse.ArgumentParser(description="Step 2: Generate data splits for a specific seed.")
    parser.add_argument("--config", required=True, help="Path to the experiment config file.")
    parser.add_argument("--seed", required=True, type=int, help="The random seed for this specific split.")
    args = parser.parse_args()

    print(f"--- Starting Step 2: Generating splits for seed {args.seed} ---")
    cfg = load_config(args.config)
    data_cfg = cfg['data']
    
    interim_root = Path(data_cfg['interim_data_root'])
    master_path = interim_root / "master_split.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"master_split.csv not found in {interim_root}. Run preprocess_tiles.py first.")
    
    df_master = pd.read_csv(master_path)
    
    # Resolve the sets of source image IDs for train, val, and test
    source_id_sets = resolve_data_splits(df_master, data_cfg['split_generation'], args.seed)
    
    output_dir = interim_root / "splits" / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, ids_set in source_id_sets.items():
        if not ids_set:
            if split_name == 'train':
                raise ValueError("Train set is empty after splitting. This should not happen. Check your config.")
            print(f"Warning: {split_name} set is empty for seed {args.seed}.")
            (output_dir / f"{split_name}.txt").write_text("")
            continue
            
        tile_ids = df_master[df_master['source_image_id'].isin(ids_set)]['tile_id'].tolist()
        
        print(f"Final Split '{split_name}': {len(ids_set)} source images -> {len(tile_ids)} tiles.")
        (output_dir / f"{split_name}.txt").write_text("\n".join(sorted(tile_ids)))

    print(f"\n--- Splits for seed {args.seed} generated successfully! ---")
    print(f"Split files are located in: {output_dir}")


if __name__ == "__main__":
    main()