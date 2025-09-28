# src/utils.py

"""
Common utility functions for the ML pipeline.
"""
import random
import copy
import yaml
import numpy as np
import torch

def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # DEV: Эти флаги важны для полной воспроизводимости на GPU,
        # но могут немного замедлить обучение. Для исследований это оправдано.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def flatten_config(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """
    Flattens a nested dictionary for logging.
    E.g., {'data': {'tile_size': 512}} -> {'data.tile_size': 512}
    """
    # DEV: MLflow отлично работает с плоскими словарями параметров.
    # Эта функция делает логирование конфигов чистым и удобным.
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def load_config(config_path: str) -> dict:
    """
    Loads a YAML configuration, inheriting from a base configuration file.
    It first loads `configs/base.yaml`, then merges the specific
    experiment config on top of it.
    """
    # DEV: Это сердце нашей модульной системы конфигов.
    # Рекурсивное обновление словарей позволяет переопределять
    # только нужные параметры, не копируя весь base.yaml.
    
    # Путь к базовому конфигу всегда относителен текущего файла
    base_path = Path(__file__).parent.parent / "configs/base.yaml"
    with open(base_path, 'r') as f:
        base_config = yaml.safe_load(f)

    with open(config_path, 'r') as f:
        exp_config = yaml.safe_load(f)

    def deep_update(base_dict, update_dict):
        """Recursively update a dictionary."""
        for k, v in update_dict.items():
            if isinstance(v, dict) and k in base_dict and isinstance(base_dict[k], dict):
                base_dict[k] = deep_update(base_dict[k], v)
            else:
                base_dict[k] = v
        return base_dict

    # Создаем копию, чтобы не изменять исходный базовый словарь в памяти
    config = copy.deepcopy(base_config)
    return deep_update(config, exp_config)