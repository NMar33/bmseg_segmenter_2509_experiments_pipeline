# segwork/models/model_builder.py

"""
Factory function for creating segmentation models based on the configuration.
"""
from typing import Dict, Any
import torch.nn as nn
from segmentation_models_pytorch import Unet

from segwork.models.adapter import ChannelAdapter

def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    Builds the segmentation model (M1, M2, or M3) based on the config.

    Args:
        config: The full experiment configuration dictionary.

    Returns:
        An initialized PyTorch model (nn.Module).
    """
    model_cfg = config['model']
    data_cfg = config['data']
    
    use_adapter = model_cfg.get('adapter', {}).get('use', False)
    use_feature_bank = data_cfg.get('feature_bank', {}).get('use', False)
    
    # Определяем количество входных каналов
    if use_feature_bank:
        in_channels = len(data_cfg['feature_bank']['channels'])
    else:
        in_channels = 1
        
    print(f"Building model with {in_channels} input channels.")

    # --- Сборка модели ---
    if use_adapter:
        # M2 or M3: N channels -> Adapter -> Pretrained Encoder
        # DEV: Это наша ключевая архитектура для M2 и M3.
        # Адаптер преобразует наши N каналов в 3, а затем они подаются
        # в стандартный U-Net с предобученным энкодером.
        adapter_out_channels = model_cfg['adapter']['out_channels']
        
        adapter = ChannelAdapter(
            c_in=in_channels,
            c_out=adapter_out_channels,
            init=model_cfg['adapter']['init']
        )
        
        net = Unet(
            encoder_name=model_cfg['encoder'],
            encoder_weights=model_cfg['encoder_weights'],
            in_channels=adapter_out_channels, # U-Net видит 3 канала после адаптера
            classes=model_cfg['classes']
        )
        
        model = nn.Sequential(adapter, net)
        print("Model built: Feature Bank -> ChannelAdapter -> Pretrained U-Net (M2/M3).")
    else:
        # M1: 1 channel -> Encoder from scratch
        # DEV: Это наш baseline. Адаптер не используется, энкодер обучается с нуля.
        if model_cfg['encoder_weights'] is not None:
            print(f"Warning: `encoder_weights` is set to '{model_cfg['encoder_weights']}' "
                  "but adapter is disabled. Forcing `encoder_weights=None` for 1-channel input.")
            model_cfg['encoder_weights'] = None

        model = Unet(
            encoder_name=model_cfg['encoder'],
            encoder_weights=None,
            in_channels=in_channels, # Должно быть 1
            classes=model_cfg['classes']
        )
        print("Model built: Raw Input -> U-Net from scratch (M1).")
        
    return model