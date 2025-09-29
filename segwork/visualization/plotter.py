# src/visualization/plotter.py

"""
A helper class to manage and render a grid of image panels using Matplotlib.
"""
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Plotter:
    """Manages the creation of a grid plot for visual analysis."""

    def __init__(self, style_config: Dict[str, Any]):
        """
        Initializes the Plotter with styling parameters.
        
        Args:
            style_config: A dictionary with styling parameters like
                          'figsize_per_panel', 'dpi', 'title_fontsize'.
        """
        self.panels: List[Tuple[np.ndarray, str]] = []
        self.style = style_config

    def add_panel(self, image_data: np.ndarray, title: str):
        """
        Adds a new panel (an image and its title) to the plot queue.

        Args:
            image_data: The image to display (expected in BGR format from generators).
            title: The title for this subplot.
        """
        # DEV: Мы ожидаем BGR, так как это стандарт для OpenCV, но Matplotlib
        # ожидает RGB. Конвертация будет сделана в `render`.
        self.panels.append((image_data, title))

    def render(self, save_path: Path | None = None, show: bool = False):
        """
        Renders the final grid plot from all added panels.

        Args:
            save_path: If provided, the plot will be saved to this file path.
            show: If True, `plt.show()` will be called for interactive display.
        """
        if not self.panels:
            print("Warning: No panels to plot.")
            return

        num_panels = len(self.panels)
        
        # --- Automatically calculate the grid layout ---
        # DEV: Простая эвристика, чтобы сетка была более-менее квадратной.
        # Предпочитаем больше строк, чем колонок для лучшей читаемости.
        cols_config = self.style.get("cols")
        if cols_config:
            cols = int(cols_config)
        else:
            cols = math.ceil(math.sqrt(num_panels))
            # Limit max columns for better readability on standard screens
            if num_panels > 4:
                cols = min(4, cols)
        
        rows = math.ceil(num_panels / cols)

        # Calculate figure size based on the number of panels and desired size per panel
        figsize_per_panel = self.style.get("figsize_per_panel", (6, 6))
        fig_width = cols * figsize_per_panel[0]
        fig_height = rows * figsize_per_panel[1]

        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
        axes = axes.flatten()

        for i, (panel_data, title) in enumerate(self.panels):
            ax = axes[i]
            # Convert BGR (from OpenCV/generators) to RGB (for Matplotlib)
            ax.imshow(cv2.cvtColor(panel_data, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=self.style.get("title_fontsize", 12))
            ax.axis('off')

        # Hide any unused subplots in the grid
        for j in range(num_panels, len(axes)):
            axes[j].axis('off')

        fig.tight_layout(pad=1.5)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.style.get("dpi", 150), bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")

        if show:
            plt.show()
        
        # Close the figure to free up memory, crucial for running in loops or notebooks
        plt.close(fig)