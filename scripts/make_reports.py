# scripts/make_reports.py

"""
Script to collect evaluation results and generate summary tables for the paper.
"""
import argparse
from pathlib import Path
import json
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Generate result tables from evaluation JSONs.")
    parser.add_argument("--results_dir", default="./reports", help="Directory containing evaluation JSON files.")
    parser.add_argument("--experiment_prefix", required=True, help="Prefix to filter JSON files (e.g., 'A_InDomain_isbi').")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # --- 1. Сбор и загрузка данных ---
    all_results = []
    json_files = sorted(results_dir.glob(f"{args.experiment_prefix}*.json"))
    
    if not json_files:
        print(f"Warning: No JSON files found with prefix '{args.experiment_prefix}' in '{results_dir}'")
        return

    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Извлекаем имя модели и сид из имени файла для группировки
            # Пример имени файла: A_InDomain_isbi_M1_Baseline_seed42_test_results.json
            parts = file_path.stem.split('_')
            data['model_name'] = "_".join(parts[3:-3]) # "M1_Baseline"
            data['seed'] = int(parts[-3].replace('seed', ''))
            all_results.append(data)
    
    df = pd.DataFrame(all_results)
    
    # --- 2. Агрегация и форматирование ---
    # DEV: Мы группируем по имени модели и считаем среднее и стандартное отклонение
    # по всем сидам. Это дает нам статистически значимые результаты.
    
    # Выбираем только колонки с метриками
    metric_cols = [col for col in df.columns if col.endswith('_mean')]
    
    summary = df.groupby('model_name')[metric_cols].agg(['mean', 'std'])
    
    # Форматируем в красивую строку "mean ± std"
    for col in metric_cols:
        summary[col, 'formatted'] = summary.apply(
            lambda row: f"{row[col, 'mean']:.4f} ± {row[col, 'std']:.4f}",
            axis=1
        )
    
    # --- 3. Вывод таблицы в формате Markdown ---
    # DEV: Markdown - идеальный формат, его легко скопировать куда угодно.
    
    # Выбираем только отформатированные колонки для вывода
    display_cols = [(col, 'formatted') for col in metric_cols]
    final_table = summary[display_cols]
    final_table.columns = [col[0].replace('_mean', '') for col in display_cols] # Убираем суффиксы

    print(f"\n--- Summary Table for '{args.experiment_prefix}' ---")
    print(final_table.to_markdown(index=True))


if __name__ == "__main__":
    main()