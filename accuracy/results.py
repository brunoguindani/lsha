import itertools
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.pardir)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, 'testing')))
from testing.results import analyze_pairwise_stats

def plot_and_test_results(num_data: int):
  columns = ['rel_lsha', 'rel_dual', 'rel_xgb']
  labels = ['$L^*_{sha}$', 'Neural Network', 'XGBoost']
  colors = ['lightblue', 'lightgreen', 'lightcoral']

  # Read data
  file_path = 'results.xlsx'
  sheet_name = f'accuracy_{num_data}'
  df = pd.read_excel(file_path, sheet_name=sheet_name)
  df_plot = df[columns].iloc[0:num_data]
  df_plot = df_plot.apply(pd.to_numeric, errors='coerce')
  df_plot = df_plot.dropna()

  # Plot data
  fig, ax = plt.subplots(figsize=(6, 5))
  box = ax.boxplot([df_plot[col] for col in columns], patch_artist=True,
                    widths=0.4, medianprops=dict(color='red'))
  for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
  ax.set_xticklabels(labels, fontsize=12)
  ax.tick_params(axis='y', labelsize=11)
  ax.grid(axis='y')
  plt.tight_layout()
  # Save plots
  os.makedirs('plots', exist_ok=True)
  output_file = os.path.join('plots', sheet_name + '.svg')
  fig.savefig(output_file)
  plt.close(fig)

  # Test data
  tests_idx = []
  tests_rows = []
  for i, j in itertools.combinations(range(len(columns)), 2):
    key, row = analyze_pairwise_stats(df_plot[columns[i]], df_plot[columns[j]],
                           labels[i], labels[j], 'accuracy')
    tests_idx.append(key)
    tests_rows.append(row)

  df_tests = pd.DataFrame(tests_rows, index=tests_idx)
  print(df_tests)
  with open(os.path.join('plots', f'{sheet_name}.txt'), 'w') as f:
    f.write(df_tests.to_latex())


if __name__ == '__main__':
  for num in (20, 30):
    print(num)
    plot_and_test_results(num)
