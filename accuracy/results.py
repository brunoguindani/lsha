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
  # colors = ['lightblue', 'lightgreen', 'lightcoral']

  # Read data
  file_path = 'results.xlsx'
  sheet_name = 'regression'
  df = pd.read_excel(file_path, sheet_name=sheet_name)
  df_plot = df[columns].iloc[0:num_data]
  df_plot = df_plot.apply(pd.to_numeric, errors='coerce')
  df_plot = df_plot.dropna()

  # Plot data
  fig, ax = plt.subplots(figsize=(4, 4))
  box = ax.boxplot([df_plot[col] for col in columns], # patch_artist=True,
                    widths=0.4, medianprops=dict(color='red'))
  for col in columns:
    print(col, "median =", df_plot[col].median())
  # for patch, color in zip(box['boxes'], colors):
  #   patch.set_facecolor(color)
  ax.set_xticklabels(labels, fontsize=12)
  ax.tick_params(axis='y', labelsize=11)
  ax.grid(axis='y', alpha=0.45)
  plt.tight_layout()
  # Save plots
  os.makedirs('plots', exist_ok=True)
  output_file = os.path.join('plots', f'{sheet_name}_{num_data}.pdf')
  fig.savefig(output_file)
  plt.close(fig)

  # Test data
  tests_idx = []
  tests_rows = []
  for i, j in itertools.combinations(range(len(columns)), 2):
    key, row = analyze_pairwise_stats(df_plot[columns[i]], df_plot[columns[j]],
                           labels[i], labels[j], 'regression')
    tests_idx.append(key)
    tests_rows.append(row)

  to_exp = lambda x: f'{x:.2e}' if isinstance(x, float) else x
  df_tests = pd.DataFrame(tests_rows, index=tests_idx).map(to_exp)
  df_tests.columns = ['\\mw{}', '\\vd{}']
  print(df_tests)
  with open(os.path.join('plots', f'{sheet_name}_{num_data}.txt'), 'w') as f:
    f.write(df_tests.to_latex(label='tab:regression'))


if __name__ == '__main__':
  plot_and_test_results(20)
