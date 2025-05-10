from collections.abc import Callable
import itertools
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.stats import mannwhitneyu

from fuzzing import MutationFuzzer

query_idxs = MutationFuzzer.QUERY_IDXS
all_transitions = MutationFuzzer.REMOVABLE_TRANS_IDS
necessary_transitions = [15, 19, 23]

query_cols = [f'query{q}' for q in query_idxs]
all_transition_cols = [f'l{t}' for t in all_transitions]
nec_transition_cols = [f'l{t}' for t in necessary_transitions]

def is_realistic(row: pd.Series) -> bool:
  return all(row[tr] == 0 for tr in nec_transition_cols)

def filter(df: pd.DataFrame, func: Callable) -> pd.DataFrame:
  return df.loc[df.apply(func, axis=1)]

def get_fraction(df: pd.DataFrame, total_counts: pd.Series):
  counts = df.groupby('seed').size()
  return (counts / total_counts).fillna(0)

def summarize(df: pd.DataFrame, print_prefix: str = ""):
  sizes = df.groupby('seed').size()
  mean = sizes.mean()
  std = sizes.std()
  print(print_prefix, mean, "±", round(std, 2))

def analyze_pairwise_stats(df_a: pd.Series, df_b: pd.Series,
      name_a: str, name_b: str, query_name: str) -> tuple[str, dict]:
  key = f"{name_a} vs {name_b}"
  result = {}

  mw_p = mannwhitneyu(df_a, df_b, alternative='two-sided').pvalue

  n1, n2 = len(df_a), len(df_b)
  rank_sum = sum(x > y for x in df_a for y in df_b)
  tie_sum = sum(x == y for x in df_a for y in df_b)
  a12 = (rank_sum + 0.5 * tie_sum) / (n1 * n2)

  if 0.44 <= a12 <= 0.56:
    vd = "N"
  elif 0.56 < a12 <= 0.64 or 0.36 <= a12 < 0.44:
    vd = "S"
  elif 0.64 < a12 <= 0.71 or 0.29 <= a12 < 0.36:
    vd = "M"
  else:
    vd = "L"

  result[f"MW {query_name}"] = mw_p
  result[f"VD {query_name}"] = vd
  return key, result

def compare_given_threshold(input_df: pd.DataFrame,
                            threshold_probs: dict[str, float]):
  df = input_df.copy()
  stats_def = {}
  stats_real = {}
  df[all_transition_cols] = df[all_transition_cols].astype(int)

  for i, query in enumerate(query_cols):
    threshold = threshold_probs[query]
    prob_to_bool = lambda p: int(p >= threshold)
    df[query] = df[query].map(prob_to_bool)

    defect_fractions = {}
    realistic_fractions = {}
    technique_labels = []

    for tec, group in df.groupby('technique'):
      total_per_seed = group.groupby('seed').size()

      group_def = group.loc[group[query] == 1]
      group_def_real = filter(group_def, is_realistic)

      defect_fractions[tec] = get_fraction(group_def, total_per_seed).values
      realistic_fractions[tec] = get_fraction(group_def_real, total_per_seed).values
      technique_labels.append(tec)

    # Plotting
    n = len(technique_labels)
    gap = 0
    defect_positions = list(range(n))
    realistic_positions = list(range(n + gap, 2 * n + gap))

    fig, ax = plt.subplots(figsize=(4, 4))

    cmap = plt.get_cmap('tab10')
    colors = [cmap(_ % 10) for _ in range(n)]
    box_width = 0.5

    for j, tec in enumerate(technique_labels):
      def_data = defect_fractions[tec]
      real_data = realistic_fractions[tec]
      color = colors[j]

      bp_def = ax.boxplot(def_data, positions=[defect_positions[j]],
                          widths=box_width, patch_artist=True,
                          medianprops=dict(color='red', lw=0.3))
      for patch in bp_def['boxes']:
        patch.set_facecolor(color)

      bp_real = ax.boxplot(real_data, positions=[realistic_positions[j]],
                           widths=box_width, patch_artist=True,
                           medianprops=dict(color='red', lw=0.3))
      for patch in bp_real['boxes']:
        patch.set_facecolor(color)

    middle_defect = sum(defect_positions) / len(defect_positions)
    middle_realistic = sum(realistic_positions) / len(realistic_positions)
    ax.set_xticks([middle_defect, middle_realistic])
    ax.set_xticklabels(["Total failures", "Realistic failures"], fontsize=11)

    ax.set_ylabel("Fraction of individuals")
    ax.tick_params(axis='x')

    ax.set_yticks([i / 10.0 for i in range(0, 11)])
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True)

    handles = [plt.Line2D([0], [0], color=color, lw=6) for color in colors]
    ax.legend(handles, technique_labels, title="Techniques", loc="upper right")

    plt.tight_layout()
    file = os.path.join('plots', f'testing_{query}_{threshold:.2f}.pdf')
    plt.savefig(file)
    plt.close()

    # Statistical comparisons
    for t1, t2 in itertools.combinations(technique_labels, 2):
      key_def, res_def = analyze_pairwise_stats(defect_fractions[t1],
                           defect_fractions[t2], str(t1), str(t2), query)
      key_real, res_real = analyze_pairwise_stats(realistic_fractions[t1],
                           realistic_fractions[t2], str(t1), str(t2), query)

      stats_def.setdefault(key_def, {}).update(res_def)
      stats_real.setdefault(key_real, {}).update(res_real)

  pd.set_option("display.max_columns", None)
  to_exp = lambda x: f'{x:.2e}' if isinstance(x, float) else x

  print("Tests for failures:")
  result_def = pd.DataFrame.from_dict(stats_def, orient='index').map(to_exp)
  print(result_def)

  print("Tests for realistic failures:")
  result_real = pd.DataFrame.from_dict(stats_real, orient='index').map(to_exp)
  print(result_real)

  full_table = result_def.to_latex() + '\n' + \
               result_real.to_latex(label='tab:testing')
  table_name = '_'.join([str(v) for v in threshold_probs.values()])
  with open(os.path.join('plots', f'{table_name}.txt'), 'w') as f:
    f.write(full_table)


def plot_times(df: pd.DataFrame):
  fig, ax = plt.subplots(figsize=(4, 4))
  groups = []
  labels = []
  for tec, group in df.groupby('technique'):
    groups.append(group['time'])
    labels.append(tec)
  box = ax.boxplot(groups, tick_labels=labels, widths=0.4,
                    medianprops=dict(color='red', lw=0.3))
  # ax.set_xticklabels(labels, fontsize=12)
  ax.tick_params(axis='y', labelsize=11)
  ax.grid(axis='y', alpha=0.45)
  ax.set_ylabel('execution time [s]')
  plt.tight_layout()
  # Save plots
  os.makedirs('plots', exist_ok=True)
  output_file = os.path.join('plots', 'testing_times.pdf')
  fig.savefig(output_file)
  plt.close(fig)


if __name__ == '__main__':
  df = pd.read_csv('testing.csv')
  times = pd.read_csv('times.csv')

  df = df.replace('genetic', 'search-based')
  times = times.replace('genetic', 'search-based')

  queries = 3
  # fig, axes = plt.subplots(1, queries)
  # for i in range(queries):
  #   col = f'query{i}'
  #   ax = axes[i]
  #   ax.hist(df[col])
  #   ax.set_title(col)
  # plt.show()
  thresholds = {'query0': 0.9, 'query1': 0.5, 'query2': 0.5}
  compare_given_threshold(df, thresholds)
  plot_times(times)
