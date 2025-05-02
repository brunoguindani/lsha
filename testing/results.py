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

def get_counts(df: pd.DataFrame):
  return df.groupby('seed').size()

def summarize(df: pd.DataFrame, print_prefix: str = ""):
  sizes = get_counts(df)
  mean = sizes.mean()
  std = sizes.std()
  print(print_prefix, mean, "±", round(std, 2))


def analyze_pairwise_stats(df_a: pd.DataFrame, df_b: pd.DataFrame,
      name_a: str, name_b: str, query_name: str) -> tuple[str, dict[float]]:
  key = f"{name_a} vs {name_b}"
  result = {}

  # Mann-Whitney U test
  mw_p = mannwhitneyu(df_a, df_b, alternative='two-sided').pvalue

  # Vargha-Delaney A12 effect size
  n1, n2 = len(df_a), len(df_b)
  rank_sum = sum(x > y for x in df_a for y in df_b)
  tie_sum = sum(x == y for x in df_a for y in df_b)
  a12 = (rank_sum + 0.5 * tie_sum) / (n1 * n2)

  if 0.44 <= a12 <= 0.56:
    vd = "neglig."
  elif 0.56 < a12 <= 0.64 or 0.36 <= a12 < 0.44:
    vd = "small"
  elif 0.64 < a12 <= 0.71 or 0.29 <= a12 < 0.36:
    vd = "medium"
  else:
    vd = "large"

  result[f"MW_{query_name}"] = mw_p
  result[f"VD_{query_name}"] = vd
  return key, result


def compare_given_threshold(input_df: pd.DataFrame,
                            threshold_probs: dict[str: float]):
  df = input_df.copy()
  stats_def = {}
  stats_real = {}
  df[all_transition_cols] = df[all_transition_cols].astype(int)

  for query in query_cols:
    threshold = threshold_probs[query]
    prob_to_bool = lambda p: int(p >= threshold)
    df[query] = df[query].map(prob_to_bool)

    defect_counts = {}
    realistic_counts = {}
    technique_labels = []

    for tec, group in df.groupby('technique'):
      group_def = group.loc[group[query] == 1]
      group_def_real = filter(group_def, is_realistic)

      defect_counts[tec] = get_counts(group_def).values
      realistic_counts[tec] = get_counts(group_def_real).values
      technique_labels.append(tec)

    # Plotting
    n = len(technique_labels)
    gap = 1
    defect_positions = list(range(n))
    realistic_positions = list(range(n + gap, 2 * n + gap))

    fig, ax = plt.subplots(figsize=(8, 6))

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(n)]
    box_width = 0.6

    for i, tec in enumerate(technique_labels):
      def_data = defect_counts[tec]
      real_data = realistic_counts[tec]
      color = colors[i]

      bp_def = ax.boxplot(def_data, positions=[defect_positions[i]],
                          widths=box_width, patch_artist=True,
                          medianprops=dict(color='red'))
      for patch in bp_def['boxes']:
        patch.set_facecolor(color)

      bp_real = ax.boxplot(real_data, positions=[realistic_positions[i]],
                           widths=box_width, patch_artist=True,
                           medianprops=dict(color='red'))
      for patch in bp_real['boxes']:
        patch.set_facecolor(color)

    middle_defect = sum(defect_positions) / len(defect_positions)
    middle_realistic = sum(realistic_positions) / len(realistic_positions)
    ax.set_xticks([middle_defect, middle_realistic])
    ax.set_xticklabels(["Defects", "Realistic defects"], fontsize=11)

    ax.set_title(f"{query} - Threshold: {threshold}")
    ax.set_ylabel("Number of mutants")
    ax.tick_params(axis='x')

    all_data = list(defect_counts.values()) + list(realistic_counts.values())
    ax.set_yticks(range(0, 505, 50))
    ax.yaxis.grid(True)
    ax.set_ylim(0, 505)

    handles = [plt.Line2D([0], [0], color=color, lw=6) for color in colors]
    ax.legend(handles, technique_labels, title="Techniques", loc="upper right")

    plt.tight_layout()
    file = os.path.join('plots', f'{query}_{threshold:.2f}.svg')
    plt.savefig(file)
    plt.close()

    # Statistical comparisons
    for t1, t2 in itertools.combinations(technique_labels, 2):
      key_def, res_def = analyze_pairwise_stats(defect_counts[t1],
                           defect_counts[t2], str(t1), str(t2), query)
      key_real, res_real = analyze_pairwise_stats(realistic_counts[t1],
                           realistic_counts[t2], str(t1), str(t2), query)

      if key_def not in stats_def:
        stats_def[key_def] = {}
      if key_real not in stats_real:
        stats_real[key_real] = {}

      stats_def[key_def].update(res_def)
      stats_real[key_real].update(res_real)

  pd.set_option("display.max_columns", None)

  print("Tests for defects:")
  result_def = pd.DataFrame.from_dict(stats_def, orient='index').round(4)
  print(result_def)

  print("Tests for realistic defects:")
  result_real = pd.DataFrame.from_dict(stats_real, orient='index').round(4)
  print(result_real)

  full_table = result_def.to_latex() + '\n' + result_real.to_latex()
  table_name = '_'.join([str(v) for v in threshold_probs.values()])
  with open(os.path.join('plots', f'{table_name}.txt'), 'w') as f:
    f.write(full_table)

if __name__ == '__main__':
  df = pd.read_csv('testing.csv')
  # queries = 3
  # fig, axes = plt.subplots(1, queries)
  # for i in range(queries):
  #   col = f'query{i}'
  #   ax = axes[i]
  #   ax.hist(df[col])
  #   ax.set_title(col)
  # plt.show()
  thresholds = {'query0': 0.9, 'query1': 0.5, 'query2': 0.5}
  compare_given_threshold(df, thresholds)
