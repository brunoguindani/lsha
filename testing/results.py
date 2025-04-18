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

def vargha_delaney(a, b):
  n1, n2 = len(a), len(b)
  rank_sum = sum(x > y for x in a for y in b)
  tie_sum = sum(x == y for x in a for y in b)
  a12 = (rank_sum + 0.5 * tie_sum) / (n1 * n2)

  # Interpretation based on thresholds
  if 0.44 <= a12 <= 0.56:
    return "neglig."
  elif 0.56 < a12 <= 0.64 or 0.36 <= a12 < 0.44:
    return "small"
  elif 0.64 < a12 <= 0.71 or 0.29 <= a12 < 0.36:
    return "medium"
  else:
    return "large"


def compare_given_threshold(input_df: pd.DataFrame, threshold_prob: float):
  prob_to_bool = lambda p: int(p >= threshold_prob)
  df = input_df.copy()
  df[query_cols] = df[query_cols].map(prob_to_bool)
  df[all_transition_cols] = df[all_transition_cols].astype(int)

  stats_def = {}
  stats_real = {}

  for query in query_cols:
    print("Requirement:", query)

    defect_counts = {}
    realistic_counts = {}
    technique_labels = []

    for tec, group in df.groupby('technique'):
      print("Technique", tec)
      group_def = group.loc[group[query] == 1]
      group_def_real = filter(group_def, is_realistic)

      summarize(group_def, " Defects")
      summarize(group_def_real, " of which realistic")

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
                          widths=box_width, patch_artist=True)
      for patch in bp_def['boxes']:
        patch.set_facecolor(color)

      bp_real = ax.boxplot(real_data, positions=[realistic_positions[i]],
                           widths=box_width, patch_artist=True)
      for patch in bp_real['boxes']:
        patch.set_facecolor(color)

    middle_defect = sum(defect_positions) / len(defect_positions)
    middle_realistic = sum(realistic_positions) / len(realistic_positions)
    ax.set_xticks([middle_defect, middle_realistic])
    ax.set_xticklabels(["Defects", "Realistic defects"], fontsize=11)

    ax.set_title(f"{query} - Threshold: {threshold_prob}")
    ax.set_ylabel("Number of mutants")
    ax.tick_params(axis='x')

    all_data = list(defect_counts.values()) + list(realistic_counts.values())
    max_y = max([max(lst) if len(lst) > 0 else 0 for lst in all_data] + [50])
    ax.set_yticks(range(0, max_y + 51, 50))
    ax.yaxis.grid(True)
    ax.set_ylim(0, 505)

    handles = [plt.Line2D([0], [0], color=color, lw=6) for color in colors]
    ax.legend(handles, technique_labels, title="Techniques", loc="upper right")

    plt.tight_layout()
    file = os.path.join('plots', f'testing_{threshold_prob}_{query}.svg')
    plt.savefig(file)
    plt.close()

    # Statistical comparisons
    for t1, t2 in itertools.combinations(technique_labels, 2):
      key = f"{t1} vs {t2}"

      if key not in stats_def:
        stats_def[key] = {}
      if key not in stats_real:
        stats_real[key] = {}

      # Mann-Whitney U
      mw_def = mannwhitneyu(defect_counts[t1], defect_counts[t2],
                            alternative='two-sided').pvalue
      mw_real = mannwhitneyu(realistic_counts[t1], realistic_counts[t2],
                             alternative='two-sided').pvalue

      # Vargha-Delaney interpretation
      vd_def = vargha_delaney(defect_counts[t1], defect_counts[t2])
      vd_real = vargha_delaney(realistic_counts[t1], realistic_counts[t2])

      stats_def[key][f"MW_{query}"] = mw_def
      stats_def[key][f"VD_{query}"] = vd_def

      stats_real[key][f"MW_{query}"] = mw_real
      stats_real[key][f"VD_{query}"] = vd_real

  pd.set_option("display.max_columns", None)

  print("Tests for defects:")
  result_def = pd.DataFrame.from_dict(stats_def, orient='index')
  print(result_def.round(4))

  print("Tests for realistic defects:")
  result_real = pd.DataFrame.from_dict(stats_real, orient='index')
  print(result_real.round(4))


if __name__ == '__main__':
  df = pd.read_csv('testing_2seeds_test.csv')
  for p in [0.5, 0.75, 0.9]:
    print(20 * "-", "\nThreshold =", p)
    compare_given_threshold(df, p)
    print(20 * "-")
