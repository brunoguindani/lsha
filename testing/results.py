from collections.abc import Callable
import pandas as pd

from fuzzing import MutationFuzzer


query_idxs = MutationFuzzer.QUERY_IDXS
all_transitions = MutationFuzzer.REMOVABLE_TRANS_IDS
necessary_transitions = [15, 19, 23]

query_cols = [f'query{q}' for q in query_idxs]
all_transition_cols = [f'l{t}' for t in all_transitions]
nec_transition_cols = [f'l{t}' for t in necessary_transitions]


def is_defect(row: pd.Series) -> bool:
  for col in query_cols:
    if row[col] == 1:
      return True
  return False

def is_realistic(row: pd.Series) -> bool:
  for tr in nec_transition_cols:
    if row[tr] == 1:
      return False
  return True

def filter(df: pd.DataFrame, func: Callable) -> pd.DataFrame:
  return df.loc[ df.apply(func, axis=1) ]

def count(df: pd.DataFrame, print_prefix: str = ""):
  sizes = df.groupby('seed').size()
  mean = sizes.mean()
  std = sizes.std()
  print(print_prefix, mean, "±", round(std, 2))
  # print(list(sizes))

def compare_given_threshold(input_df: pd.DataFrame, threshold_prob: float):
  prob_to_bool = lambda p : int(p >= threshold_prob)
  df = input_df.copy()
  df[query_cols] = df[query_cols].map(prob_to_bool)
  df[all_transition_cols] = df[all_transition_cols].astype(int)

  # Loop over queries
  for query in query_cols:
    print("Requirement:", query)
    # Loop over techniques
    for tec, group in df.groupby('technique'):
      # Count number of iterations
      print("Technique", tec)
      # count(group, " Total size")
      # Filter on mutants that violate query
      group_def = group.loc[ group[query] == 1 ]
      count(group_def, " Defects")
      group_def_real = filter(group_def, is_realistic)
      count(group_def_real, " of which realistic")
    print()


if __name__ == '__main__':
  df = pd.read_csv('testing_2seeds_test.csv')
  for p in [0.5, 0.75, 0.9]:
    print(20*"-", "\nThreshold =", p)
    compare_given_threshold(df, p)
    print(20*"-")
