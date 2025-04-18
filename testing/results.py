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

def filter_and_count(df: pd.DataFrame, func: Callable | None,
                     print_prefix: str = "") -> pd.DataFrame:
  if func is not None:
    df_filtered = df.loc[ df.apply(func, axis=1) ]
  else:
    df_filtered = df

  sizes = df_filtered.groupby('seed').size()
  mean = sizes.mean()
  std = sizes.std()
  print(print_prefix, mean, "±", round(std, 2))
  return df_filtered

def compare_given_threshold(df: pd.DataFrame, threshold_prob: float):
  prob_to_bool = lambda p : int(p >= threshold_prob)
  df[query_cols] = df[query_cols].map(prob_to_bool)
  df[all_transition_cols] = df[all_transition_cols].astype(int)

  for tec, group in df.groupby('technique'):
    print(tec)
    filter_and_count(group, None, "Total size")
    filter_and_count(group, is_defect, "Total defects")

    for query in query_cols:
      print(query)
      group_defects = group.loc[ group[query] == 1 ]
      filter_and_count(group_defects, None, "Defects")
      filter_and_count(group_defects, is_realistic, "of which realistic")
    print()


if __name__ == '__main__':
  df = pd.read_csv('testing_2seeds_test.csv')
  for p in [0.5, 0.75, 0.9]:
    print("Threshold =", p)
    compare_given_threshold(df, p)
    print(10*"-")
