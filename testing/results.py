import pandas as pd

queries = range(3)
all_transitions = range(10, 32)
necessary_transitions = [15, 19, 23]

query_cols = [f'query{q}' for q in queries]
all_transition_cols = [f'l{t}' for t in all_transitions]
nec_transition_cols = [f'l{t}' for t in necessary_transitions]

df = pd.read_csv('testing.csv')

# Pre-processing: turn positive and negative probability differences from
# genetic algorithm into booleans
def prob_delta_to_bool(p):
  if 0 < p < 1:
    return 1.0
  elif -1 <= p < 0:
    return 0.0
  else:
    return p
df[query_cols] = df[query_cols].map(prob_delta_to_bool)
df[all_transition_cols] = df[all_transition_cols].astype(int)

def row_is_valid(row):
  for tr in nec_transition_cols:
    if row[tr] == 1:
      return False
  return True

for tec, group in df.groupby('technique'):
  group_valid = group.loc[ group.apply(row_is_valid, axis=1) ]

  size_full = group.groupby('seed').size().mean()
  group_valid_seed = group_valid.groupby('seed').size()
  mean = group_valid_seed.mean()
  std = group_valid_seed.std()
  print(tec, "->", mean, "±", std)
  print(size_full)

  for query in query_cols:
    group_valid_query = group_valid.loc[ group_valid[query] == 1 ]
    sizes_query = group_valid_query.groupby('seed').size()
    mean_query = sizes_query.mean()
    std_query = sizes_query.std()
    print(" ", query, "->", mean_query, "±", std_query)

  print("\n")
