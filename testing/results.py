import pandas as pd

queries = range(3)
all_transitions = range(10, 32)
necessary_transitions = [13, 21]  # TODO just an example

query_cols = [f'query{q}' for q in queries]
all_transition_cols = [f'l{t}' for t in all_transitions]
nec_transition_cols = [f'l{t}' for t in necessary_transitions]

df = pd.read_csv('testing.csv')
print(df)

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
print(df)

def row_is_valid(row):
  for tr in nec_transition_cols:
    if row[tr] == 1:
      return False
  return True

for tec, group in df.groupby('technique'):
  print(tec)
  group_valid = group.loc[ group.apply(row_is_valid, axis=1) ]

  full_size = group.groupby('seed').size().mean()
  valid_size = group_valid.groupby('seed').size().mean()
  print(full_size, valid_size)
