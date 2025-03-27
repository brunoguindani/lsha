import matplotlib.pyplot as plt
import pandas as pd

# Must be 'events' or 'traces'
case = 'traces'

df_events_orig = pd.read_csv(f'{case}.csv',
                    usecols=[case, 'time', 'locations', 'transitions'])
df_events_orig['time'] /= 60

events_mean = df_events_orig.groupby(case).mean()
events_std = df_events_orig.groupby(case).std()

column_labels = ['time [min]', 'locations', 'transitions']
events_mean.columns = column_labels
events_std.columns = column_labels
x_values = events_mean.index

fig, ax = plt.subplots()
for col in column_labels:
  mean = events_mean[col]
  std = events_std[col]
  ax.plot(mean, marker='o', label=col)
  ax.fill_between(x_values, mean-std, mean+std, alpha=0.2)
ax.grid()
ax.legend()
ax.set_xlabel(case)

fig.savefig(f'{case}.svg', bbox_inches='tight')
