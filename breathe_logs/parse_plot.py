import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import sys


metric_to_low_high_values = {'HeartRate': (70, 80),
                             # 'TotalLungVolume': (1950, 2100),
                             'TidalVolume': (200, 500),
                             'RespirationRate': (10, 13.5),
                             'CarbonDioxide': (45, 65),
                             'OxygenSaturation': (0.965, 0.976),
}

label_to_metric = { 'HR': 'HeartRate',
                  # (1950, 2100): 'TotalLungVolume',
                    'TV': 'TidalVolume',
                    'RR': 'RespirationRate',
                    'CD': 'CarbonDioxide',
                    'OX': 'OxygenSaturation',
}


def parse_log(log_file) -> pd.DataFrame:
  with open(log_file, 'r') as file:
    content = file.read()

  pattern = r"---------------------------\n(.*?)\n---------------------------"
  matches = re.findall(pattern, content, re.DOTALL)

  # Parse each block
  data = []
  for match in matches:
    # Extract key-value pairs
    block_data = {}
    for line in match.split('\n'):
      if ': ' in line:
        key, value = line.split(': ', 1)
        try:
          block_data[key.strip()] = float(value.replace('cmH2O', '').strip())
        except ValueError:
          pass
    data.append(block_data)

  # Convert to DataFrame
  df = pd.DataFrame(data).round(4)
  df.set_index('SimTime', inplace=True) 
  return df


def display_dataframe(df_, metrics: list[str] | None = None,
                           colors: pd.DataFrame | None = None,
                           file: str | None = None) -> None:
  df = df_ if metrics is None else df_[metrics]
  numcols = df.shape[1]

  fig, axes = plt.subplots(numcols, 1, figsize=(12, 10))
  if colors is None:
    plot_colors = plt.get_cmap('Set1')
  for i in range(numcols):
    column = df.columns[i]
    ax = axes[i]
    color = plot_colors(i) if colors is None else colors[column]
    ax.scatter(df.index, df.iloc[:, i], marker='.', s=3, ls='', color=color)
    ax.set_title(column)
    ax.set_xlim(-0.01, df.index.max()+0.01)
    ax.grid(axis='both')

  fig.tight_layout(pad=1.0)
  if file is None:
    plt.show()
  else:
    fig.savefig(file)
    print("Plot saved to", file)


if __name__ == '__main__':
  base_name = sys.argv[1]
  log_file = os.path.join('logs', base_name + '.txt')
  os.makedirs('signals', exist_ok=True)
  png_file = os.path.join('signals', base_name + '.png')
  csv_file = os.path.join('signals', base_name + '.csv')
  metrics = list(metric_to_low_high_values.keys())
  df = parse_log(log_file)
  df.to_csv(csv_file)
  display_dataframe(df, metrics=metrics, file=png_file)
