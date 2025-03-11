import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import sys


metric_to_low_high_values = {'HeartRate': (65, 90),
                             'TidalVolume': (300, 600),
                             'RespirationRate': (10, 17),
                             'CarbonDioxide': (40, 65),
                             'OxygenSaturation': (0.915, 0.99),
}

# Apparently stuff breaks down when changing the order of these metrics
ventilator_metrics = ['FractionInspiredOxygen', 'RespirationRate_vent',
  'PositiveEndExpiratoryPressure', 'TidalVolume_vent']

label_to_metric = { 'hr': 'HeartRate',
                    'tv': 'TidalVolume',
                    'rr': 'RespirationRate',
                    'cd': 'CarbonDioxide',
                    'ox': 'OxygenSaturation',
                    'fiox': 'FractionInspiredOxygen', 
                    'peep': 'PositiveEndExpiratoryPressure',
                    'rera': 'RespirationRate',
                    'tvol': 'TidalVolume',
}


def parse_log(log_file) -> pd.DataFrame:
  with open(log_file, 'r') as file:
    content = file.read()

  # Use regex to extract sections
  pattern = r"---------------------------\n(.*?)\n---------------------------"
  matches = re.findall(pattern, content, re.DOTALL)

  data = []
  for match in matches:
    block_data = {}
    feature_counts = {}

    for line in match.split('\n'):
      if ': ' in line:
        key, value = line.split(': ', 1)
        key = key.strip()
        value = value.strip()

        # Ensure numerical values are correctly converted
        try:
          value = float(value.replace('L/min', '')
                              .replace('cmH2O', '')
                              .replace('mL', '')
                              .replace('/min', '')
                              .replace('s', ''))
        except ValueError:
          pass  # Keep as string if conversion fails

        # Handle duplicate feature names
        if key in feature_counts:
          feature_counts[key] += 1
          key = f"{key}_vent"  # Append '_vent' to second occurrence
        else:
          feature_counts[key] = 1

        block_data[key] = value

    data.append(block_data)

  # Convert to DataFrame
  df = pd.DataFrame(data)
  df.set_index('SimTime', inplace=True)
  return df


def display_dataframe(df_, metrics: list[str] | None = None,
                           colors: pd.DataFrame | None = None,
                           file: str | None = None) -> None:
  print("Preparing dataset for", file)
  df = df_ if metrics is None else df_[metrics]
  numcols = df.shape[1]

  fig, axes = plt.subplots(numcols, 1, figsize=(12, 18))
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
    print("Plot saved to", file, "\n")


if __name__ == '__main__':
  base_name = sys.argv[1]
  log_file = os.path.join('logs', base_name + '.txt')
  os.makedirs('signals', exist_ok=True)
  png_file = os.path.join('signals', base_name + '.png')
  csv_file = os.path.join('signals', base_name + '.csv')

  df = parse_log(log_file)
  # Remove outliers
  df['RespirationRate'] = df['RespirationRate'].clip(upper=40)
  df['TidalVolume'] = df['TidalVolume'].clip(upper=900)
  df.to_csv(csv_file)

  patient_metrics = list(metric_to_low_high_values.keys())
  all_metrics = patient_metrics + ventilator_metrics
  display_dataframe(df, metrics=all_metrics, file=png_file)
