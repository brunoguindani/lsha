from collections.abc import Callable
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

from parse_plot import metric_to_low_high_values, \
                       patient_label_to_metric, ventilator_label_to_metric
from parse_plot import display_dataframe

values_to_colors = {2: 'green', 1: 'red', 3: 'red'}


def midrange_transform(low_bound: float, high_bound: float) \
                       -> Callable[[float], object]:
  def transform_val(x: float) -> object:
    if x < low_bound:
      return 1  # low
    elif x > high_bound:
      return 3  # high
    else:
      return 2  # ok
  return transform_val


def get_peak_smoothing_transformation() -> Callable[[pd.Series], pd.Series]:
    def transform_col(col: pd.Series) -> pd.Series:
      # Get largest value in the first 10 seconds to initialize peak
      prev_peak = max(col.loc[0.00:10.00])

      ret_col = col.copy()
      ret_col.iloc[0] = prev_peak
      for i in range(1, len(col)-1):
        if col.iloc[i] > col.iloc[i+1] and col.iloc[i] >= col.iloc[i-1]:
          # i.e. if i-th value is a peak wrt its previous and next points
          prev_peak = col.iloc[i]
        ret_col.iloc[i] = prev_peak
      ret_col.iloc[-1] = ret_col.iloc[-2]

      return ret_col

    return transform_col


def get_midrange_transformation(metric: str) \
                                -> Callable[[pd.Series], pd.Series]:
  low, high = metric_to_low_high_values[metric]

  def transform_col(col: pd.Series) -> pd.Series:
    transform_val = midrange_transform(low, high)
    return col.apply(transform_val)
  
  return transform_col


def apply_transformations(df: pd.DataFrame, columns: list[str]) \
                          -> pd.DataFrame:
  df_out = pd.DataFrame(index=df.index)
  for column in columns:
    print(column)
    trans = get_midrange_transformation(column)
    print("Applying...")
    df_out[column] = trans(df[column])
    print("Applied")
  return df_out

if __name__ == '__main__':
  base_name = sys.argv[1]
  input_csv = os.path.join('signals', 'accuracy', base_name + '.csv')
  os.makedirs('processed_signals', exist_ok=True)
  processed_csv_folder = os.path.join('processed_signals', 'accuracy', base_name)
  processed_csv = os.path.join(processed_csv_folder, base_name + '.csv')
  os.makedirs('labeled_signals', exist_ok=True)
  labeled_csv = os.path.join('labeled_signals', 'accuracy', base_name + '.csv')
  png_file = os.path.join('labeled_signals', base_name + '.png')
  patient_metrics = list(patient_label_to_metric.values())
  ventilator_metrics = list(ventilator_label_to_metric.values())

  # Read signals
  df_in = pd.read_csv(input_csv, index_col='SimTime')
  print(df_in[patient_metrics])
  # Create processed signals
  df_out = pd.concat((df_in[patient_metrics], df_in[ventilator_metrics]),
                     axis=1)
  peak_smoothing = get_peak_smoothing_transformation()
  df_out['CarbonDioxide'] = peak_smoothing(df_out['CarbonDioxide'])
  print(df_out)
  os.makedirs(processed_csv_folder, exist_ok=True)
  df_out.to_csv(processed_csv)
  print("DataFrame saved to", processed_csv)
  df_labels = apply_transformations(df_out, patient_metrics)
  print(df_labels)
  df_labels.to_csv(labeled_csv)
  print("DataFrame saved to", labeled_csv)
  df_colors = df_labels.replace(values_to_colors)
  # display_dataframe(df_in, metrics=patient_metrics, colors=df_colors,
  #                          file=png_file)
