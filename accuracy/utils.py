import numpy as np
import os
import pandas as pd
import random
import sys
import tensorflow as tf
import warnings


# Set random seeds for reproducibility
seed = 20250424
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
warnings.filterwarnings('ignore')

class HiddenPrints:
  """Context class to suppress printed output"""
  def __enter__(self):
    self._original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
  def __exit__(self, exc_type, exc_val, exc_tb):
    sys.stdout.close()
    sys.stdout = self._original_stdout

def read_data(file: str) -> tuple[list[str], list[list[str]], list[list[int]],
                                  list[int]]:
  df = pd.read_csv(file, index_col='scenario')
  index = df.index.tolist()
  sequences = [row.split(',') for row in df['events']]
  values = [[int(v) for v in row.split(',')] for row in df['signals']]
  targets = df['target'].tolist()
  return index, sequences, values, targets
