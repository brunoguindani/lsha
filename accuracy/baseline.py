import numpy as np
import os
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Bidirectional, \
     Conv1D, Dense, Dropout, Embedding, GlobalMaxPooling1D, LSTM, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
import xgboost as xgb


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

def get_cnn_1():
  model = Sequential()
  model.add(Embedding(input_dim=len(encoder.classes_), output_dim=8))
  model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
  model.add(GlobalMaxPooling1D())
  model.add(Dense(16, activation='relu'))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse', metrics=['mae'])

  fit_args = dict(epochs=50, batch_size=2, verbose=0)
  return model, fit_args

def get_cnn_2():
  model = Sequential()
  model.add(Embedding(input_dim=len(encoder.classes_), output_dim=16))
  model.add(Conv1D(filters=64, kernel_size=3, activation='relu',
                   padding='same'))
  model.add(BatchNormalization())
  model.add(Conv1D(filters=64, kernel_size=3, activation='relu',
                   padding='same'))
  model.add(GlobalMaxPooling1D())
  model.add(Dense(32, activation='relu'))
  model.add(Dropout(0.3))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse', metrics=['mae'])

  fit_args = dict(epochs=50, batch_size=2, verbose=0)
  return model, fit_args

def get_rnn_1():
    model = Sequential()
    model.add(Embedding(input_dim=len(encoder.classes_), output_dim=16,
                        mask_zero=True))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    fit_args = dict(epochs=50, batch_size=2, verbose=0)
    return model, fit_args

def get_rnn_2():
    model = Sequential()
    model.add(Embedding(input_dim=len(encoder.classes_), output_dim=32,
                        mask_zero=True))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    fit_args = dict(epochs=50, batch_size=2, verbose=0)
    return model, fit_args

def get_xgboost():
  model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                           objective='reg:squarederror', subsample=0.8,
                           colsample_bytree=0.8, random_state=42, n_jobs=-1)
  return model, {}


if __name__ == '__main__':
  # Read data for training and testing
  _, seq_train, _, tar_train = read_data('training_data.csv')
  labels, seq_test, _, tar_test = read_data('test_data.csv')

  # Fit encoder to transform events into categorical variables
  encoder = LabelEncoder()
  all_letters = sorted(set(l for seq in seq_train for l in seq))
  encoder.fit(all_letters)
  # Get features and targets
  X = pad_sequences([encoder.transform(s) for s in seq_train], padding='post')
  y = np.array(tar_train)

  # Train model
  model, fit_args = get_xgboost()
  model.fit(X, y, **fit_args)

  # Print predictions on test set
  print("\n", type(model), ":", sep="")
  for seq in seq_test:
    # Remove events that do not appear in training traces
    seq = [s for s in seq if s != 'hr1']

    test_encoded = encoder.transform(seq)
    test_encoded = pad_sequences([test_encoded], maxlen=X.shape[1],
                                                 padding='post')
    with HiddenPrints():
      prediction = model.predict(test_encoded)
    print(int(prediction))
