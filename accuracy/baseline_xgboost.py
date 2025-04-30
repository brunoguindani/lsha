import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import xgboost as xgb

from utils import read_data, seed


def build_features(event_seqs, signal_seqs, max_event_len, max_signal_len):
  # Remove unknown events
  known_events = set(encoder.classes_)
  cleaned_event_seqs = [[e for e in seq if e in known_events]
                           for seq in event_seqs]
  # Encode signals and pad events/signals
  X_event = [encoder.transform(seq) for seq in cleaned_event_seqs]
  X_event = pad_sequences(X_event, padding='post', maxlen=max_event_len)
  X_signal = pad_sequences(signal_seqs, padding='post', maxlen=max_signal_len,
                                        dtype='float32')
  # Combine features
  return np.hstack([X_event, X_signal])

def get_xgboost_model():
  return xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=seed,
                          objective='reg:squarederror')

if __name__ == '__main__':
  # Read data
  _, seq_train, sig_train, tar_train = read_data('training_data.csv')
  labels, seq_test, sig_test, tar_test = read_data('test_data.csv')
  # Fit encoder on training events
  encoder = LabelEncoder()
  encoder.fit([event for seq in seq_train for event in seq])
  # Prepare training features
  max_event_len = max(len(seq) for seq in seq_train)
  max_signal_len = max(len(sig) for sig in sig_train)
  X_train = build_features(seq_train, sig_train, max_event_len, max_signal_len)
  y_train = np.array(tar_train)
  # Build and train model
  model = get_xgboost_model()
  model.fit(X_train, y_train)
  # Prepare test sequences
  X_test = build_features(seq_test, sig_test, max_event_len, max_signal_len)
  # Compute predictions
  predictions = model.predict(X_test)
  for pred in predictions:
    print(int(pred))
