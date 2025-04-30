import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Conv1D, Dense, Dropout, \
     Embedding, GlobalMaxPooling1D, Input, Lambda, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import HiddenPrints, read_data


def get_event_signal_model(max_event_len: int, max_signal_len: int):
  # Event branch
  event_input = Input(shape=(max_event_len,), name='event_input')
  x1 = Embedding(input_dim=len(encoder.classes_), output_dim=12,
                 mask_zero=True)(event_input)
  x1 = LSTM(24)(x1)
  # Signal branch
  signal_input = Input(shape=(max_signal_len,), name='signal_input')
  x2 = Lambda(lambda x: tf.expand_dims(x, -1))(signal_input)
  x2 = Conv1D(16, kernel_size=3, activation='relu', padding='same')(x2)
  x2 = GlobalMaxPooling1D()(x2)
  # Combine
  combined = Concatenate()([x1, x2])
  x = Dense(32, activation='relu')(combined)
  x = Dropout(0.2)(x)
  output = Dense(1)(x)
  # Create and compile model
  model = Model(inputs=[event_input, signal_input], outputs=output)
  model.compile(optimizer='adam', loss='mse', metrics=['mae'])
  # Create arguments for fit() function
  fit_args = dict(epochs=50, batch_size=2, verbose=0)
  return model, fit_args


if __name__ == '__main__':
  # Read data
  _, seq_train, sig_train, tar_train = read_data('training_data.csv')
  labels, seq_test, sig_test, tar_test = read_data('test_data.csv')
  # Fit encoder on training events and encode full event sequences
  encoder = LabelEncoder()
  all_events = sorted(set(e for s in seq_train for e in s))
  encoder.fit(all_events)
  X_event = pad_sequences([encoder.transform(s) for s in seq_train],
                          padding='post')
  max_event_len = X_event.shape[1]
  # Pad signal values
  X_signal = pad_sequences(sig_train, padding='post', dtype='float32')
  max_signal_len = X_signal.shape[1]
  y_train = np.array(tar_train)
  # Build and train model
  model, fit_args = get_event_signal_model(max_event_len, max_signal_len)
  model.fit({'event_input': X_event, 'signal_input': X_signal}, y_train,
            **fit_args)
  # Compute predictions
  print("\n", type(model), ":", sep="")
  for seq, sig in zip(seq_test, sig_test):
    # Clean up test sequences
    seq = [s for s in seq if s in encoder.classes_]

    test_event = pad_sequences([encoder.transform(seq)],
                               maxlen=max_event_len, padding='post')
    test_signal = pad_sequences([sig], maxlen=max_signal_len,
                                padding='post', dtype='float32')

    with HiddenPrints():
      pred = model.predict({'event_input': test_event,
                            'signal_input': test_signal})
    print(int(pred[0][0]))
