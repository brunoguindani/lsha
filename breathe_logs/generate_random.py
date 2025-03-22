import numpy as np
import os
import pandas as pd

from parse_plot import metric_to_low_high_values

SHA_VAR_NAME = 'TidalVolume'
SHA_VAR_BOUNDS = metric_to_low_high_values[SHA_VAR_NAME]

rng = np.random.default_rng(seed=20250318)

patient_states_to_values = {1: 0, 2: None, 3: 1000}
patient_events_to_states = {
  'HeartRate': 2,
  'TidalVolume': 2,
  'RespirationRate': 2,
  'CarbonDioxide': 2,
  'OxygenSaturation': 2,
}
patient_states = set(patient_states_to_values.keys())
patient_events = list(patient_events_to_states.keys())

ventilator_new_states = (1, 3)
ventilator_events = [
  'FractionInspiredOxygen',
  'PositiveEndExpiratoryPressure',
  'RespirationRate_vent',
  'TidalVolume_vent',
  'off.',
]
ventilator_df_keys = ventilator_events[0:-1]  # excluding off

ventilator_on_states = list(patient_events) + ventilator_events
ventilator_off_states = patient_events + ['on.']


def generate_random_signal(num_events: int, event_delta: float,
                           row_delta: float, output_path: str):
  ventilator_on = False
  curr_time = 0.0

  dir_path = os.path.dirname(os.path.realpath(__file__))
  init_path = os.path.join(dir_path, 'initial_state.csv')
  df = pd.read_csv(init_path, index_col='SimTime')
  init_row = df.loc[0.0].copy()
  prev_row = init_row.copy()

  # Fill df with one row per event
  for i in range(num_events):
    curr_time += event_delta
    new_row = prev_row.copy()
    # Make list of valid events and sample among them
    events = ventilator_on_states if ventilator_on else ventilator_off_states
    new_event = rng.choice(events)
    print(i, curr_time, new_event, end=" ")
    if new_event == 'on.':
      ventilator_on = True
      new_row[ventilator_df_keys] = 0
      print()
    elif new_event == 'off.':
      ventilator_on = False
      new_row[ventilator_df_keys] = np.nan
      print()
    elif new_event in ventilator_events:
      # Sample 1 (metric increased) or 3 (decreased)
      new_state = rng.choice(ventilator_new_states)
      new_row[new_event] += (+1 if new_state == 3 else -1)
      print(new_state)
    else:
      # Choose among 1/2/3 (low/ok/high) but excluding the current value
      current_state = patient_events_to_states[new_event]
      valid_states = tuple(set(patient_states).difference({current_state}))
      new_state = rng.choice(valid_states)
      if new_event == SHA_VAR_NAME:
        # Three cases to sample realistic value for modeled variable
        if new_state == 2:
          # Take original OK value
          new_row[new_event] = init_row[new_event]
        elif new_state == 1:
          # Sample low value
          new_row[new_event] = SHA_VAR_BOUNDS[0] * rng.random()
        else:  # if new_state == 3:
          # Sample high value
          new_row[new_event] = 1 + SHA_VAR_BOUNDS[1]
      elif new_state == 2:
        # Take original OK value
        new_row[new_event] = init_row[new_event]
      else:
        # Set low (0) or high (1000) value (value itself doesn't matter)
        new_row[new_event] = patient_states_to_values[new_state]
      patient_events_to_states[new_event] = new_state
      print(new_state)

    # Write new row and store it as previous row
    df.loc[curr_time] = new_row
    prev_row = new_row.copy()

  # Set correct sample frequency (no events happen in newly added rows).
  # Note that there needs to be a row after the last event to prevent an empty
  # segment (hence the 2 in the next line)
  new_index = np.arange(df.index[0], df.index[-1]+2*row_delta, row_delta)
  df_new = df.fillna(np.inf).reindex(new_index).ffill().replace(np.inf, np.nan)
  df_new.to_csv(output_path)
  print("Saved to", output_path)


if __name__ == '__main__':
  max_events = 50
  event_delta = 6.0
  row_delta = 1.0
  for i in range(1, max_events+1):
    base_name = 'SIM' + str(i).zfill(3)
    folder_path = os.path.join('generated', base_name)
    os.makedirs(folder_path, exist_ok=True)
    output_path = os.path.join(folder_path, base_name + '.csv')
    generate_random_signal(i, event_delta, row_delta, output_path)
