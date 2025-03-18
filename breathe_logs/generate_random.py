import numpy as np
import os
import pandas as pd

rng = np.random.default_rng(seed=20250318)

row_delta = 0.04

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


def generate_random_signal(base_name: str, num_events: int,
                           event_delta: float):
  ventilator_on = False
  curr_time = 0.0

  df = pd.read_csv('initial_state.csv', index_col='SimTime')
  init_row = df.loc[0.0].copy()
  prev_row = init_row.copy()

  for i in range(num_events):
    curr_time += event_delta
    new_row = prev_row.copy()
    events = ventilator_on_states if ventilator_on else ventilator_off_states
    event = rng.choice(events)
    print(i, curr_time, event, end=" ")
    if event == 'on.':
      ventilator_on = True
      new_row[ventilator_df_keys] = 0
      print()
    elif event == 'off.':
      ventilator_on = False
      new_row[ventilator_df_keys] = np.nan
      print()
    elif event in ventilator_events:
      new_state = rng.choice(ventilator_new_states)
      new_row[event] += (+1 if new_state == 3 else -1)
      print(new_state)
    else:
      current_state = patient_events_to_states[event]
      valid_states = tuple(set(patient_states).difference({current_state}))
      new_state = rng.choice(valid_states)
      new_row[event] = (init_row[event] if new_state == 2 \
                         else patient_states_to_values[new_state])
      patient_events_to_states[event] = new_state
      print(new_state)

    df.loc[curr_time] = new_row
    prev_row = new_row.copy()

  new_index = np.arange(df.index[0], df.index[-1]+row_delta, row_delta)
  df_new = df.fillna(np.inf).reindex(new_index).ffill().replace(np.inf, np.nan)
  csv_folder = os.path.join('generated', base_name)
  os.makedirs(csv_folder, exist_ok=True)
  csv_path = os.path.join(csv_folder, base_name+'.csv')
  df_new.to_csv(csv_path)
  print("Saved to", csv_path)


if __name__ == '__main__':
  max_events = 100
  event_delta = 5.0
  for i in range(1, max_events+1):
    base_name = f'SIM_zzz_generated_{str(i).zfill(3)}'
    generate_random_signal(base_name, i, event_delta)
