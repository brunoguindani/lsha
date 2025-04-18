from dataclasses import dataclass
import json
import numpy as np
import os


engine_file = "../breathe.engine/states/StandardFemale@0s.json"
rng = np.random.default_rng(seed=20250417)


@dataclass
class VentilatorParam:
  name: str
  scalar_type: str
  unit: str | None
  delta: float | None
  init: float
  min_val: float
  max_val: float

PARAMS = [
  VentilatorParam("Flow", "ScalarVolumePerTime", "L/min", 5, 60, 0, 100),
  VentilatorParam("FractionInspiredOxygen", "Scalar0To1", None, None, 0.21,
                                                          0.21, 1.0),
  VentilatorParam("InspiratoryPeriod", "ScalarTime", "s", 0.5, 1, 0.1, 5),
  VentilatorParam("PositiveEndExpiratoryPressure", "ScalarPressure", "cmH2O",
                                                          1, 5, 0, 20),
  VentilatorParam("RespirationRate", "ScalarFrequency", "1/min", 2, 12, 5, 40),
  VentilatorParam("TidalVolume", "ScalarVolume", "mL", 100, 900, 100, 2000),
]

INIT_STATE = [p.init for p in PARAMS]


def create_scenario(*args, file_path=None):
  def advance_time_dict():
    return {
      "AdvanceTime": {
        "Action": {},
        "Time": {
          "ScalarTime": {
            "Value": 1.0,
            "Unit": "s"
          }
        }
      }
    }

  def equipment_action_dict(params):
    action = {
      "EquipmentAction": {
        "MechanicalVentilatorVolumeControl": {
          "MechanicalVentilatorMode": {
            "MechanicalVentilatorAction": {
              "EquipmentAction": {
                "Action": {}
              }
            },
            "Connection": "On"
          },
          "InspirationWaveform": "Square"
        }
      }
    }

    for param, value in zip(PARAMS, params):
      ent = {param.scalar_type: {"Value": value}}
      if param.unit:
        ent[param.scalar_type]["Unit"] = param.unit
      action["EquipmentAction"]["MechanicalVentilatorVolumeControl"] \
            [param.name] = ent

    return action

  result = {
    "Name": "myscenario",
    "EngineStateFile": engine_file,
    "AnyAction": []
  }

  for arg in args:
    if isinstance(arg, int):
      result["AnyAction"].extend([advance_time_dict() for _ in range(arg)])
    elif isinstance(arg, dict):
      result["AnyAction"].append(arg)
    elif isinstance(arg, list) and len(arg) == len(PARAMS):
      result["AnyAction"].append(equipment_action_dict(arg))
    else:
      raise ValueError(f"Args must be int, dict, "
                       f"or list of {len(PARAMS)} floats")

  if file_path:
    with open(file_path, "w") as f:
      json.dump(result, f, indent=2)

  return result


def generate_random_scenario(num_events: int, time_interval: tuple[int, int],
                             file_path=None, initial_action: dict = None):
  modifiable_indices = [i for i, p in enumerate(PARAMS) if p.delta is not None]
  if not modifiable_indices:
    raise ValueError("No modifiable parameters: all deltas are None")

  args = []

  # Optional initial action block
  if initial_action:
    args.append(10)  # 10 seconds delay
    args.append(initial_action)

  current_state = INIT_STATE[:]

  for i in range(num_events):
    rand_time = int(rng.integers(time_interval[0], time_interval[1] + 1))
    args.append(rand_time)

    idx = rng.choice(modifiable_indices)
    param = PARAMS[idx]
    delta = param.delta if rng.choice([True, False]) else -param.delta

    new_state = current_state[:]
    new_val = new_state[idx] + delta
    new_state[idx] = int(np.clip(new_val, param.min_val, param.max_val))
    args.append(new_state)

    print(f"Event {i}: delay {rand_time}, param {param.name} "
          f"-> {new_state[idx]}")
    current_state = new_state

  return create_scenario(*args, file_path=file_path)


if __name__ == '__main__':
  root_folder = 'scenarios'

  # Load multiple actions from a file
  with open('custom_actions.json', 'r') as f:
    custom_actions = json.load(f)

  if not isinstance(custom_actions, list):
    raise ValueError("custom_actions.json must contain a list of action dicts")

  for i, action in enumerate(custom_actions):
    file_path = os.path.join(root_folder, f'accuracy_{i}.json')
    print(f"Generating scenario {i} -> {file_path}")
    data = generate_random_scenario(
      num_events=20,
      time_interval=(3, 6),
      file_path=file_path,
      initial_action=action
    )
    print()
