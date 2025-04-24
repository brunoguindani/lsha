import numpy as np
import os
import pandas as pd
import re

location_template_file = os.path.join('templates', 'location_template.xml')
template_file = os.path.join('templates', 'safest_template.xml')
transition_template_file = os.path.join('templates', 'transition_template.xml')
env_template_file = os.path.join('templates', 'environment_template.xml')
env_loc_template_file = os.path.join('templates', 'env_loc_template.xml')
env_trans_template_file = os.path.join('templates', 'env_trans_template.xml')

distribution_regex = r'(?P<name>\w+)\((?P<value>-?\d+\.\d+)\)'
location_regex = r"(?P<name>\w+): (?P<flowcond>.+)"
transition_regex = r"(?P<source>\w+) -> (?P<target>\w+) \((?P<label>[\w.]+)\)"

fixed_params = {'patient_param': 0.2}


def generate_system_decl(names: list[str]) -> str:
  names_lower = [n.lower() for n in names]
  system_decl = ''
  for name in names_lower:
    name_decl = f'{name} = {name.title()}();\n'
    system_decl += name_decl
  joined_names = ', '.join(names_lower)
  system_decl += f'system {joined_names};\n'
  return system_decl


def _write_automaton(source_file: str, output_path: str, templates_xml: str,
                     environment: bool, parameters: dict[str: float]) -> None:
  # Read files into strings
  with open(template_file, 'r') as f, \
       open(location_template_file, 'r') as fl, \
       open(transition_template_file, 'r') as ft, \
       open(source_file, 'r') as fs:
    template = f.read()
    location_template = fl.read()
    transition_template = ft.read()
    source = fs.readlines()

  # Initialize strings that we will incrementally build
  locations_strg = ''
  transitions_strg = ''

  # Initialize coordinate-related objects
  location_coordinates = {}
  x = 0
  y = 0
  xy_upper_bound = 1000
  rng = np.random.default_rng(seed=20250312)

  # Initialize maps (distributions -> values, locations -> distributions)
  distrib_values = {'None': 400}
  locations_distrib = {}
  patient_symbol = '?' if environment else '!'

  # Collect distribution values
  for line in source:
    match = re.match(distribution_regex, line)
    if match:
      distrib_values[match.group('name')] = int(float(match.group('value')))

  # Parse locations
  for line in source:
    match = re.match(location_regex, line)
    if match:
      # Randomly sample coordinates in the UPPAAL file
      x = rng.integers(xy_upper_bound)
      y = rng.integers(xy_upper_bound)
      location_name = match.group('name')
      # Flow conditions are either like '(f_0, D_14)' or 'None'
      distribution_name = match.group('flowcond').split(', ')[-1]
      locations_distrib[location_name] = distribution_name
      # Initialize location string and append it to the incremental string
      new_location = location_template.format(id=location_name,
        name=location_name,
        x=x, name_x=x-40, inv_x=x-40, rate_x=x-40,
        y=y, name_y=y-40, inv_y=y-30, rate_y=y-20)
      locations_strg += new_location
      location_coordinates[location_name] = (x, y)

  # Parse transitions
  for line in source:
    match = re.match(transition_regex, line)
    if match:
      # Coordinates are the average of source and destination
      source_name = match.group('source')
      target_name = match.group('target')
      x = (location_coordinates[source_name][0] + \
           location_coordinates[target_name][0]) // 2
      y = (location_coordinates[source_name][1] + \
           location_coordinates[target_name][1]) // 2
      transition_label = match.group('label').rstrip('.')
      if len(transition_label) == 3 and transition_label != 'off':
        # Transition of the form rr1 -> fired by patient (or environment)
        # Create bool assignment, like 'tv_ok=false'
        bool_name = f'{transition_label[:-1]}_ok'
        bool_val = 'true' if transition_label[-1] == '2' else 'false'
        bool_assignment = f', {bool_name}={bool_val}'
        transition_label += patient_symbol
      else:
        # Transition of the form rera3 -> listened by patient
        bool_assignment = ''
        transition_label += '?'
      # Initialize transition string
      location_value = distrib_values[ locations_distrib[target_name] ]
      new_transition = transition_template.format(id=line.strip(),
        source=source_name, target=target_name, label=transition_label,
        ass_value=location_value, label_x=x, label_y=y, ass_x=x, ass_y=y+10,
        bool_assignment=bool_assignment)
      transitions_strg += new_transition

  # Create system declaration
  system_comps = ['doctor', 'patient']
  if environment:
    system_comps.append('environment')
  system_decl = generate_system_decl(system_comps)

  # Create and save XML
  final_xml = template.format(templates=templates_xml, system=system_decl,
    locations=locations_strg, transitions=transitions_strg, **parameters)
  # print(final_xml)
  with open(output_path, 'w') as f:
    f.write(final_xml)
  # print("Saved to", output_path)


def write_doctor_patient_automaton(source_file: str, doctor_path: str,
      output_path: str, parameters: dict[str: float]) -> None:
  with open(doctor_path, 'r') as f:
    doctor_xml = f.read()
  _write_automaton(source_file, output_path, doctor_xml, False,
                   parameters)


def generate_environment_xml(events_csv: str) -> str:
  # Read events DataFrame
  df = pd.read_csv(events_csv, index_col='time')
  # Read files into strings
  with open(env_template_file, 'r') as f, \
       open(env_loc_template_file, 'r') as fl, \
       open(env_trans_template_file, 'r') as ft:
    template = f.read()
    location_template = fl.read()
    transition_template = ft.read()

  # Initialize strings that we will incrementally build
  locations_strg = ''
  transitions_strg = ''
  curr_loc_id = 0
  curr_trans_id = 10000
  curr_loc_x = 0
  loc_y = 0
  delta_x = 100

  for t, ev in df.iterrows():
    # UPPAAL has issues with floating-point clocks and initial invariants
    t = int(t)
    # Write location
    new_loc = location_template.format(numid=curr_loc_id, time=t,
      loc_x=curr_loc_x, loc_y=loc_y, inv_x=curr_loc_x, inv_y=loc_y)
    locations_strg += new_loc
    curr_loc_x += delta_x
    # Write transition
    new_trans = transition_template.format(numid=curr_trans_id, time=t,
      label=ev['event']+'!', srcid=curr_loc_id, trgid=curr_loc_id+1,
      guard_x=curr_loc_x, guard_y=loc_y+20, label_x=curr_loc_x, label_y=loc_y+40)
    transitions_strg += new_trans
    # Increase indexes and coordinates
    curr_loc_id += 1
    curr_trans_id += 1
    curr_loc_x += delta_x

  # Last location to end the chain
  new_loc = location_template.format(numid=curr_loc_id, time=9999999,
    loc_x=curr_loc_x, loc_y=loc_y, inv_x=curr_loc_x, inv_y=loc_y)
  locations_strg += new_loc

  final_xml = template.format(locations=locations_strg,
                              transitions=transitions_strg)
  return final_xml


def write_environment_doctor_patient_automaton(source_file: str,
    doctor_path: str, events_csv: str, output_path: str,
    parameters: dict[str: float]) -> None:
  with open(doctor_path, 'r') as f:
    doctor_xml = f.read()
  env_xml = generate_environment_xml(events_csv)
  templates_xml = env_xml + doctor_xml
  _write_automaton(source_file, output_path, templates_xml, True,
                   parameters)


if __name__ == '__main__':
  parameters = {'alpha': 0.5, 'beta': 0.5, 'doctor_param': 0.2} | fixed_params
  source_name = 'safest_04d_delta1'
  source_path = os.path.join('..', 'sha_learning', 'resources', 'learned_sha',
                             source_name + '.log')
  doctor_name = 'doctor_AC_exp'
  doctor_path = os.path.join('templates', doctor_name + '.xml')
  file_name = source_name + '.xml'
  output_path = os.path.join('generated', file_name)
  write_doctor_patient_automaton(source_path, doctor_path, output_path,
                                 parameters)
  env_traces_folder = os.path.join('..', 'breathe_logs', 'environment_traces',
                                   'accuracy')
  for trace_name in os.listdir(env_traces_folder):
    trace_csv_path = os.path.join(env_traces_folder, trace_name)
    output_name = source_name + '_' + trace_name.replace('.csv', '.xml')
    output_path = os.path.join('generated', 'accuracy', output_name)
    print(output_path)
    write_environment_doctor_patient_automaton(source_path, doctor_path,
        trace_csv_path, output_path, parameters)
