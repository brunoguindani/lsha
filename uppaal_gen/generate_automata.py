import numpy as np
import os
import re

location_template_file = os.path.join('templates', 'location_template.xml')
template_file = os.path.join('templates', 'safest_template.xml')
transition_template_file = os.path.join('templates', 'transition_template.xml')

distribution_regex = r'(?P<name>\w+)\((?P<value>-?\d+\.\d+)\)'
location_regex = r"(?P<name>\w+): (?P<flowcond>.+)"
transition_regex = r"(?P<source>\w+) -> (?P<target>\w+) \((?P<label>[\w.]+)\)"

fixed_params = {
  'patient_param': 0.2,
  'query_bound0': 0.75,  # upper-bound prob for patient stability
  'query_bound1': 0.85,  # lower-bound prob for long non-breathing period
  'query_bound2': 0.95,  # lower-bound prob for critical health (sum of bools)
}


def write_automaton(source_file: str, doctor_path: str, output_path: str,
                    parameters: dict[str: float]):
  # Read files into strings
  with open(template_file, 'r') as f, \
       open(location_template_file, 'r') as fl, \
       open(transition_template_file, 'r') as ft, \
       open(doctor_path, 'r') as fd, open(source_file, 'r') as fs:
    template = f.read()
    location_template = fl.read()
    transition_template = ft.read()
    doctor = fd.read()
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
        # Transition of the form rr1 -> fired by patient
        # Create bool assignment, like 'tv_ok=false'
        bool_name = f'{transition_label[:-1]}_ok'
        bool_val = 'true' if transition_label[-1] == '2' else 'false'
        bool_assignment = f', {bool_name}={bool_val}'
        transition_label += '!'
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

  # Create and save XML
  final_xml = template.format(doctor=doctor, locations=locations_strg,
                              transitions=transitions_strg, **parameters)
  # print(final_xml)
  with open(output_path, 'w') as f:
    f.write(final_xml)
  # print("Saved to", output_path)


if __name__ == '__main__':
  source_name = 'safest_04d_delta1'
  source_path = os.path.join('..', 'sha_learning', 'resources', 'learned_sha',
                             source_name + '.log')
  doctor_name = 'doctor_AC_exp'
  doctor_path = os.path.join('templates', doctor_name + '.xml')
  file_name = source_name + '_' + doctor_name + '.xml'
  output_path = os.path.join('generated', file_name)
  parameters = {'alpha': 0.7, 'beta': 0.5, 'doctor_param': 0.2} | fixed_params
  write_automaton(source_path, doctor_path, output_path, parameters)
