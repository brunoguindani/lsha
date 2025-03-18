import numpy as np
import os
import re

doctor_file = os.path.join('templates', 'doctor.xml')
location_template_file = os.path.join('templates', 'location_template.xml')
template_file = os.path.join('templates', 'safest_template.xml')
transition_template_file = os.path.join('templates', 'transition_template.xml')

distribution_regex = r'(?P<name>\w+)\((?P<value>-?\d+\.\d+)\)'
location_regex = r"(?P<name>\w+): (?P<flowcond>.+)"
transition_regex = r"(?P<source>\w+) -> (?P<target>\w+) \((?P<label>[\w.]+)\)"


def write_automaton(source_file: str, parameters: dict[str: float]):
  # Read files into strings
  with open(template_file, 'r') as f, \
       open(location_template_file, 'r') as fl, \
       open(transition_template_file, 'r') as ft, \
       open(doctor_file, 'r') as fd, open(source_file, 'r') as fs:
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
        transition_label += '!'
      else:
        # Transition of the form rera3 -> listened by patient
        transition_label += '?'
      # Initialize transition string
      location_value = distrib_values[ locations_distrib[target_name] ]
      print(target_name, location_value)
      new_transition = transition_template.format(id=line.strip(),
        source=source_name, target=target_name, label=transition_label,
        ass_value=location_value, label_x=x, label_y=y, ass_x=x, ass_y=y+10)
      transitions_strg += new_transition

  # Create and save XML
  final_xml = template.format(doctor=doctor, locations=locations_strg,
                              transitions=transitions_strg, **parameters)
  print(final_xml)
  file_name = os.path.split(source_file)[-1].replace('.log', '.xml')
  output_path = os.path.join('generated', file_name)
  with open(output_path, 'w') as f:
    f.write(final_xml)


if __name__ == '__main__':
  name = 'safest_with_doctor_04d_delta1'
  print(os.getcwd())
  source_path = os.path.join('..', 'sha_learning', 'resources', 'learned_sha',
                             name + '.log')
  parameters = dict(alpha=0.7, beta=0.5, doctor_rate=0.2, patient_rate=0.2)
  write_automaton(source_path, parameters)
