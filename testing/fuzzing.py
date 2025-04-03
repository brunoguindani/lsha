from collections import defaultdict
import numpy as np
import os
import re
import subprocess

from generate_automata import fixed_params, write_automaton


class MutationFuzzer:
  params_type = dict[str: float]
  params_bounds = {
    'alpha': (0, 1),
    'beta': (0, 1),
    'doctor_param': (1/15, 1)
  }
  SAT_STRG = '-- Formula is satisfied.'
  NOTSAT_STRG = '-- Formula is NOT satisfied.'
  PATIENT_PATH = os.path.join('..', 'sha_learning', 'resources',
                             'learned_sha', 'safest_04d_delta1.log')
  OUTPUT_ROOT = os.path.join('generated', 'fuzzing')
  REMOVABLE_TRANS_IDS = [10, 11, 12, 13, 14, 15, 17, 19, 20, 21, 22, 23, 24,
                         26, 28, 29, 30, 31]
  TRANS_XML_REGEX = r'<transition id="id{trans_id}">.*?</transition>'
  DOC_XML_REGEX = r'<template>\s*<name[^>]*>\s*Doctor\s*</name>.*?</template>'
  LOC_VERIF_REGEX = r'State:\s*\(.*?patient\.(\w+)\s*\)'
  TRANS_VERIF_REGEX = r'Transition:\s*(patient\.\w+->patient\.\w+)'
  CI_REGEX = r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]\s*\(95% CI\)'

  def __init__(self, mutation_factor: float, seed: int):
    self.mutation_factor = mutation_factor
    self.uppaal_seed = seed
    self.rng = np.random.default_rng(seed)
    self.population = []

  def write_mutant(self, model_file: str, params: params_type) -> str:
    """Insert input params into a template file and save the UPPAAL model"""
    # Extract doctor from automaton file
    doctor_file = self.extract_doctor_from_file(model_file)

    # Convert params dict into string
    all_params = params | fixed_params
    params_strg = ''.join([f'{k}{round(v, 3)}' for k, v in params.items()])

    # Build output file
    model_name = os.path.split(doctor_file)[-1].replace('_doctor.xml', '.xml')
    output_name_no_params = str(model_name).split('__')[-1]
    output_name = params_strg + '__' + output_name_no_params
    output_path = os.path.join(self.OUTPUT_ROOT, output_name)
    write_automaton(self.PATIENT_PATH, doctor_file, output_path, all_params)
    return output_path

  def get_params_from_file(self, file_path: str) -> params_type:
    """Parse parameters from file name"""
    params_strg = os.path.split(file_path)[-1].split('__')[0]
    matches = re.findall(r'([a-zA-Z_]+)(\d+\.\d+)', params_strg)
    params = {k: float(v) for k, v in matches}
    return params

  def extract_doctor_from_file(self, file_path: str) -> str:
    """Find doctor sub-template in file and write it to a new file"""
    with open(file_path, 'r') as f:
      text = f.read()

    match = re.search(self.DOC_XML_REGEX, text, re.DOTALL)
    if match:
      doctor = match.group(0)
      new_file_name = os.path.split(file_path)[-1]
      new_file_name = new_file_name.replace('.xml', '_doctor.xml')
      new_file_path = os.path.join(self.OUTPUT_ROOT, new_file_name)
      with open(new_file_path, 'w') as f:
        f.write(doctor)
      return new_file_path
    else:
      raise RuntimeError(f"Doctor not found in file {file_path}")

  def mutate_parameters(self, params: params_type) -> params_type:
    """Mutate input params by applying a random mutation"""
    # Randomly choose a parameter to mutate
    key = self.rng.choice(tuple(self.params_bounds.keys()))
    print("Mutating parameter", key)
    new_val = params[key]
    # Apply small amount of noise to nominal mutation factor
    factor = self.mutation_factor * self.rng.normal(loc=1.0, scale=0.01)
    # Heads or tails to multiply or divide
    if self.rng.random() < 0.5:
      new_val *= factor
    else:
      new_val /= factor
    # Ensure that the parameter stays within its bounds
    new_val = np.clip(new_val, *self.params_bounds[key])
    # Create output parameter dictionary
    out = params.copy()
    out[key] = new_val
    return out

  def remove_transition(self, model_file: str, index: int) -> str:
    """
    Remove the transition with the given index from the model

    Returns the path to the mutated model, or to the unchanged model if the
    transition is not present
    """
    with open(model_file, 'r') as f:
      text = f.read()
    pattern = self.TRANS_XML_REGEX.format(trans_id=index)
    new_text = re.sub(pattern, '', text, count=1, flags=re.DOTALL)

    new_model_file = model_file.replace('.xml', f'_t{index}.xml')
    with open(new_model_file, 'w') as f:
      f.write(new_text)

    return new_model_file

  def remove_random_transition(self, model_file: str) -> str:
    """
    Mutate input model by randomly removing one among a list of transitions

    Returns the path to the mutated model, or to the unchanged model if the
    sampled transition is not present
    """
    index = self.rng.choice(self.REMOVABLE_TRANS_IDS)
    print("Removing transition", index)
    return self.remove_transition(model_file, index)

  def get_random_parameters(self) -> params_type:
    """Sample parameters uniformly randomly within their bounds"""
    out = {}
    for key, (lb, ub) in self.params_bounds.items():
      out[key] = lb + (ub-lb) * self.rng.random()
    return out

  def sample_and_mutate(self) -> str:
    """
    Choose and mutate a random mutant from the `population`

    Parameter mutation and transition removal are equally likely mutations
    """
    old_mutant = self.rng.choice(self.population)

    # Heads or tails to remove transition or mutate parameters
    if self.rng.random() < 0.5:
      old_params = self.get_params_from_file(old_mutant)
      new_params = self.mutate_parameters(old_params)
      new_mutant = self.write_mutant(old_mutant, new_params)
    else:
      new_mutant = self.remove_random_transition(old_mutant)
    return new_mutant

  def store_mutant(self, mutant: str) -> None:
    """Add mutant to the `population`"""
    self.population.append(mutant)

  def count_mutants(self) -> int:
    """Return size of the stored `population`"""
    return len(self.population)

  def compute_elements_coverage(self, model_file: str, num_runs: int) \
                                -> tuple[dict[str: int], dict[str: int]]:
    """
    Count average number of covered locations and transitions by executing and
    checking `num_runs` traces
    """
    query_idx = 4  # index of fake property to execute simulations
    location_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    base_seed = str(self.uppaal_seed)
    suffix_len = len(str(num_runs)) + 1

    for i in range(num_runs):
      # Build command to execute
      seed_i = base_seed + str(i).zfill(suffix_len)
      cmd = ['verifyta', model_file, '-q', '-r', str(seed_i), '--query-index',
             str(query_idx), '-t0']
      # print(cmd)
      output = subprocess.run(cmd, capture_output=True, text=True).stdout
      # Find and count coverage of locations and transitions
      matches_loc = re.findall(self.LOC_VERIF_REGEX, output)
      for loc in matches_loc:
        location_counts[loc] += 1
      matches_trans = re.findall(self.TRANS_VERIF_REGEX, output)
      for trans in matches_trans:
        transition_counts[trans] += 1

    # Finally, compute averages
    for loc in location_counts.keys():
      location_counts[loc] /= num_runs
    for trans in transition_counts.keys():
      transition_counts[trans] /= num_runs

    self.uppaal_seed += 1
    return location_counts, transition_counts

  def verify_query_bool(self, model_file: str, query_idx: int, seed: int) \
                        -> bool:
    """Return whether query `query_idx` from the UPPAAL model is satisfied"""
    cmd = ['verifyta', model_file, '-q', '-r', str(seed), '--query-index',
           str(query_idx)]
    # print(cmd)
    output = subprocess.run(cmd, capture_output=True, text=True).stdout
    if self.SAT_STRG in output:
      return True
    elif self.NOTSAT_STRG in output:
      return False
    else:
      raise RuntimeError("Formula is neither satisfied nor unsatisfied: "
                         + output)

  def count_verified_queries(self, model_file: str, query_idxs: list[int]) \
                             -> dict[int: int]:
    """Count number of queries within the UPPAAL model that are satisfied"""
    out = dict.fromkeys(query_idxs, 0)
    for idx in query_idxs:
      result = self.verify_query_bool(model_file, idx, self.uppaal_seed)
      # if result: print(idx, result)
      out[idx] += result
      self.uppaal_seed += 1
    return out

  def verify_query_prob(self, model_file: str, query_idx: int, seed: int) \
                        -> float:
    """Compute pointwise estimate of probabilistic query"""
    cmd = ['verifyta', model_file, '-q', '-r', str(seed), '--query-index',
           str(query_idx)]
    # print(cmd)
    output = subprocess.run(cmd, capture_output=True, text=True).stdout
    # Find Confidence Interval in output
    match = re.search(self.CI_REGEX, output)
    if match:
      # Compute midpoint of CI
      return 0.5 * (float(match.group(1)) + float(match.group(2)))
    else:
      raise RuntimeError(f"Confidence Interval not found in {output}")

  def eval_probabilistic_queries(self, model_file: str,
                                 query_idxs: list[int]) -> list[float]:
    """Evaluate probabilities computed in individual UPPAAL queries"""
    out = []
    for idx in query_idxs:
      result = self.verify_query_prob(model_file, idx, self.uppaal_seed)
      out.append(result)
      self.uppaal_seed += 1
    return out



def perform_fuzzing_experiments(mutation_factor: float, use_fuzzing: bool,
                                seed: int) -> int:
  """If use_fuzzing is False, parameters will be uniformly randomly sampled"""
  print("\nFuzzing:", use_fuzzing, "seed:", seed, "\n")
  iterations = 100
  runs_per_simul = 10
  trans_uniform_prob = 0.75

  # Initialize fuzzer and coverage
  fuzzer = MutationFuzzer(mutation_factor, seed)
  initial_model = os.path.join('templates', 'doctor_AC_exp.xml')
  params = {k: (v[0] + v[1])/2 for k, v in fuzzer.params_bounds.items()}
  mutant = fuzzer.write_mutant(initial_model, params)
  fuzzer.store_mutant(mutant)
  print(mutant)

  loc, trans = fuzzer.compute_elements_coverage(mutant, runs_per_simul)
  curr_loc_coverage = len(loc)
  curr_trans_coverage = len(trans)

  for i in range(iterations):
    print("Iteration", i)
    print("Current coverage:", curr_loc_coverage, curr_trans_coverage)
    # Get new mutant, either by mutation or randomly
    if use_fuzzing:
      mutant = fuzzer.sample_and_mutate()
    else:
      # Create random mutant starting from initial model
      mutant = initial_model
      for tr_id in fuzzer.REMOVABLE_TRANS_IDS:
        # Remove each transition with an independent chance
        if fuzzer.rng.random() < trans_uniform_prob:
          mutant = fuzzer.remove_transition(mutant)
      params = fuzzer.get_random_parameters()
      mutant = fuzzer.write_mutant(mutant, params)

    # If either type of coverage increases, mutant will be stored
    loc, trans = fuzzer.compute_elements_coverage(mutant, runs_per_simul)
    num_loc = len(loc)
    num_trans = len(trans)
    if num_loc > curr_loc_coverage:
      print("Location coverage increased to", num_loc)
      curr_loc_coverage = num_loc
      fuzzer.store_mutant(mutant)
    if num_trans > curr_trans_coverage:
      print("Transition coverage increased to", num_trans)
      curr_trans_coverage = num_trans
      fuzzer.store_mutant(mutant)



if __name__ == '__main__':
  mutation_factor = 1.5
  seed = 20250320
  num_experiments = 5

  for use_fuzzing in (True, False):
    for i in range(num_experiments):
      perform_fuzzing_experiments(mutation_factor, use_fuzzing, seed)
      seed += 1
