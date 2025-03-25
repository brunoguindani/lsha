import numpy as np
import os
import subprocess

from generate_automata import fixed_params, write_automaton


class Fuzzer:
  params_type = dict[str: float]
  params_bounds = {
    'alpha': (0, 1),
    'beta': (0, 1),
    'doctor_param': (1/15, 1)
  }
  SAT_STRG = '-- Formula is satisfied.'
  NOTSAT_STRG = '-- Formula is NOT satisfied.'

  def __init__(self, num_queries: int, mutation_factor: float, seed: int):
    self.num_queries = num_queries
    self.mutation_factor = mutation_factor
    self.uppaal_seed = seed
    self.rng = np.random.default_rng(seed)
    self.population = []

  def parameters_to_model_file(self, doctor_params: params_type) -> str:
    """Insert input params into a template file and save the Uppaal model"""
    source_name = 'safest_04d_delta1'
    output_basename = '04d_'
    source_path = os.path.join('..', 'sha_learning', 'resources',
                               'learned_sha', source_name + '.log')
    doctor_name = 'doctor_AC_exp'
    doctor_path = os.path.join('templates', doctor_name + '.xml')
    all_params = doctor_params | fixed_params
    params_strg = '_'.join([f'{k}{round(v, 3)}' \
                           for k, v in doctor_params.items()])
    output_name = output_basename + params_strg + '.xml'
    output_path = os.path.join('generated', 'fuzzing', output_name)
    write_automaton(source_path, doctor_path, output_path, all_params)
    return output_path

  def mutate_parameters(self, params: params_type) -> params_type:
    """Mutate input params by applying a random mutation"""
    # Randomly choose a parameter to mutate
    key = self.rng.choice(tuple(self.params_bounds.keys()))
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

  def get_random_parameters(self) -> params_type:
    """Sample parameters uniformly randomly within their bounds"""
    out = {}
    for key, (lb, ub) in self.params_bounds.items():
      out[key] = lb + (ub-lb) * self.rng.random()
    return out

  def sample_mutant(self) -> params_type:
    """Choose and mutate a random mutant from `population`"""
    index = self.rng.integers(len(self.population))
    return self.mutate_parameters(self.population[index])

  def verify_query(self, model_file: str, query_idx: int, seed: int) -> bool:
    """Return whether query `query_idx` from the Uppaal model is satisfied"""
    cmd = ['verifyta', model_file, '-q', '-r', str(seed), '--query-index',
           str(query_idx)]
    output = subprocess.run(cmd, capture_output=True, text=True).stdout
    if self.SAT_STRG in output:
      return True
    elif self.NOTSAT_STRG in output:
      return False
    else:
      raise RuntimeError("Formula is neither satisfied nor unsatisfied: "
                         + output)

  def count_verified_queries(self, model_file: str) -> int:
    """Count number of queries within the Uppaal model that are satisfied"""
    verified_queries = 0
    for query_idx in range(self.num_queries):
      result = self.verify_query(model_file, query_idx, self.uppaal_seed)
      # if result: print(query_idx, result)
      verified_queries += result
      self.uppaal_seed += 1
    return verified_queries

  def get_coverage(self, params: params_type) -> int:
    """Utility method combining the other class methods"""
    model_file = self.parameters_to_model_file(params)
    return self.count_verified_queries(model_file)



def perform_fuzzing_experiments(mutation_factor: float, use_fuzzing: bool,
                                seed: int) -> int:
  """If use_fuzzing is False, parameters will be uniformly randomly sampled"""
  num_queries = 3
  iterations = 100

  # Initialize fuzzer and coverage
  fuzzer = Fuzzer(num_queries, mutation_factor, seed)
  params = {k: (v[0] + v[1])/2 for k, v in fuzzer.params_bounds.items()}
  coverage = fuzzer.get_coverage(params)
  fuzzer.population.append(params)

  for i in range(iterations):
    params = fuzzer.sample_mutant() if use_fuzzing \
                                    else fuzzer.get_random_parameters()
    # print(params)
    coverage = fuzzer.get_coverage(params)
    if coverage > 0:
      fuzzer.population.append(params)

  return len(fuzzer.population)



if __name__ == '__main__':
  mutation_factors = [1.05, 1.1, 1.2, 1.25, 1.5]
  base_seed = 20250320
  num_experiments = 5

  for mut in mutation_factors:
    for use_fuzzing in (False, True):
      populations = []
      for i in range(num_experiments):
        pop = perform_fuzzing_experiments(mut, use_fuzzing, base_seed+i)
        populations.append(pop)
      print(mut, use_fuzzing, *(populations), np.mean(populations),
                                              np.std(populations), sep=',')
