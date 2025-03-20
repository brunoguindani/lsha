import numpy as np
import os
import subprocess

from generate_automata import fixed_params, write_automaton


class Fuzzer:
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

  def parameters_to_model_file(self, doctor_params: dict[str: float]) -> str:
    source_name = 'safest_04d_delta1'
    output_basename = '04d_'
    source_path = os.path.join('..', 'sha_learning', 'resources',
                               'learned_sha', source_name + '.log')
    doctor_name = 'doctor_AC_exp'
    doctor_path = os.path.join('templates', doctor_name + '.xml')
    all_params = doctor_params | fixed_params
    params_strg = '_'.join([f'{k}{round(v, 3)}' for k, v in params.items()])
    output_name = output_basename + params_strg + '.xml'
    output_path = os.path.join('generated', 'fuzzing', output_name)
    write_automaton(source_path, doctor_path, output_path, all_params)
    return output_path

  def mutate_parameters(self, params: dict[str: float]) -> dict[str: float]:
    key = self.rng.choice(tuple(self.params_bounds.keys()))
    new_val = params[key]
    if self.rng.random() < 0.5:  # heads or tails
      new_val *= self.mutation_factor
    else:
      new_val /= self.mutation_factor
    new_val = np.clip(new_val, *self.params_bounds[key])
    out = params.copy()
    out[key] = new_val
    return out

  def sample_mutant(self) -> dict[str: float]:
    index = self.rng.integers(len(self.population))
    return self.mutate_parameters(self.population[index])

  def verify_query(self, model_file: str, query_idx: int, seed: int) -> bool:
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
    verified_queries = 0
    for query_idx in range(self.num_queries):
      result = self.verify_query(model_file, query_idx, self.uppaal_seed)
      if result:
        print(query_idx, result)
      verified_queries += result
      self.uppaal_seed += 1
    return verified_queries

  def get_coverage(self, params: dict[str: float]) -> int:
    model_file = self.parameters_to_model_file(params)
    return self.count_verified_queries(model_file)


if __name__ == '__main__':
  num_queries = 3
  iterations = 200
  seed = 20250320
  mutation_factor = 1.2

  # Initialize fuzzer and coverage
  fuzzer = Fuzzer(num_queries, mutation_factor, seed)
  params = {k: (v[0] + v[1])/2 for k, v in fuzzer.params_bounds.items()}
  # print(params)
  coverage = fuzzer.get_coverage(params)
  fuzzer.population.append(params)
  print()

  for i in range(iterations):
    params = fuzzer.sample_mutant()
    coverage = fuzzer.get_coverage(params)
    # print(coverage, "<-", params)
    if coverage > 0:
      fuzzer.population.append(params)
    print()
