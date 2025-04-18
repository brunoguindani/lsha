import os
import pygad
import re

from fuzzing import MutationFuzzer, patient_name


class MonoObjectiveGeneticSearcher(MutationFuzzer):
  BASE_FILE = os.path.join('generated', patient_name + '.xml')

  def __init__(self, seed: int, log_file: str):
    super().__init__(None, seed, log_file)
    # Bounds for doctor parameters
    self.space = [{'low': l, 'high': u}
                  for l, u in super().params_bounds.values()
    ]
    self.num_genes = self.num_params + self.num_trans
    # Boolean-like variables for including (0) or excluding (1) transitions
    self.space.extend([[0, 1] for _ in range(self.num_trans)])

  def values_to_file(self, values: list) -> str:
    # Split variables representing parameters and transition bools
    params_values = values[:self.num_params]
    trans_bools = values[self.num_params:]
    # Write initial mutant to file after building parameters dict
    params_dict = {k: v
                   for k, v in zip(self.params_bounds.keys(), params_values)}
    file = self.write_mutant(self.BASE_FILE, params_dict)
    # Remove transitions associated with a value of 1
    for b, trans_id in zip(trans_bools, self.REMOVABLE_TRANS_IDS):
      if b:
        file = self.remove_transition(file, trans_id)
    print(file)
    return file

  def write_to_log(self, *args) -> None:
    strg = ','.join([str(_) for _ in args]) + '\n'
    with open(self.log_file, 'a') as f:
      f.write(strg)

  def get_signed_distance(self, value: float, bound: float,
                          bound_is_upper: bool) -> float:
    """
    Return signed distance from bound

    We are testing to find "bad" doctors. UPPAAL queries represent negative
    properties, i.e., properties that a bad doctor satisfies. For instance, if
    we have an upper-bound probability property, we'd like to be as deep as
    possible within the bounded region, i.e., as far away as possible from
    the `bound` by having a low probability `value`. The smaller `value` is
    compared to `bound`, the better: we return large positive values by
    subtracting `value` from `bound`.
    The reverse is true is the property is lower-bound
    """
    if bound_is_upper:
      return bound - value
    else:
      return value - bound

  def get_fitness(self, ga: pygad.GA, individual: list, idx: int) -> float:
    """
    Build mutant from `individual` and compute uni-dim. total fitness score

    Since we are testing to find "bad" doctors, the fitness to maximize is
    based on the probability of certain negative properties holding: the
    larger, the better. Note: the `ga` and `idx` parameters are required by the
    PyGAD API
    """
    model_file = self.values_to_file(individual)
    probs = self.eval_probabilistic_queries(model_file)
    return sum(probs)

  def run_GA(self):
    """Run the Genetic Algorithm maximizing the fitness score"""
    ga = pygad.GA(num_generations=50, num_parents_mating=5,
                  fitness_func=self.get_fitness, sol_per_pop=10,
                  num_genes=self.num_genes, gene_space=self.space,
                  mutation_percent_genes=20,
                  random_seed=self.rng.integers(20250000))
    ga.run()
    solution, solution_fitness, _ = ga.best_solution()
    print("Best fitness:", solution_fitness)
    print("Best mutant:", solution)
    print(f"Population:\n", ga.population, sep="")



class MultiObjectiveGeneticSearcher(MonoObjectiveGeneticSearcher):
  def get_fitness(self, ga: pygad.GA, individual: list, idx: int) \
                  -> tuple[float]:
    """
    Build mutant from `individual` and compute multi-dim. fitness scores

    Since we are testing to find "bad" doctors, the fitness to maximize is
    based on the probability of certain negative properties holding: the
    larger, the better. Note: the `ga` and `idx` parameters are required by the
    PyGAD API
    """
    model_file = self.values_to_file(individual)
    probs = self.eval_probabilistic_queries(model_file)
    return tuple(probs)

  def run_GA(self, seed: int):
    """Run the Genetic Algorithm maximizing the fitness score"""
    ga = pygad.GA(num_generations=49, num_parents_mating=5,
                  fitness_func=self.get_fitness, sol_per_pop=10,
                  num_genes=self.num_genes, gene_space=self.space,
                  mutation_percent_genes=20, save_solutions=True,
                  random_seed=int(self.rng.integers(20250000)))
    ga.run()
    solution, solution_fitness, _ = ga.best_solution()
    print("Fitness values with largest element:", solution_fitness)
    print(f"Pareto front:\n", ga.pareto_fronts, sep="")
    for j in range(len(ga.solutions)):
      sol = ga.solutions[j]
      fit = ga.solutions_fitness[j]
      self.write_to_log(*sol, *fit, seed, 'genetic')



if __name__ == '__main__':
  seed = 20250320
  num_experiments = 20
  log_file = 'testing.csv'

  for i in range(num_experiments):
    searcher = MultiObjectiveGeneticSearcher(seed, log_file)
    searcher.run_GA(seed)
    seed += 1
