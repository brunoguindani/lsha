import os
import pygad
import re

from fuzzing import MutationFuzzer
from generate_automata import query_bounds, query_bound_is_upper


class MonoObjectiveGeneticSearcher(MutationFuzzer):
  BASE_FILE = os.path.join('generated', 'safest_04d_delta1_doctor_AC_exp.xml')
  TRANS_NAME_REGEX = r"_t(\d+)"

  def __init__(self, seed: int, query_idxs: list[int]):
    super().__init__(None, seed)
    # Indexes of queries to evaluate fitness
    self.query_idxs = query_idxs
    # Bounds for doctor parameters
    self.space = [{'low': l, 'high': u}
                  for l, u in super().params_bounds.values()
    ]
    self.num_params = len(self.space)
    self.num_trans = len(super().REMOVABLE_TRANS_IDS)
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
    return file

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

  def get_signed_distances(self, individual: list) -> tuple[float]:
    """Compute signed distances associated with each query"""
    model_file = self.values_to_file(individual)
    probs = self.eval_probabilistic_queries(model_file, self.query_idxs)
    distances = []
    for prob, bound, is_upper in zip(probs, query_bounds.values(),
                                     query_bound_is_upper):
      dist = self.get_signed_distance(prob, bound, is_upper)
      distances.append(dist)
    # print(distances)
    return tuple(distances)


  def get_unidim_fitness(self, ga: pygad.GA, individual: list, idx: int) \
                         -> float:
    """
    Build mutant from vector of `individual` and compute total fitness score

    `ga` and `idx` parameters are required by the PyGAD API
    """
    return sum(self.get_signed_distances(individual))

  def run_GA(self):
    """Run the Genetic Algorithm optimization"""
    ga = pygad.GA(num_generations=50, num_parents_mating=5,
                  fitness_func=self.get_unidim_fitness, sol_per_pop=10,
                  num_genes=self.num_genes, gene_space=self.space,
                  mutation_percent_genes=20,
                  random_seed=self.rng.integers(20250000))
    ga.run()
    solution, solution_fitness, _ = ga.best_solution()
    print("Best fitness:", solution_fitness)
    print("Best mutant:", solution)
    print(f"Population:\n", ga.population, sep="")



if __name__ == '__main__':
  searcher = MonoObjectiveGeneticSearcher(20250403, [5, 6, 7])
  searcher.run_GA()
