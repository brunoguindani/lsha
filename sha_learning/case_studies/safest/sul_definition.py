import os
from pygad import GA

from sha_learning.case_studies.safest.sul_functions import label_event, parse_data, get_vol_param, is_chg_pt
from sha_learning.case_studies.safest.sul_functions import metrics_to_labels
from sha_learning.domain.lshafeatures import Event, NormalDistribution, Trace
from sha_learning.domain.sigfeatures import Timestamp, SampledSignal
from sha_learning.domain.sulfeatures import SystemUnderLearning, RealValuedVar, FlowCondition


labels: list[str] = list(metrics_to_labels.values())

def base4(n: int) -> str:
    """Convert n to a base-4 number"""
    digits: list[str] = [str(n // 4**i % 4) for i in range(n.bit_length() // 2, -1, -1) if n // 4**i]
    return ''.join(digits).zfill(4)

# Define events
events: list[Event] = []
# We always assume that the metrics have the same order as in `labels`
for i in range(4 ** len(labels) - 1):
    base4enc = base4(i+1)
    # Build a list of four 3-char strings with metric label (2 chars) + event label (1 char)
    ev_list: list[str] = [l+e for l, e in zip(labels, base4enc)]
    # Compress list into a single string and create event
    ev_strg = ''.join(ev_list)
    ev = Event('', ev_strg, ev_strg.lower())
    events.append(ev)
    print(i, base4enc, "->", ev)

# Define flow conditions
# TODO check
def vol_model(interval: list[Timestamp], V_0: float) -> list[float]:
    """Solution of the ODE modeling the variable"""
    return [V_0] * len(interval)
fc = FlowCondition(0, vol_model)

# define distributions
# (unused except in the thermostat example)
# TODO check
off_distr = NormalDistribution(0, 0.0, 0.0)

model2distr = {0: []}
vol = RealValuedVar([fc], [], model2distr, label='Vl')

# TODO check WTF are these
DRIVER_SIG = ['HR', 'Vl', 'RR', 'O2']
DEFAULT_M = 0  # default value of what?
DEFAULT_DISTR = 0  # default value of what?
args = {'name': 'energy', 'driver': DRIVER_SIG, 'default_m': DEFAULT_M, 'default_d': DEFAULT_DISTR}

safest_cs = SystemUnderLearning([vol], events, parse_data, label_event, get_vol_param, is_chg_pt, args=args)



test = True
if test:
    traces_folder = '/home/bruno/DEIB_Dropbox/safest/breathe_logs/processed_signals/'
    traces_files = os.listdir(traces_folder)
    traces_files.sort()
    for file in traces_files:
        print("Testing", file)
        # testing data to signals conversion
        new_signals: list[SampledSignal] = parse_data(traces_folder + file)
        # testing chg pts identification
        chg_pts = safest_cs.find_chg_pts([sig for sig in new_signals if sig.label in DRIVER_SIG])
        # testing event labeling
        id_events = [label_event(events, new_signals, pt.t) for pt in chg_pts]  # [:10]]
        # testing signal to trace conversion
        safest_cs.process_data(traces_folder + file)
        trace = safest_cs.timed_traces[-1]
        if file != '00.csv':  # which contains no events
            print('{}\t{}\t{}\t{}'.format(file, Trace(tt=trace),
                                          trace.t[-1].to_secs() - trace.t[0].to_secs(), len(trace)))

    unique_events: list[tuple[str, set[Event]]] = []
    for i, t_trace in enumerate(safest_cs.timed_traces):
        trace = Trace(tt=t_trace)
        unique_events.append((traces_files[i], set(trace.events)))
    unique_events.sort(key=lambda s: len(s[1]), reverse=True)
    print(unique_events)

    unique_events.sort(key=lambda s: s[0])
    print(unique_events)

    def fitness_function(ga_instance: GA, solution, solution_idx):
        output = [unique_events[i] for i in solution]
        output_set = set()
        for tt in output:
            output_set = output_set.union(tt[1])
        return len(output_set) * sum([len(tt[1]) for tt in output])

    CLUSTER_DIM = 9

    ga_instance = GA(num_generations=50,
                     num_parents_mating=4,
                     fitness_func=fitness_function,
                     sol_per_pop=8,
                     num_genes=CLUSTER_DIM,
                     gene_type=int,
                     init_range_low=0,
                     init_range_high=len(safest_cs.timed_traces) - 1,
                     parent_selection_type="sss",
                     keep_parents=1,
                     crossover_type="single_point",
                     mutation_type="random",
                     mutation_percent_genes=10)

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
