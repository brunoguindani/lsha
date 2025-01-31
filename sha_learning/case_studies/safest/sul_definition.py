import os

from sha_learning.case_studies.safest.sul_functions import label_event, parse_data, get_vol_param, is_chg_pt
from sha_learning.case_studies.safest.sul_functions import metrics_to_labels
from sha_learning.domain.lshafeatures import Event, NormalDistribution, TimedTrace, Trace
from sha_learning.domain.sigfeatures import ChangePoint, Timestamp, SampledSignal
from sha_learning.domain.sulfeatures import SystemUnderLearning, RealValuedVar, FlowCondition, ProbDistribution
from sha_learning.learning_setup.teacher import Teacher


signal_labels: list[str] = list(metrics_to_labels.values())
TERM_CHAR = '.'

def base4(n: int) -> str:
    """Convert n to a base-4 number"""
    digits: list[str] = [str(n // 4**i % 4) for i in range(n.bit_length() // 2, -1, -1) if n // 4**i]
    return ''.join(digits).zfill(4)

# Define events
events: list[Event] = []
# We always assume that the metrics have the same order as in `signal_labels`
for i in range(4 ** len(signal_labels) - 1):
    base4enc = base4(i+1)
    # Build a list of four or less 2-char strings, each with metric label (1 char) + event label (1 char)
    # Events with zero, i.e., unchanged metric, are not included
    ev_list: list[str] = [l+e if e != '0' else '' for l, e in zip(signal_labels, base4enc)]
    # Compress list into a single string (adding a termination char) and create event
    ev_strg = ''.join(ev_list) + TERM_CHAR
    ev = Event('', ev_strg, ev_strg.lower())
    events.append(ev)
    # print(i, base4enc, "->", ev)

# Define flow conditions
def vol_model(interval: list[Timestamp], V_0: float) -> list[float]:
    """Solution of the ODE modeling the variable"""
    return [V_0] * len(interval)
fc = FlowCondition(0, vol_model)

model2distr = {0: []}
vol = RealValuedVar([fc], [], model2distr, label='V')

# Other args: CS name, list of event-determing signals, indexes of default model (flow condition)
# and default distribution when no events take place
args = {'name': 'safest', 'driver': signal_labels, 'default_m': 0, 'default_d': 0}

# Initialize SUL object
safest_cs = SystemUnderLearning([vol], events, parse_data, label_event, get_vol_param, is_chg_pt, args=args)


test = False
if test:
    traces_folder = '/home/bruno/DEIB_Dropbox/safest/breathe_logs/processed_signals/'
    traces_files = os.listdir(traces_folder)
    traces_files.sort()

    teacher = Teacher(safest_cs, start_dt='2025-01-01-00-00-00', end_dt='2025-01-01-00-01-00')

    for file in traces_files:
        if file == '00.csv':  # which contains no events
            continue

        # Testing data to signals conversion
        new_signals: list[SampledSignal] = parse_data(traces_folder + file)
        # Testing chg pts identification
        chg_pts: list[ChangePoint] = safest_cs.find_chg_pts([sig for sig in new_signals if sig.label in signal_labels])
        # Testing event labeling
        id_events: list[Event] = [label_event(events, new_signals, pt.t) for pt in chg_pts]
        # Testing signal to trace conversion
        safest_cs.process_data(traces_folder + file)
        t_trace: TimedTrace = safest_cs.timed_traces[-1]
        trace = Trace(tt=t_trace)
        # Testing model identification and hypothesis testing queries
        model: FlowCondition = teacher.mi_query(trace)
        distr: ProbDistribution = teacher.ht_query(trace, model)
        print('\t'.join([str(_) for _ in [file, trace, t_trace.t[-1].to_secs() - t_trace.t[0].to_secs(),
                                          len(trace), model, distr]]))
