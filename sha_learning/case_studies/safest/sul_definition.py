import os

from sha_learning.case_studies.safest.sul_functions import label_event, parse_data, get_vol_param, is_chg_pt
from sha_learning.case_studies.safest.sul_functions import metrics_to_labels
from sha_learning.domain.lshafeatures import Event, NormalDistribution, Trace
from sha_learning.domain.sigfeatures import Timestamp, SampledSignal
from sha_learning.domain.sulfeatures import SystemUnderLearning, RealValuedVar, FlowCondition


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

# indexes of default model (flow condition) and distribution when no events take place
DEFAULT_MODEL = 0
DEFAULT_DISTR = 0

args = {'name': 'energy', 'driver': signal_labels, 'default_m': DEFAULT_MODEL, 'default_d': DEFAULT_DISTR}

safest_cs = SystemUnderLearning([vol], events, parse_data, label_event, get_vol_param, is_chg_pt, args=args)



test = False
if test:
    traces_folder = '/home/bruno/DEIB_Dropbox/safest/breathe_logs/processed_signals/'
    traces_files = os.listdir(traces_folder)
    traces_files.sort()
    for file in traces_files:
        print("Testing", file)
        # testing data to signals conversion
        new_signals: list[SampledSignal] = parse_data(traces_folder + file)
        # testing chg pts identification
        chg_pts = safest_cs.find_chg_pts([sig for sig in new_signals if sig.label in signal_labels])
        # testing event labeling
        id_events = [label_event(events, new_signals, pt.t) for pt in chg_pts]  # [:10]]
        # testing signal to trace conversion
        safest_cs.process_data(traces_folder + file)
        trace = safest_cs.timed_traces[-1]
        if file != '00.csv':  # which contains no events
            print('{}\t{}\t{}\t{}'.format(file, Trace(tt=trace),
                                          trace.t[-1].to_secs() - trace.t[0].to_secs(), len(trace)))
