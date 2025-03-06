import os

from sha_learning.case_studies.safest.sul_functions import label_event, parse_data, get_vol_param, is_chg_pt
from sha_learning.case_studies.safest.sul_functions import signal_labels, trans_values
from sha_learning.domain.lshafeatures import Event, NormalDistribution, TimedTrace, Trace
from sha_learning.domain.sigfeatures import ChangePoint, Timestamp, SampledSignal
from sha_learning.domain.sulfeatures import SystemUnderLearning, RealValuedVar, FlowCondition, ProbDistribution
from sha_learning.learning_setup.teacher import Teacher


# Define events
events: list[Event] = []
for lab in signal_labels:
    for dest in trans_values:
        ev_strg = lab + str(dest)
        ev = Event('', ev_strg, ev_strg.lower())
        print(len(events), "->", ev)
        events.append(ev)
print()

# Define flow conditions
def vol_model(interval: list[Timestamp], V_0: float) -> list[float]:
    """Solution of the ODE modeling the variable"""
    return [V_0] * len(interval)
fc = FlowCondition(0, vol_model)

model2distr = {0: []}
vol = RealValuedVar([fc], [], model2distr, label='TV')

# Other args: CS name, list of event-determing signals, indexes of default model (flow condition)
# and default distribution when no events take place
args = {'name': 'safest', 'driver': signal_labels, 'default_m': 0, 'default_d': 0}

# Initialize SUL object
safest_cs = SystemUnderLearning([vol], events, parse_data, label_event, get_vol_param, is_chg_pt, args=args)


test = False
if test:
    traces_folder = '/home/bruno/DEIB_Dropbox/safest/lsha/breathe_logs/processed_signals/'
    traces_files = os.listdir(traces_folder)
    traces_files.sort()

    for file in traces_files:
        # Testing data to signals conversion
        new_signals: list[SampledSignal] = parse_data(traces_folder + file)
        # Testing chg pts identification
        chg_pts: list[ChangePoint] = safest_cs.find_chg_pts([sig for sig in new_signals if sig.label in signal_labels])
        # Testing event labeling
        id_events: list[Event] = [label_event(events, new_signals, pt.t) for pt in chg_pts]
        # Testing signal to trace conversion
        safest_cs.process_data(traces_folder + file)
        trace = safest_cs.timed_traces[-1]
        print(file)
        if file != 'SIM00':  # which contains no events
            print(Trace(tt=trace), "\n")
            # print('{}\t{}\t{}\t{}'.format(file, Trace(tt=trace),
            #                               trace.t[-1].to_secs() - trace.t[0].to_secs(), len(trace)))

    # Test segment identification
    test_trace = Trace(safest_cs.traces[0][:1])
    segments = safest_cs.get_segments(test_trace)

    # Test model identification
    teacher = Teacher(safest_cs, start_dt='2025-01-01-00-00-00', end_dt='2025-01-01-00-02-30')
    identified_model: FlowCondition = teacher.mi_query(test_trace)
    print(identified_model)

    # Test distr identification
    # Loop over traces from files
    for i, trace in enumerate(teacher.timed_traces):
        print("Trace", i)
        # Loop over events in the trace
        for j in range(len(trace.e)+1):
            # Consider trace i up to its jth event
            test_trace = Trace(safest_cs.traces[i][:j])
            # Hypothesis testing
            identified_distr: ProbDistribution = teacher.ht_query(test_trace, identified_model, save=True)
    
            segments = safest_cs.get_segments(test_trace)
            avg_metrics = sum([teacher.sul.get_ht_params(segment, identified_model)
                               for segment in segments]) / len(segments)
    
            try:
                print('{}:\t{:.3f}->{}'.format(test_trace.events[-1].symbol, avg_metrics, identified_distr.params))
            except IndexError:
                print('{}:\t{:.3f}->{}'.format(test_trace, avg_metrics, identified_distr.params))

        print()
