import os

from sha_learning.case_studies.safest.sul_functions import label_event, parse_data, get_vol_param, is_chg_pt
from sha_learning.case_studies.safest.sul_functions import patient_metrics, signal_labels, trans_values
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
# Add ventilators ON and OFF events
events.append(Event('', 'on.', 'on.'))
print(len(events)-1, "->", events[-1])
events.append(Event('', 'off.', 'off.'))
print(len(events)-1, "->", events[-1])
print()

# Environment-controlled events on patient
env_events = events[0:15]

# Define flow conditions
def vol_model(interval: list[Timestamp], V_0: float) -> list[float]:
    """Solution of the ODE modeling the variable"""
    return [V_0] * len(interval)
fc = FlowCondition(0, vol_model)

model2distr = {0: []}
vol = RealValuedVar([fc], [], model2distr, label='tv')

# Other args: CS name, list of event-determing signals, indexes of default model (flow condition)
# and default distribution when no events take place
args = {'name': 'safest', 'driver': signal_labels, 'default_m': 0, 'default_d': 0}

# Initialize SUL object
safest_cs = SystemUnderLearning([vol], events, parse_data, label_event, get_vol_param, is_chg_pt, args=args)


test = False
if test:
    root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, 'breathe_logs'))
    traces_folder = os.path.join(root_folder, 'processed_signals', 'accuracy')
    traces_files = os.listdir(traces_folder)
    traces_files.sort()
    env_traces_folder = os.path.join(root_folder, 'environment_traces', 'accuracy')
    os.makedirs(env_traces_folder, exist_ok=True)

    def get_timestamp_index(signal: SampledSignal, ts: Timestamp):
      t = 60*ts.min + ts.sec
      for p in signal.points:
        t_p = 60*p.timestamp.min + p.timestamp.sec
        if abs(t_p - t) < 0.01:
          return signal.points.index(p)

    for file in traces_files:
        file_path = os.path.join(traces_folder, file)
        # Testing data to signals conversion
        new_signals: list[SampledSignal] = parse_data(file_path)
        # Testing chg pts identification
        chg_pts: list[ChangePoint] = safest_cs.find_chg_pts([sig for sig in new_signals if sig.label in signal_labels])
        # Testing event labeling
        id_events: list[Event] = [label_event(events, new_signals, pt.t) for pt in chg_pts]
        # Testing signal to trace conversion
        safest_cs.process_data(file_path)
        timed_trace = safest_cs.timed_traces[-1]
        trace = Trace(tt=timed_trace)

        ## Identification of last event and tv value
        tv_signal = [s for s in new_signals if s.label == 'tv'][0]
        last_change_ts = timed_trace.t[-1]

        last_point_idx = get_timestamp_index(tv_signal, last_change_ts)
        # print(file)
        # print("t-1:", dict(zip(signal_labels, [round(s.points[last_point_idx-1].value, 2) for s in new_signals])))
        # print("t  :", dict(zip(signal_labels, [round(s.points[last_point_idx].value, 2) for s in new_signals])))
        print(file, ",", int(tv_signal.points[last_point_idx].value), sep="")

        # print(trace, "\n")
        print(file, '"' + ','.join([e.symbol for e in trace.events]) + '"', int(tv_signal.points[last_point_idx].value), sep=",")
        # Write environment trace in separate file
        env_trace_file = os.path.join(env_traces_folder, file + '.csv')
        with open(env_trace_file, 'w') as f:
          f.write('time,event\n')
          for i in range(len(timed_trace)):
            event = timed_trace.e[i]
            dt = timed_trace.t[i]
            secs = 60 * dt.min + dt.sec
            if event in env_events:
              f.write(f'{secs},{event}\n')

    exit()
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
