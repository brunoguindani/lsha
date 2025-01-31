import configparser
import os
import pandas as pd
from sha_learning.domain.lshafeatures import Event, FlowCondition
from sha_learning.domain.sigfeatures import SampledSignal, Timestamp, SignalPoint
from sha_learning.learning_setup.logger import Logger


# We will always assume that the metrics (and lists of values) have the same order as in here
metrics_to_labels = {'HeartRate' : 'H', 'TotalLungVolume' : 'V', 'RespirationRate' : 'R', 'OxygenSaturation': 'O'}

metrics_to_low_high_values = {'H': (70, 80), 'V': (1950, 2100),
                             'R': (10, 13), 'O': (0.965, 0.976)}


def transform_val(x: float, low: float, high: float) -> int:
    if x < low:
        return 1
    elif x > high:
        return 3
    else:
        return 2


def is_chg_pt(curr: list[float], prev: list[float]) -> bool:
    for metric, c, p in zip(metrics_to_labels.values(), curr, prev):
        low, high = metrics_to_low_high_values[metric]
        c_lab = transform_val(c, low, high)
        p_lab = transform_val(p, low, high)
        if c_lab != p_lab:
            return True
    return False


def label_event(events: list[Event], signals: list[SampledSignal], t: Timestamp) -> Event:
    # Find index of timestamp t in the signals. This assumes that the signals all contain the same timestamps
    for idx_t in range(1, len(signals[0].points)):
        if signals[0].points[idx_t].timestamp == t:
            break

    # Collect signals at current and previous timestamp
    signals_t = []
    signals_tm1 = []
    for s in signals:
        low, high = metrics_to_low_high_values[s.label]
        t_val = transform_val(s.points[idx_t].value, low=low, high=high)
        signals_t.append(t_val)
        tm1_val = transform_val(s.points[idx_t-1].value, low=low, high=high)
        signals_tm1.append(tm1_val)
    # print("curr:", signals_t, "prev:", signals_tm1, "at", t)

    # Build base-4 encoding of event. Output is a 4-char string (plus termination char) where
    # 0 means unchanged, 1, 2, 3 means that curr has changed to that value
    ev_strg = ''
    for curr, prev in zip(signals_t, signals_tm1):
        ev_strg += '0' if curr == prev else str(curr)

    ev_idx = int(ev_strg, 4) - 1
    identified_event = events[ev_idx]
    # print(ev_strg, "-> idx", ev_idx, "->", identified_event, "\n")

    return identified_event


def parse_data(path: str) -> list[SampledSignal]:
    df = pd.read_csv(path, index_col='SimTime')

    signals = {key: SampledSignal([], label=lab) for key, lab in metrics_to_labels.items()}

    for t, row in df.iterrows():
        for col in df.columns:
            ts = Timestamp(y=2025, m=1, d=1, h=0, min=int(t) // 60, sec=round(t % 60, 2))
            signals[col].points.append(SignalPoint(ts, row[col]))

    return list(signals.values())


def get_vol_param(segment: list[SignalPoint], flow: FlowCondition):
    sum_vol = sum([pt.value for pt in segment])
    avg_vol = sum_vol / len(segment)
    return avg_vol
