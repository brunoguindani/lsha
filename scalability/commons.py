from memory_profiler import memory_usage
import os
import runpy
import shutil
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'breathe_logs'))
from breathe_logs.generate_random import generate_random_signal

output_csv_folder = os.path.join(os.path.dirname(__file__))
batch_size = 5
event_delta = 30.0
row_delta = 1.0

os.environ['RES_PATH'] = '/home/bruno/DEIB_Dropbox/safest/lsha/sha_learning/resources'
root_folder = os.path.join('breathe_logs', 'generated')

def write_to_log(log_name, *args):
  file = os.path.join(output_csv_folder, log_name)
  with open(file, 'a') as f:
    f.write(','.join([str(_) for _ in args]) + '\n')

def count_locations_and_transitions(log_path: str) -> tuple[int, int]:
  with open(log_path, 'r') as f:
    content = f.read()
  locations = content.count(':')
  transitions = content.count('->')
  return locations, transitions
