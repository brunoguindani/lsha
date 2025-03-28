import numpy as np

from commons import *

max_traces = 50
num_events = 20
base_seed = int(sys.argv[1])
log_file = 'traces.csv'

rng = np.random.default_rng(base_seed)
shutil.rmtree(root_folder)

current_traces = 0
while current_traces < max_traces:

  # Add `batch_size` new signals
  for i in range(batch_size):
    base_name = 'SIM' + str(current_traces).zfill(4)
    folder_path = os.path.join(root_folder, base_name)
    os.makedirs(folder_path, exist_ok=True)
    output_path = os.path.join(folder_path, base_name + '.csv')
    seed = rng.integers(20250000)
    generate_random_signal(num_events, event_delta, row_delta, seed, output_path)
    current_traces += 1

  # Start LSHA with all current traces
  sha_name = f'safest_s{base_seed}_t{current_traces}_e{num_events}'
  sys.argv = ['learn_model.py', sha_name]
  start = time.time()
  mem_usage = memory_usage((runpy.run_module, ('sha_learning.learn_model',), {'run_name': '__main__'}))
  t = time.time() - start
  max_usage = max(mem_usage)
  log_path = os.path.join('sha_learning', 'resources', 'learned_sha', 'scale_tests', sha_name+'.log')
  loc, trans = count_locations_and_transitions(log_path)
  write_to_log(log_file, base_seed, current_traces, num_events, t, max_usage, loc, trans)
