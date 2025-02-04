import configparser
import os
import sys
import warnings
from datetime import datetime

import sha_learning.pltr.lsha_report as report
import sha_learning.pltr.sha_pltr as ha_pltr
from sha_learning.case_studies.safest.sul_definition import safest_cs
from sha_learning.domain.lshafeatures import Trace
from sha_learning.domain.obstable import ObsTable
from sha_learning.domain.sulfeatures import SystemUnderLearning
from sha_learning.learning_setup.learner import Learner
from sha_learning.learning_setup.logger import Logger
from sha_learning.learning_setup.teacher import Teacher
from sha_learning.pltr.energy_pltr import distr_hist

# LEARNING PROCEDURE SETUP
warnings.filterwarnings('ignore')
startTime = datetime.now()

config = configparser.ConfigParser()
config.sections()
config.read(
    os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + 'sha_learning/resources/config/config.ini')
config.sections()

CS = 'safest'
LOGGER = Logger('LSHA')


SUL: SystemUnderLearning = safest_cs
events_labels_dict = None
TEACHER = Teacher(SUL, start_dt='2025-01-01-00-00-00', end_dt='2025-01-01-00-01-00')
# start_dt, end_dt = start and end datetimes in string form
# unused int args start_ts, end_ts = start and end times in seconds

long_traces = [Trace(events=[e]) for e in SUL.events]
obs_table = ObsTable([], [Trace(events=[])], long_traces)
LEARNER = Learner(TEACHER, obs_table)

# RUN LEARNING ALGORITHM:
LEARNED_HA = LEARNER.run_lsha(filter_empty=True)

# PLOT (AND SAVE) RESULT
HA_SAVE_PATH = config['SUL CONFIGURATION']['SHA_SAVE_PATH'].format(os.getcwd())

SHA_NAME = '{}_{}'.format(CS, config['SUL CONFIGURATION']['CS_VERSION'])
graphviz_sha = ha_pltr.to_graphviz(LEARNED_HA, SHA_NAME, HA_SAVE_PATH, view=True)

# saving sha source to .txt file
sha_source = graphviz_sha.source
with open(HA_SAVE_PATH.format(os.getcwd()) + SHA_NAME + '_source.txt', 'w') as f:
    f.write(sha_source)

if config['DEFAULT']['PLOT_DISTR'] == 'True' and config['LSHA PARAMETERS']['HT_QUERY_TYPE'] == 'S':
    distr_hist(TEACHER.hist, SHA_NAME)

report.save_data(TEACHER.symbols, TEACHER.distributions, LEARNER.obs_table,
                 len(TEACHER.signals), datetime.now() - startTime, SHA_NAME, events_labels_dict,
                 os.getcwd())
LOGGER.info('----> EXPERIMENTAL RESULTS SAVED IN: {}{}.txt'.format(
    config['SUL CONFIGURATION']['REPORT_SAVE_PATH'].format(os.getcwd()), SHA_NAME)
)
