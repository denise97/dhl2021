#
# Adapted alarm_table_generation.ipynb of Jonas Chromik (https://gitlab.hpi.de/jonas.chromik/mimic-alarms)
#
# To execute this script on server, chartevents_clean.parquet and unique_icustays_in_chartevents_subset.parquet
# have to be in the subdirectory "/data".
#
# Afterwards, you can run the following command to install all needed modules:
#
# pip3 install pandas pyarrow tqdm
#
# Finally, you can execute the script:
#
# python3 generate_alarm_data.py
#

import pandas as pd
from itertools import zip_longest
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


PATH = './data/chartevents_clean.parquet'
COLS = ['ROW_ID', 'ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM_CLEAN', 'VALUEUOM']
PARAMS = pd.DataFrame({
    'LABEL':            ['HR',      'NBPs',     'SpO2'],
    'VALUE':            [220045,    220179,     220277],
    'THRESHOLD_HIGH':   [220046,    223751,     223769],
    'THRESHOLD_LOW':    [220047,    223752,     223770]})

chartevents_clean = pd.read_parquet(PATH, engine='pyarrow')
chartevents_clean.CHARTTIME = pd.to_datetime(chartevents_clean.CHARTTIME)
chartevents_clean = chartevents_clean[chartevents_clean['VALUENUM_CLEAN'].notna()]

icustays = pd.read_parquet('./data/unique_icustays_in_chartevents_subset.parquet', engine='pyarrow')
violations = pd.DataFrame([])


def violations_for(icustay_id):
    stayevents = chartevents_clean[chartevents_clean.ICUSTAY_ID == icustay_id]
    return pd.concat([_violations(stayevents, param) for param in PARAMS.index])


def _violations(stayevents, param):
    value_events = stayevents[stayevents.ITEMID == PARAMS.VALUE[param]]
    high_events = stayevents[stayevents.ITEMID == PARAMS.THRESHOLD_HIGH[param]]
    low_events = stayevents[stayevents.ITEMID == PARAMS.THRESHOLD_LOW[param]]

    violations = pd.DataFrame([])

    for old, new in zip_longest(low_events.itertuples(), low_events[1:].itertuples(), fillvalue=None):
        if new is not None:
            local_violations = value_events[
                (value_events.CHARTTIME >= old.CHARTTIME) &
                (value_events.CHARTTIME < new.CHARTTIME) &
                (value_events.VALUENUM < old.VALUENUM)]
        else:
            local_violations = value_events[
                (value_events.CHARTTIME >= old.CHARTTIME) &
                (value_events.VALUENUM < old.VALUENUM)]

        local_violations = local_violations[COLS]

        local_violations.insert(6, 'THRESHOLD_ROW_ID', old.ROW_ID)
        local_violations.insert(7, 'THRESHOLD_ITEMID', old.ITEMID)
        local_violations.insert(8, 'THRESHOLD_CHARTTIME', old.CHARTTIME)
        local_violations.insert(9, 'THRESHOLD_VALUE', old.VALUENUM)
        local_violations.insert(10, 'THRESHOLD_TYPE', 'LOW')

        violations = violations.append(local_violations)

    for old, new in zip_longest(high_events.itertuples(), high_events[1:].itertuples(), fillvalue=None):
        if new is not None:
            local_violations = value_events[
                (value_events.CHARTTIME >= old.CHARTTIME) &
                (value_events.CHARTTIME < new.CHARTTIME) &
                (value_events.VALUENUM > old.VALUENUM)]
        else:
            local_violations = value_events[
                (value_events.CHARTTIME >= old.CHARTTIME) &
                (value_events.VALUENUM > old.VALUENUM)]

        local_violations = local_violations[COLS]

        local_violations.insert(6, 'THRESHOLD_ROW_ID', old.ROW_ID)
        local_violations.insert(7, 'THRESHOLD_ITEMID', old.ITEMID)
        local_violations.insert(8, 'THRESHOLD_CHARTTIME', old.CHARTTIME)
        local_violations.insert(9, 'THRESHOLD_VALUE', old.VALUENUM)
        local_violations.insert(10, 'THRESHOLD_TYPE', 'HIGH')

        violations = violations.append(local_violations)

    return violations


with Pool(cpu_count()) as p:
    violations = pd.concat(tqdm(p.imap(violations_for, icustays.ICUSTAY_ID), total=len(icustays), smoothing=0))

violations['ROW_ID'] = violations['ROW_ID'].astype(int)
violations['ITEMID'] = violations['ITEMID'].astype(int)
violations['THRESHOLD_ROW_ID'] = violations['THRESHOLD_ROW_ID'].astype(int)
violations['THRESHOLD_ITEMID'] = violations['THRESHOLD_ITEMID'].astype(int)

violations.to_csv('./data/alarm_data.csv', index=False)
