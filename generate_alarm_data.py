"""
    This is an adapted version of alarm_table_generation.ipynb by Jonas Chromik
    (https://gitlab.hpi.de/jonas.chromik/mimic-alarms).

    To execute this script on server, chartevents_clean.parquet respectively
    chartevents_clean_values_and_thresholds_with_chunkid_65 and unique_icustays_in_chartevents_subset.parquet
    have to be in the subdirectory "/data".

    Afterwards, you can run the following command to install all needed modules:
        pip3 install pandas pyarrow tqdm

    Finally, you can execute the script:
        python3 generate_alarm_data.py
"""


import pandas as pd
from argparse import ArgumentParser, RawTextHelpFormatter
from itertools import zip_longest
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def alarms_for(icustay_id):
    stayevents = chartevents_clean[chartevents_clean.ICUSTAY_ID == icustay_id]
    return pd.concat([_alarms(stayevents, param) for param in PARAMS.index])


def _alarms(stayevents, param):
    value_events = stayevents[stayevents.ITEMID == PARAMS.VALUE[param]]
    high_events = stayevents[stayevents.ITEMID == PARAMS.THRESHOLD_HIGH[param]]
    low_events = stayevents[stayevents.ITEMID == PARAMS.THRESHOLD_LOW[param]]

    alarms = pd.DataFrame([])

    for old, new in zip_longest(low_events.itertuples(), low_events[1:].itertuples(), fillvalue=None):
        if new is not None:
            local_alarms = value_events[
                (value_events.CHARTTIME >= old.CHARTTIME) &
                (value_events.CHARTTIME < new.CHARTTIME) &
                (value_events.VALUENUM < old.VALUENUM)]
        else:
            local_alarms = value_events[
                (value_events.CHARTTIME >= old.CHARTTIME) &
                (value_events.VALUENUM < old.VALUENUM)]

        local_alarms = local_alarms[COLS]

        local_alarms.insert(6, 'THRESHOLD_ROW_ID', old.ROW_ID)
        local_alarms.insert(7, 'THRESHOLD_ITEMID', old.ITEMID)
        local_alarms.insert(8, 'THRESHOLD_CHARTTIME', old.CHARTTIME)
        local_alarms.insert(9, 'THRESHOLD_VALUE', old.VALUENUM)
        local_alarms.insert(10, 'THRESHOLD_TYPE', 'LOW')

        alarms = alarms.append(local_alarms)

    for old, new in zip_longest(high_events.itertuples(), high_events[1:].itertuples(), fillvalue=None):
        if new is not None:
            local_alarms = value_events[
                (value_events.CHARTTIME >= old.CHARTTIME) &
                (value_events.CHARTTIME < new.CHARTTIME) &
                (value_events.VALUENUM > old.VALUENUM)]
        else:
            local_alarms = value_events[
                (value_events.CHARTTIME >= old.CHARTTIME) &
                (value_events.VALUENUM > old.VALUENUM)]

        local_alarms = local_alarms[COLS]

        local_alarms.insert(6, 'THRESHOLD_ROW_ID', old.ROW_ID)
        local_alarms.insert(7, 'THRESHOLD_ITEMID', old.ITEMID)
        local_alarms.insert(8, 'THRESHOLD_CHARTTIME', old.CHARTTIME)
        local_alarms.insert(9, 'THRESHOLD_VALUE', old.VALUENUM)
        local_alarms.insert(10, 'THRESHOLD_TYPE', 'HIGH')

        alarms = alarms.append(local_alarms)

    return alarms


if __name__ == "__main__":
    # Add custom help message and optional script parameter "--chunks"
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument("--chunks", action="store_true", help="execute with chunked version of cleaned chartevents")
    args = parser.parse_args()

    # Define input path and filename of resulting CSV, depending on script parameter "--chunks"
    if args.chunks:
        PATH = './data/chartevents_clean_values_and_thresholds_with_chunkid_65.parquet'
        FILENAME = './data/alarm_data_with_chunks_65.csv'
    else:
        PATH = './data/chartevents_clean.parquet'
        FILENAME = './data/alarm_data.csv'

    # Define columns to include into alarm data generation
    COLS = ['ROW_ID', 'ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM_CLEAN', 'VALUEUOM']

    # Define parameter properties
    PARAMS = pd.DataFrame({
        'LABEL':            ['HR',      'NBPs',     'SpO2'],
        'VALUE':            [220045,    220179,     220277],
        'THRESHOLD_HIGH':   [220046,    223751,     223769],
        'THRESHOLD_LOW':    [220047,    223752,     223770]})

    # Read and prepare cleaned chart events
    chartevents_clean = pd.read_parquet(PATH, engine='pyarrow')
    chartevents_clean.CHARTTIME = pd.to_datetime(chartevents_clean.CHARTTIME)
    chartevents_clean = chartevents_clean[chartevents_clean['VALUENUM_CLEAN'].notna()]

    # Read unique ICU stays
    icustays = pd.read_parquet('./data/unique_icustays_in_chartevents_subset.parquet', engine='pyarrow')

    # Create alarm data
    alarms = pd.DataFrame([])
    with Pool(cpu_count()) as p:
        alarms = pd.concat(tqdm(p.imap(alarms_for, icustays.ICUSTAY_ID), total=len(icustays), smoothing=0))

    # Convert row and item IDs to integers
    for col in ['ROW_ID', 'ITEMID', 'THRESHOLD_ROW_ID', 'THRESHOLD_ITEMID']:
        alarms[col] = alarms[col].astype(int)

    # Write CSV
    alarms.to_csv(FILENAME, index=False)
