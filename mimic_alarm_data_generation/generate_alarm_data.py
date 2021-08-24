"""
    This is an adapted version of alarm_table_generation.ipynb by Jonas Chromik
    (https://gitlab.hpi.de/jonas.chromik/mimic-alarms) that generates a CSV containing the alarm data.

    To execute this script, a parquet file containing the cleaned CHARTEVENTS table must be available as the input.

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
    value_series = stayevents[stayevents.ITEMID == PARAMS.VALUE[param]]
    high_thresholds = stayevents[stayevents.ITEMID == PARAMS.THRESHOLD_HIGH[param]]
    low_thresholds = stayevents[stayevents.ITEMID == PARAMS.THRESHOLD_LOW[param]]

    alarms = pd.DataFrame([])

    # Check if there is a value that is lower than previously defined threshold of type low
    for old, new in zip_longest(low_thresholds.itertuples(), low_thresholds[1:].itertuples(), fillvalue=None):
        if new is not None:
            local_alarms = value_series[
                (value_series.CHARTTIME >= old.CHARTTIME) &
                (value_series.CHARTTIME < new.CHARTTIME) &
                (value_series.VALUENUM_CLEAN < old.VALUENUM_CLEAN)]
        else:
            local_alarms = value_series[
                (value_series.CHARTTIME >= old.CHARTTIME) &
                (value_series.VALUENUM_CLEAN < old.VALUENUM_CLEAN)]

        local_alarms = local_alarms[COLS]

        local_alarms.insert(6, 'THRESHOLD_ROW_ID', old.ROW_ID)
        local_alarms.insert(7, 'THRESHOLD_ITEMID', old.ITEMID)
        local_alarms.insert(8, 'THRESHOLD_CHARTTIME', old.CHARTTIME)
        local_alarms.insert(9, 'THRESHOLD_VALUE', old.VALUENUM_CLEAN)
        local_alarms.insert(10, 'THRESHOLD_TYPE', 'LOW')

        alarms = alarms.append(local_alarms)

    # Check if there is a value that is higher than previously defined threshold of type high
    for old, new in zip_longest(high_thresholds.itertuples(), high_thresholds[1:].itertuples(), fillvalue=None):
        if new is not None:
            local_alarms = value_series[
                (value_series.CHARTTIME >= old.CHARTTIME) &
                (value_series.CHARTTIME < new.CHARTTIME) &
                (value_series.VALUENUM_CLEAN > old.VALUENUM_CLEAN)]
        else:
            local_alarms = value_series[
                (value_series.CHARTTIME >= old.CHARTTIME) &
                (value_series.VALUENUM_CLEAN > old.VALUENUM_CLEAN)]

        local_alarms = local_alarms[COLS]

        local_alarms.insert(6, 'THRESHOLD_ROW_ID', old.ROW_ID)
        local_alarms.insert(7, 'THRESHOLD_ITEMID', old.ITEMID)
        local_alarms.insert(8, 'THRESHOLD_CHARTTIME', old.CHARTTIME)
        local_alarms.insert(9, 'THRESHOLD_VALUE', old.VALUENUM_CLEAN)
        local_alarms.insert(10, 'THRESHOLD_TYPE', 'HIGH')

        alarms = alarms.append(local_alarms)

    return alarms


if __name__ == "__main__":
    # Add custom help message and script parameters for in- and output files
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument("-c", "--chartevents-path",
                        required=True,
                        help="path to cleaned CHARTEVENTS parquet file (input)")
    parser.add_argument("-a", "--alarm-data-path",
                        required=True,
                        help="path to folder where alarm data CSV will be stored (output)")
    args = parser.parse_args()

    # Define input path and filename of resulting CSV
    CHARTEVENTS_PATH = args.chartevents_path
    ALARM_DATA_PATH = args.alarm_data_path + '/alarm_data.csv'

    # Define columns to include into alarm data generation
    COLS = ['ROW_ID', 'ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM_CLEAN', 'VALUEUOM']

    # Define parameter properties
    PARAMS = pd.DataFrame({
        'LABEL':            ['HR',      'NBPs',     'SpO2'],
        'VALUE':            [220045,    220179,     220277],
        'THRESHOLD_HIGH':   [220046,    223751,     223769],
        'THRESHOLD_LOW':    [220047,    223752,     223770]})

    # Read and prepare cleaned chart events
    chartevents_clean = pd.read_parquet(CHARTEVENTS_PATH, engine='pyarrow')
    chartevents_clean.CHARTTIME = pd.to_datetime(chartevents_clean.CHARTTIME)
    chartevents_clean = chartevents_clean.sort_values('CHARTTIME')
    chartevents_clean = chartevents_clean[chartevents_clean['VALUENUM_CLEAN'].notna()]

    # Get unique ICU stays
    unique_icu_stays = pd.unique(chartevents_clean.ICUSTAY_ID)

    # Create alarm data
    alarms = pd.DataFrame([])
    with Pool(cpu_count()) as p:
        alarms = pd.concat(tqdm(p.imap(alarms_for, unique_icu_stays), total=len(unique_icu_stays), smoothing=0))

    # Write CSV
    alarms.to_csv(ALARM_DATA_PATH, index=False)
