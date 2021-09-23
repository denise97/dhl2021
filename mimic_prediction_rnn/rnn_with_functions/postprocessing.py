from io_helpers import *

import numpy as np
import os
import pandas as pd


def fill_alarm_triggering_col(chunk, alarm_type, pred, configs):
    if pred:
        suffix = '_PREDICTION'
        if configs.with_exogenous_input:
            col_to_check = f'VALUE_PREDICTION_{alarm_type.upper()}'
        else:
            col_to_check = 'VALUE_PREDICTION'
    else:
        suffix = ''
        if alarm_type == 'Low':
            col_to_check = 'VITAL_PARAMTER_VALUE_MIN_RESAMPLING'
        else:
            col_to_check = 'VITAL_PARAMTER_VALUE_MAX_RESAMPLING'

    chunk[f'{alarm_type.upper()}_ALARM_TRIGGERED{suffix}'] = False

    if alarm_type == 'Low':
        chunk.loc[chunk[col_to_check]
                  < chunk[f'THRESHOLD_VALUE_{alarm_type.upper()}'],
                  f'{alarm_type.upper()}_ALARM_TRIGGERED{suffix}'] = True
    else:
        chunk.loc[chunk[col_to_check]
                  > chunk[f'THRESHOLD_VALUE_{alarm_type.upper()}'],
                  f'{alarm_type.upper()}_ALARM_TRIGGERED{suffix}'] = True

    return chunk[f'{alarm_type.upper()}_ALARM_TRIGGERED{suffix}']


def fill_alarm_triggering_cols(chunk, final_pred, configs):

    # Add boolean column indicating triggered alarm for original value
    fill_alarm_triggering_col(chunk, 'Low', False, configs)
    fill_alarm_triggering_col(chunk, 'High', False, configs)

    # Add columns with predicted values to chunk
    if configs.with_exogenous_input:
        chunk['VALUE_PREDICTION_LOW'] = final_pred[0].Value
        chunk['VALUE_PREDICTION_HIGH'] = final_pred[1].Value
    else:
        chunk['VALUE_PREDICTION'] = final_pred[0].Value

    # Add boolean indicating triggered alarm for predicted value
    fill_alarm_triggering_col(chunk, 'Low', True, configs)
    fill_alarm_triggering_col(chunk, 'High', True, configs)


def get_alarm_triggering_indices(chunk, triggered, alarm_type, pred):
    if pred:
        suffix = '_PREDICTION'
    else:
        suffix = ''

    col_name = f'{alarm_type.upper()}_ALARM_TRIGGERED{suffix}'

    if triggered:
        return set(chunk.index[chunk[col_name]])
    else:
        return set(chunk.index[~chunk[col_name]])


def get_alarm_triggering_indices_for_alarm_type(chunk, alarm_type):
    return get_alarm_triggering_indices(chunk, True, alarm_type, False), \
           get_alarm_triggering_indices(chunk, True, alarm_type, True), \
           get_alarm_triggering_indices(chunk, False, alarm_type, False), \
           get_alarm_triggering_indices(chunk, False, alarm_type, True)


def get_chunk_matrices_of_all_windows(model_type, parameter, configs):
    confusion_matrix_chunks_concat = pd.DataFrame(
        columns=['CHUNK_ID', 'PARAMETER', 'MODEL', 'ENDOGENOUS', 'EXOGENOUS', 'FIRST_FORECAST', 'ALARM_TYPE', 'FP',
                 'TP', 'FN', 'TN', 'N_HIGH_ALARMS', 'N_LOW_ALARMS', 'N_ITERATIONS'])

    script_path = get_script_path(configs)

    for file in os.listdir(f'{script_path}/confusion_matrices/'):
        if os.path.isfile(os.path.join(f'{script_path}/confusion_matrices/', file)) and \
                file.startswith(f'confusion_matrix_chunks_{model_type}_{parameter}') and \
                file.endswith(f'.pickle'):
            current_chunk_matrix = read_pickle_file(f'{script_path}/confusion_matrices/{file}')

            confusion_matrix_chunks_concat = pd.concat([confusion_matrix_chunks_concat, current_chunk_matrix])

    confusion_matrix_chunks_concat.reset_index(inplace=True, drop=True)

    return confusion_matrix_chunks_concat


def get_chunk_matrices_for_alarm_type(model_type, parameter, alarm_type, configs):
    chunk_matrices = get_chunk_matrices_of_all_windows(model_type, parameter, configs)
    return chunk_matrices[chunk_matrices['ALARM_TYPE'] == alarm_type]
