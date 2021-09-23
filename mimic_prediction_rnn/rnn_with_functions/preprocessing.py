from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from io_helpers import *

import numpy as np
import pandas as pd
import statistics


# Turn off SettingWithCopyWarning
pd.options.mode.chained_assignment = None


class Configurations:
    def __init__(self, configs):
        self.model_types = configs.model_types
        self.parameters = configs.parameters
        self.input_length = configs.input_length
        self.output_length = configs.output_length
        self.prediction_data_size = configs.prediction_data_size
        self.n_chunks = configs.n_chunks
        self.n_windows = configs.n_windows
        self.with_exogenous_input = configs.with_exogenous_input
        self.scaling_method = configs.scaling_method


def extract_relevant_chunk_ids(chunks, configs, parameter):
    relevant_chunk_ids = list()

    for chunk_id in pd.unique(chunks.CHUNK_ID_FILLED_TH):
        current_chunk = chunks[chunks['CHUNK_ID_FILLED_TH'] == chunk_id]

        # At least input_chunk_length + output_chunk_length, e.g. 12 + 1 = 13, data points are required
        if len(current_chunk) >= (configs.input_length + configs.output_length):
            relevant_chunk_ids.append(chunk_id)

    path = get_script_path(configs)
    write_pickle_file(f'{path}/chunk_ids/relevant_chunk_ids_{parameter}.pickle', relevant_chunk_ids)
    return relevant_chunk_ids


def extract_prediction_chunk_ids(window_idx, relevant_chunk_ids, configs, parameter):
    # Calculate number of chunks corresponding to prediction dataset size percentage, e.g. 20%
    percentage = int((configs.prediction_data_size * len(relevant_chunk_ids)) / 100)

    # Extract certain percentage, e.g. 20%, of chunk IDs for prediction (and catch last window to avoid ignoring chunks)
    if window_idx == (configs.n_windows - 1):
        pred_chunk_ids = [chunk_id for chunk_id in relevant_chunk_ids[percentage * window_idx:]]
    else:
        pred_chunk_ids = [chunk_id for chunk_id in relevant_chunk_ids[percentage * window_idx:
                                                                      percentage * (window_idx + 1)]]

    path = get_script_path(configs)
    write_pickle_file(f'{path}/chunk_ids/pred_chunk_ids_{parameter}_win{window_idx}.pickle', pred_chunk_ids)
    return pred_chunk_ids


def extract_train_chunk_ids(pred_chunk_ids, relevant_chunk_ids, configs, window_idx, parameter):
    # Extract chunk IDs for training that are not in prediction dataset
    train_chunk_ids = [chunk_id for chunk_id in relevant_chunk_ids if chunk_id not in pred_chunk_ids]

    path = get_script_path(configs)
    write_pickle_file(f'{path}/chunk_ids/train_chunk_ids_{parameter}_win{window_idx}.pickle', train_chunk_ids)

    return train_chunk_ids


def calc_standard_scaling_metrics(chunks, pred_chunk_ids, train_chunk_ids, resampling_methods):
    train_chunks = chunks[chunks['CHUNK_ID_FILLED_TH'].isin(train_chunk_ids)]
    pred_chunks = chunks[chunks['CHUNK_ID_FILLED_TH'].isin(pred_chunk_ids)]

    # Collect all values per dataset type (train/ predict)
    train_values, pred_values = list(), list()
    for resampling in resampling_methods:
        train_values = train_values + train_chunks[f'VITAL_PARAMTER_VALUE_{resampling}_RESAMPLING'].tolist()
        pred_values = pred_values + pred_chunks[f'VITAL_PARAMTER_VALUE_{resampling}_RESAMPLING'].tolist()

    return statistics.mean(train_values), statistics.stdev(train_values), statistics.mean(pred_values), \
           statistics.stdev(pred_values)


def apply_standard_scaling(chunk_values, mean, std):
    return (chunk_values - mean) / std


def revert_standard_scaling(parameter, window_idx, scaled_series, configs):
    path = get_script_path(configs)

    means = read_pickle_file(f'{path}/metrics/means_{parameter}.pickle')
    mean = means[f'win{window_idx}_pred']

    stds = read_pickle_file(f'{path}/metrics/standard_deviations_{parameter}.pickle')
    std = stds[f'win{window_idx}_pred']

    scaled_series_df = scaled_series.pd_dataframe()
    scaled_series_df.reset_index(level=0, inplace=True)
    scaled_series_df.columns = ['Time', 'Value_Scaled']

    scaled_series_df['Value'] = (scaled_series_df['Value_Scaled'] * std) + mean
    return scaled_series_df[['Time', 'Value']]


def create_time_series(resampling_methods, chunk_ids, chunk_type, original_chunks, parameter,
                       window_idx, configs, mean=0, std=1):
    # Apply filler as some time series have missing measurements what would lead to ValueError in prediction
    filler = MissingValuesFiller()

    for resampling in resampling_methods:
        series_per_resampling = dict()
        pred_scalers = dict()

        for chunk_id in chunk_ids:
            current_chunk = original_chunks[original_chunks['CHUNK_ID_FILLED_TH'] == chunk_id]

            # Scale chunk values if it is configured and create filled time series
            if configs.scaling_method == 'standard':
                current_chunk[f'SCALED_{resampling}'] = apply_standard_scaling(
                    current_chunk[f'VITAL_PARAMTER_VALUE_{resampling}_RESAMPLING'], mean, std)

                series_per_resampling[chunk_id] = filler.transform(TimeSeries.from_dataframe(
                    df=current_chunk,
                    time_col='CHARTTIME',
                    value_cols=[f'SCALED_{resampling}'],
                    freq='H'))

            elif configs.scaling_method == 'min-max':
                # Darts uses MinMaxScaler by default
                current_scaler = Scaler()

                series_per_resampling[chunk_id] = current_scaler.fit_transform(filler.transform(
                    TimeSeries.from_dataframe(
                        df=current_chunk,
                        time_col='CHARTTIME',
                        value_cols=[f'VITAL_PARAMTER_VALUE_{resampling}_RESAMPLING'],
                        freq='H')))

                if chunk_type == 'pred' and \
                        ((configs.with_exogenous_input and resampling != 'MEDIAN') or not configs.with_exogenous_input):
                    pred_scalers[chunk_id] = current_scaler

            else:  # apply no scaling
                series_per_resampling[chunk_id] = filler.transform(TimeSeries.from_dataframe(
                    df=current_chunk,
                    time_col='CHARTTIME',
                    value_cols=[f'VITAL_PARAMTER_VALUE_{resampling}_RESAMPLING'],
                    freq='H'))

        # Save series dict
        path = get_script_path(configs)
        write_pickle_file(f'{path}/time_series/time_series_{parameter}_win{window_idx}_{chunk_type}_'
                          f'{resampling.capitalize()}.pickle', series_per_resampling)

        # Save scaler dict if it was filled
        if pred_scalers:
            write_pickle_file(f'{path}/scalers/scalers_{parameter}_win{window_idx}_{resampling.capitalize()}.pickle',
                              pred_scalers)
