"""
    ALARM FORECASTING WITH RNNMODEL CLASS BY DARTS AND ENDOGENOUS AS WELL AS EXOGENOUS INPUT

    Before executing, you can adjust the configurations in configs.py and have to ensure that the parquet files
    containing the resampled chunks are located in the "data/resampling" folder.

    Afterwards, you have to install Darts' DL package:
    pip3 install pickle5 u8darts[torch]
"""


from darts.models import RNNModel
from io_helpers import *
from postprocessing import *
from preprocessing import *

import configs
import importlib
import numpy as np
import os
import pandas as pd
import sys
import time


################
# SETUP SCRIPT #
################

# Set configurations
importlib.reload(configs)
configurations = Configurations(configs)

# Check validity of configurations parameters (defined in configs.py)
verify_configs(configurations)

if not configurations.with_exogenous_input:
    raise Warning('Update the "with_exogenous_input" configuration parameter to forecast with exogenous inputs. '
                  'Otherwise, use the "forecast_alarms.py" script.')

endogenous_input_low = 'Min'
endogenous_input_high = 'Max'
exogenous_input = 'Median'
resampling_methods = [endogenous_input_low.upper(), endogenous_input_high.upper(), exogenous_input.upper()]

script_path = get_script_path(configurations)

# Create main folder of script, and sub-folders for preprocessing and forecasting data
create_folders(configurations)

# Extract abbreviation and col value for scaling method
if configurations.scaling_method == 'standard':
    scaling_abbrev = 's1'
elif configurations.scaling_method == 'min-max':
    scaling_abbrev = 's2'
else:
    scaling_abbrev = 'n'

#################
# PREPROCESSING #
#################

# Note: Use 'sys.stderr' to pipe all console output into file
print('Preprocessing ...', file=sys.stderr)

for parameter in configurations.parameters:
    # Perform preprocessing if it has not been executed for current parameter yet
    if not os.path.isfile(f'{script_path}/prep_finished_{parameter}.txt'):
        chunks = pd.read_parquet(f'data/resampling/resample_output_{parameter}_first{configurations.n_chunks}.parquet',
                                 engine='pyarrow')
        relevant_chunk_ids = extract_relevant_chunk_ids(chunks, configurations, parameter)

        means, stds = dict(), dict()

        for window_idx in range(configurations.n_windows):
            pred_chunk_ids = extract_prediction_chunk_ids(window_idx, relevant_chunk_ids, configurations, parameter)
            train_chunk_ids = extract_train_chunk_ids(pred_chunk_ids, relevant_chunk_ids, configurations, window_idx,
                                                      parameter)

            if configurations.scaling_method == 'standard':
                train_mean, train_std, pred_mean, pred_std = calc_standard_scaling_metrics(chunks, pred_chunk_ids,
                                                                                           train_chunk_ids,
                                                                                           resampling_methods)

                # Add metrics to dicts
                means[f'win{window_idx}_train'] = train_mean
                stds[f'win{window_idx}_train'] = train_std

                means[f'win{window_idx}_pred'] = pred_mean
                stds[f'win{window_idx}_pred'] = pred_std

                # Export time series and scaler dicts for standard scaled chunks
                create_time_series(resampling_methods, train_chunk_ids, 'train', chunks, parameter, window_idx,
                                   configurations, train_mean, train_std)
                create_time_series(resampling_methods, pred_chunk_ids, 'pred', chunks, parameter, window_idx,
                                   configurations, pred_mean, pred_std)

            else:
                # Export time series and scaler dicts for mix-max or no scaled chunks
                create_time_series(resampling_methods, train_chunk_ids, 'train', chunks, parameter, window_idx,
                                   configurations)
                create_time_series(resampling_methods, pred_chunk_ids, 'pred', chunks, parameter, window_idx,
                                   configurations)

        if configurations.scaling_method == 'standard':
            # Export dicts containing means and standard deviations for current parameter
            write_pickle_file(f'{script_path}/metrics/means_{parameter}.pickle', means)
            write_pickle_file(f'{script_path}/metrics/standard_deviations_{parameter}.pickle', stds)

        # Create small .txt file if preprocessing was successful for current parameter
        prep_finished_f = open(f'{script_path}/prep_finished_{parameter}.txt', 'w')
        prep_finished_f.write(f'Preprocessing for {parameter.upper()} is finished.\n')
        prep_finished_f.close()

    else:
        print(f'Preprocessing was already performed and thus was not repeated again for parameter {parameter.upper()}',
              file=sys.stderr)

######################
# SERIES FORECASTING #
######################

# Create model-level confusion matrix
confusion_matrix_models = pd.DataFrame(
    columns=['ID', 'PARAMETER', 'MODEL', 'ENDOGENOUS', 'EXOGENOUS', 'FIRST_FORECAST', 'ALARM_TYPE',
             'FP', 'TP', 'FN', 'TN', 'N_HIGH_ALARMS', 'N_LOW_ALARMS', 'N_CHUNKS', 'N_ITERATIONS'])

for model_type in configurations.model_types:
    print(f'Current Model: {model_type}', file=sys.stderr)

    for parameter in configurations.parameters:
        print(f'Current Parameter: {parameter.upper()}', file=sys.stderr)
        start_time = time.time()

        for window_idx in range(configurations.n_windows):
            print(f'Current Window: {window_idx}', file=sys.stderr)

            # Create model
            model = RNNModel(model=model_type,
                             input_chunk_length=configurations.input_length,
                             output_chunk_length=configurations.output_length,
                             # batch_size must be <= input_length (bug fixed in Darts version 0.9.0)
                             batch_size=configurations.input_length)

            # Read time-series input
            train_series_endo_low = read_pickle_file(f'{script_path}/time_series/time_series_{parameter}_win{window_idx}'
                                                     f'_train_{endogenous_input_low}.pickle')

            train_series_endo_high = read_pickle_file(f'{script_path}/time_series/time_series_{parameter}_win{window_idx}'
                                                      f'_train_{endogenous_input_high}.pickle')

            train_series_exo = read_pickle_file(f'{script_path}/time_series/time_series_{parameter}_win{window_idx}_'
                                                f'train_{exogenous_input}.pickle')

            pred_series_endo_low = read_pickle_file(f'{script_path}/time_series/time_series_{parameter}_win{window_idx}'
                                                    f'_pred_{endogenous_input_low}.pickle')

            pred_series_endo_high = read_pickle_file(f'{script_path}/time_series/time_series_{parameter}_win{window_idx}'
                                                     f'_pred_{endogenous_input_high}.pickle')

            pred_series_exo = read_pickle_file(f'{script_path}/time_series/time_series_{parameter}_win{window_idx}_pred'
                                               f'_{exogenous_input}.pickle')

            print('Pre-train for low alarm forecasting...', file=sys.stderr)

            # Pre-train with (e.g. 80%) of relevant MIN series (steady training set)
            model_low = model
            model_low.fit(series=list(train_series_endo_low.values()),
                          covariates=list(train_series_exo.values()),
                          verbose=True)

            write_pickle_file(f'{script_path}/training/pre-trained_model_{model_type}_{parameter}_win{window_idx}_'
                              f'{endogenous_input_low}.pickle', model_low)

            print('Pre-train for high alarm forecasting...', file=sys.stderr)

            # Pre-train with (e.g. 80%) of relevant MAX series (steady training set)
            model_high = model
            model_high.fit(series=list(train_series_endo_high.values()),
                           covariates=list(train_series_exo.values()),
                           verbose=True)

            write_pickle_file(f'{script_path}/training/pre-trained_model_{model_type}_{parameter}_win{window_idx}_'
                              f'{endogenous_input_high}.pickle', model_high)

            confusion_matrix_chunks = pd.DataFrame(
                columns=['CHUNK_ID', 'SCALING', 'PARAMETER', 'MODEL', 'ENDOGENOUS', 'EXOGENOUS', 'FIRST_FORECAST',
                         'ALARM_TYPE', 'FP', 'TP', 'FN', 'TN', 'N_HIGH_ALARMS', 'N_LOW_ALARMS', 'N_ITERATIONS'])

            # Iterate chunk IDs we want to predict
            for chunk_id in pred_series_endo_low.keys():

                print('Series forecasting for low alarms...', file=sys.stderr)

                # Load original pre-trained model
                model_for_iterations_low = read_pickle_file(f'{script_path}/training/pre-trained_model_{model_type}_'
                                                            f'{parameter}_win{window_idx}_{endogenous_input_low}'
                                                            f'.pickle')

                # Create empty DataFrame for prediction result
                # Note: Have to use DataFrame because append() function of TimeSeries do not work
                final_pred_low = pd.DataFrame(columns=['Time', 'Value'])

                # Do not iterate whole series-to-predict because of input length
                for iteration in range(len(pred_series_endo_low[chunk_id]) - configurations.input_length):

                    # Predict as many measurements as defined in config 'output_length'
                    current_pred_low = model_for_iterations_low.predict(
                        n=configurations.output_length,
                        series=pred_series_endo_low[chunk_id][:configurations.input_length + iteration],
                        covariates=pred_series_exo[chunk_id][:configurations.input_length + iteration])

                    # Collect predicted measurement/s (and rescale if needed)
                    if configurations.scaling_method == 'standard':
                        current_pred_low = revert_standard_scaling(parameter, window_idx, current_pred_low,
                                                                   configurations)
                        final_pred_low = pd.concat([final_pred_low, current_pred_low], axis=0, ignore_index=True)

                    else:
                        if configurations.scaling_method == 'min-max':
                            pred_scalers = read_pickle_file(f'{script_path}/scalers/scalers_{parameter}_win{window_idx}'
                                                            f'_{endogenous_input_low}.pickle')
                            current_pred_low = pred_scalers[chunk_id].inverse_transform(current_pred_low)
                        final_pred_low = final_pred_low.append({'Time': current_pred_low.start_time(),
                                                                'Value': current_pred_low.first_value()},
                                                               ignore_index=True)

                write_pickle_file(f'{script_path}/prediction/prediction_{model_type}_{parameter}_win{window_idx}_'
                                  f'{chunk_id}_{endogenous_input_low}.pickle', final_pred_low)

                print('Series forecasting for high alarms...', file=sys.stderr)

                # Load original pre-trained model
                model_for_iterations_high = read_pickle_file(f'{script_path}/training/pre-trained_model_{model_type}_'
                                                             f'{parameter}_win{window_idx}_{endogenous_input_high}'
                                                             f'.pickle')

                # Create empty DataFrame for prediction result
                # Note: Have to use DataFrame because append() function of TimeSeries do not work
                final_pred_high = pd.DataFrame(columns=['Time', 'Value'])

                # Do not iterate whole series-to-predict because of input length
                for iteration in range(len(pred_series_endo_high[chunk_id]) - configurations.input_length):

                    # Predict as many measurements as defined in config 'output_length'
                    current_pred_high = model_for_iterations_high.predict(
                        n=configurations.output_length,
                        series=pred_series_endo_high[chunk_id][:configurations.input_length + iteration],
                        covariates=pred_series_exo[chunk_id][:configurations.input_length + iteration])

                    # Collect predicted measurement/s (and rescale if needed)
                    if configurations.scaling_method == 'standard':
                        current_pred_high = revert_standard_scaling(parameter, window_idx, current_pred_high,
                                                                   configurations)
                        final_pred_high = pd.concat([final_pred_high, current_pred_high], axis=0, ignore_index=True)

                    else:
                        if configurations.scaling_method == 'min-max':
                            pred_scalers = read_pickle_file(f'{script_path}/scalers/scalers_{parameter}_win{window_idx}'
                                                            f'_{endogenous_input_high}.pickle')
                            current_pred_high = pred_scalers[chunk_id].inverse_transform(current_pred_high)
                        final_pred_high = final_pred_high.append({'Time': current_pred_high.start_time(),
                                                                  'Value': current_pred_high.first_value()},
                                                                 ignore_index=True)

                write_pickle_file(f'{script_path}/prediction/prediction_{model_type}_{parameter}_win{window_idx}_'
                                  f'{chunk_id}_{endogenous_input_high}.pickle', final_pred_high)

                #####################
                # ALARM FORECASTING #
                #####################

                # Extract original chunk
                chunks = pd.read_parquet(f'data/resampling/resample_output_{parameter}_first{configurations.n_chunks}'
                                         f'.parquet', engine='pyarrow')
                original_chunk = chunks[chunks['CHUNK_ID_FILLED_TH'] == chunk_id].sort_values('CHARTTIME')
                original_chunk = original_chunk[configurations.input_length:].reset_index()

                fill_alarm_triggering_cols(original_chunk, [final_pred_low, final_pred_high], configurations)

                # Get indices where booleans are set for low alarms
                low_triggered, low_triggered_pred, low_not_triggered, low_not_triggered_pred = \
                    get_alarm_triggering_indices_for_alarm_type(original_chunk, 'Low')

                # Get indices where booleans are set for high alarms
                high_triggered, high_triggered_pred, high_not_triggered, high_not_triggered_pred = \
                    get_alarm_triggering_indices_for_alarm_type(original_chunk, 'High')

                # Fill confusion matrix for low threshold analysis
                confusion_matrix_chunks = confusion_matrix_chunks.append({
                    'CHUNK_ID': chunk_id,
                    'PARAMETER': parameter.upper(),
                    'MODEL': model_type,
                    'SCALING': configurations.scaling_method,
                    'LIBRARY': 'darts',
                    'ENDOGENOUS': endogenous_input_low,
                    'EXOGENOUS': exogenous_input,
                    'FIRST_FORECAST': configurations.input_length,
                    'ALARM_TYPE': 'Low',
                    # Following 4 metrics look at how many indices are shared
                    'TP': len(low_triggered.intersection(low_triggered_pred)),
                    'FN': len(low_triggered.intersection(low_not_triggered_pred)),
                    'FP': len(low_not_triggered.intersection(low_triggered_pred)),
                    'TN': len(low_not_triggered.intersection(low_not_triggered_pred)),
                    'N_HIGH_ALARMS': len(high_triggered),
                    'N_LOW_ALARMS': len(low_triggered),
                    'N_ITERATIONS': len(pred_series_endo_low[chunk_id]) - configurations.input_length
                }, ignore_index=True)

                # Fill confusion matrix for high threshold analysis
                confusion_matrix_chunks = confusion_matrix_chunks.append({
                    'CHUNK_ID': chunk_id,
                    'PARAMETER': parameter.upper(),
                    'MODEL': model_type,
                    'SCALING': configurations.scaling_method,
                    'LIBRARY': 'darts',
                    'ENDOGENOUS': endogenous_input_high,
                    'EXOGENOUS': exogenous_input,
                    'FIRST_FORECAST': configurations.input_length,
                    'ALARM_TYPE': 'High',
                    # Following 4 metrics look at how many indices are shared
                    'TP': len(high_triggered.intersection(high_triggered_pred)),
                    'FN': len(high_triggered.intersection(high_not_triggered_pred)),
                    'FP': len(high_not_triggered.intersection(high_triggered_pred)),
                    'TN': len(high_not_triggered.intersection(high_not_triggered_pred)),
                    'N_HIGH_ALARMS': len(high_triggered),
                    'N_LOW_ALARMS': len(low_triggered),
                    'N_ITERATIONS': len(pred_series_endo_high[chunk_id]) - configurations.input_length
                }, ignore_index=True)

            # Save chunk-level confusion matrix after all chunks are processed
            write_pickle_file(f'{script_path}/confusion_matrices/confusion_matrix_chunks_{model_type}_{parameter}_'
                              f'win{window_idx}_{endogenous_input_high}.pickle',
                              confusion_matrix_chunks)

        runtime = time.time() - start_time

        # Fill model-level confusion matrix per parameter and model type (LOW alarm forecasting)
        confusion_matrix_chunks_concat_low = get_chunk_matrices_for_alarm_type(model_type, parameter, 'Low',
                                                                               configurations)

        confusion_matrix_models = confusion_matrix_models.append({
            # RN = Vanilla RNN, LS = LSTM, GR = GRU
            # 02 = model with covariates
            # H = High
            'ID': f'{parameter.upper()}_{model_type[:2]}_02_{scaling_abbrev}_L',
            'PARAMETER': parameter.upper(),
            'RUNTIME': runtime,
            'MODEL': model_type,
            'SCALING': configurations.scaling_method,
            'LIBRARY': 'darts',
            'ENDOGENOUS': endogenous_input_low,
            'EXOGENOUS': exogenous_input,
            'FIRST_FORECAST': configurations.input_length,
            'ALARM_TYPE': 'Low',
            'FP': confusion_matrix_chunks_concat_low['FP'].sum(),
            'TP': confusion_matrix_chunks_concat_low['TP'].sum(),
            'FN': confusion_matrix_chunks_concat_low['FN'].sum(),
            'TN': confusion_matrix_chunks_concat_low['TN'].sum(),
            'N_HIGH_ALARMS': confusion_matrix_chunks_concat_low['N_HIGH_ALARMS'].sum(),
            'N_LOW_ALARMS': confusion_matrix_chunks_concat_low['N_LOW_ALARMS'].sum(),
            'N_CHUNKS': len(confusion_matrix_chunks_concat_low),
            'N_ITERATIONS': confusion_matrix_chunks_concat_low['N_ITERATIONS'].sum()
        }, ignore_index=True)

        # Fill model-level confusion matrix per parameter and model type (HIGH alarm forecasting)
        confusion_matrix_chunks_concat_high = get_chunk_matrices_for_alarm_type(model_type, parameter, 'High',
                                                                                configurations)

        confusion_matrix_models = confusion_matrix_models.append({
            # RN = Vanilla RNN, LS = LSTM, GR = GRU
            # 02 = model with covariates
            # H = High
            'ID': f'{parameter.upper()}_{model_type[:2]}_02_{scaling_abbrev}_H',
            'PARAMETER': parameter.upper(),
            'RUNTIME': runtime,
            'MODEL': model_type,
            'SCALING': configurations.scaling_method,
            'LIBRARY': 'darts',
            'ENDOGENOUS': endogenous_input_high,
            'EXOGENOUS': exogenous_input,
            'FIRST_FORECAST': configurations.input_length,
            'ALARM_TYPE': 'High',
            'FP': confusion_matrix_chunks_concat_high['FP'].sum(),
            'TP': confusion_matrix_chunks_concat_high['TP'].sum(),
            'FN': confusion_matrix_chunks_concat_high['FN'].sum(),
            'TN': confusion_matrix_chunks_concat_high['TN'].sum(),
            'N_HIGH_ALARMS': confusion_matrix_chunks_concat_high['N_HIGH_ALARMS'].sum(),
            'N_LOW_ALARMS': confusion_matrix_chunks_concat_high['N_LOW_ALARMS'].sum(),
            'N_CHUNKS': len(confusion_matrix_chunks_concat_high),
            'N_ITERATIONS': confusion_matrix_chunks_concat_high['N_ITERATIONS'].sum()
        }, ignore_index=True)

# Save model-level confusion matrix after all model types and parameter are processed
# Note: adjust path name if you want to execute this script in parallel with different parameters/ model types
write_pickle_file(f'{script_path}/confusion_matrices/confusion_matrix_models_{"_".join(configurations.model_types)}_'
                  f'{"_".join(configurations.parameters)}.pickle', confusion_matrix_models)

print('\nFinished.', file=sys.stderr)
sys.stderr.close()
