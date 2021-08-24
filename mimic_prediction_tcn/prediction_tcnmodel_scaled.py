"""
    PREDICTION WITH TCN MODEL, ALL PARAMETERS, AND MEDIAN RESAMPLED CHUNKS WHICH ARE MIN-MAX SCALED

    This script assumes that there is already the subdirectory '/TCNModel' in the directory '/data'. If you want to
    adjust which input size is taken and what parameters and models are used for the prediction, have a look at the five
    variables from line 28 to 39.

    Lastly, you have to install some packages:
    pip3 install u8darts[torch] seaborn
"""

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.models import TCNModel

import numpy as np
import os
import pandas as pd
import pickle
import sys
import time


####################
# Adjust Variables #
####################

# Parameter can be {'hr', 'bp', 'o2'}
parameters = ['hr', 'bp', 'o2']

# Number of chunks can be 1000, 2000 or 15000
n_chunks = 1000

# Style can be 'all' or '20_percent'
style = 'all'

# Define input length to imitate ARIMA training size (start after this length with forecast) and predict one data point
input_length = 12
output_length = 1

###########################################
# Create Folders, Init Variables & Models #
###########################################

approach = 'TCNModel'

# Create main folder for this script
if not os.path.isdir(f'./data/{approach}/{n_chunks}_chunks'):
    os.mkdir(f'./data/{approach}/{n_chunks}_chunks')
if not os.path.isdir(f'./data/{approach}/{n_chunks}_chunks/{style}'):
    os.mkdir(f'./data/{approach}/{n_chunks}_chunks/{style}')

# Create model-level confusion matrix
confusion_matrix_models = pd.DataFrame(
    columns=['ID', 'PARAMETER', 'MODEL', 'ENDOGENOUS', 'EXOGENOUS', 'FIRST_FORECAST', 'ALARM_TYPE',
             'FP', 'TP', 'FN', 'TN', 'N_HIGH_ALARMS', 'N_LOW_ALARMS', 'N_CHUNKS', 'N_ITERATIONS'])

# Note: Not changeable, see other scripts ending with "covariates" for MAX and MIN
endogenous_input = 'Median'
exogenous_input = np.nan

if style == 'all':
    n_windows = 5
elif style == '20_percent':
    n_windows = 1
else:
    raise ValueError('The style has to be "all" or "20_percent".')

# Note: Only use filler for now, remove after resampling script is fixed
filler = MissingValuesFiller()

# Create model
model = TCNModel(input_chunk_length=input_length,
                 output_chunk_length=output_length,
                 batch_size=input_length)  # batch_size must be <= input_length (bug fixed in Darts version 0.9.0)

for parameter in parameters:
    print(f'\n##############################\nCurrent Parameter: {parameter.upper()}\n'
          f'##############################\n', file=sys.stderr)

    start_time = time.time()

    # Create sub folder for each parameter
    if not os.path.isdir(f'./data/{approach}/{n_chunks}_chunks/{style}/{parameter}'):
        os.mkdir(f'./data/{approach}/{n_chunks}_chunks/{style}/{parameter}')

    # Create sub folder for the input type (median as endogenous variable)
    if not os.path.isdir(f'./data/{approach}/{n_chunks}_chunks/{style}/{parameter}/{endogenous_input}'):
        os.mkdir(f'./data/{approach}/{n_chunks}_chunks/{style}/{parameter}/{endogenous_input}')

    ###############################
    # Preprocess Resampled Chunks #
    ###############################

    print('Read resampled series and extract chunk IDs...', file=sys.stderr)

    # Extract first n_chunks resampled series
    resampled = pd.read_parquet(f'./data/resampling/resample_output_{parameter}_first{n_chunks}.parquet',
                                engine='pyarrow')

    # Extract relevant chunks
    relevant_series = dict()

    # Collect all series with minimal length
    for chunk_id in pd.unique(resampled.CHUNK_ID_FILLED_TH):
        current_series = resampled[resampled['CHUNK_ID_FILLED_TH'] == chunk_id]

        # At least input_chunk_length + output_chunk_length = 12 + 1 = 13 data points are required
        if len(current_series) > 12:
            relevant_series[chunk_id] = filler.transform(TimeSeries.from_dataframe(
                df=current_series,
                time_col='CHARTTIME',
                value_cols=[f'VITAL_PARAMTER_VALUE_{endogenous_input.upper()}_RESAMPLING'],
                freq='H'))

    # Extract all relevant chunk IDs
    relevant_chunk_ids = list(relevant_series.keys())

    # Calculate number of chunks corresponding to 20% of chunks
    twenty_percent = int((20 * len(relevant_chunk_ids)) / 100)

    # Iterate five times different 20% of the chunks (= 5 windows) to predict all chunks
    for window_idx in range(n_windows):

        print(f'{window_idx}. window\n', file=sys.stderr)

        # Extract 20% of series for prediction and catch last window to avoid ignoring chunks
        if window_idx == 4:
            pred_series = {chunk_id: relevant_series[chunk_id]
                           for chunk_id in list(relevant_series)[twenty_percent * window_idx:]}
        else:
            pred_series = {chunk_id: relevant_series[chunk_id]
                           for chunk_id in list(relevant_series)[twenty_percent*window_idx:
                                                                 twenty_percent*(window_idx+1)]}

        # Extract 80% of series for training
        train_series = {chunk_id: relevant_series[chunk_id] for chunk_id in relevant_chunk_ids
                        if chunk_id not in list(pred_series.keys())}

        # Define and fit scalers for training and prediction set
        pred_scalers = dict()

        # Normalize values
        for chunk_id in train_series.keys():
            current_scaler = Scaler()
            train_series[chunk_id] = current_scaler.fit_transform(train_series[chunk_id])

        for chunk_id in pred_series.keys():
            current_scaler = Scaler()
            pred_series[chunk_id] = current_scaler.fit_transform(pred_series[chunk_id])
            pred_scalers[chunk_id] = current_scaler

        print(f'#Chunks for training: {len(train_series)}', file=sys.stderr)
        print(f'#Chunks for prediction: {len(pred_series)}', file=sys.stderr)

        # Save training dict as pickle file
        train_series_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/{parameter}/{endogenous_input}/'
                              f'01_train_series_scaled_window{window_idx}.pickle', 'wb')
        pickle.dump(train_series, train_series_f, protocol=pickle.HIGHEST_PROTOCOL)
        train_series_f.close()

        # Save prediction dict as pickle file
        pred_series_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/{parameter}/{endogenous_input}/'
                             f'02_pred_series_scaled_window{window_idx}.pickle', 'wb')
        pickle.dump(pred_series, pred_series_f, protocol=pickle.HIGHEST_PROTOCOL)
        pred_series_f.close()

        # Save scalers for chunks to predict as pickle file
        pred_scalers_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/{parameter}/{endogenous_input}/'
                              f'03_pred_scalers_window{window_idx}.pickle', 'wb')
        pickle.dump(pred_scalers, pred_scalers_f, protocol=pickle.HIGHEST_PROTOCOL)
        pred_scalers_f.close()

        ###################
        # Pre-train Model #
        ###################

        print('Pre-train model...', file=sys.stderr)
        param_model = model

        # Pre-train with 80% of relevant series (steady training set)
        param_model.fit(series=list(train_series.values()),
                        verbose=True)

        # Save pre-trained model as pickle file
        pretrained_model_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/{parameter}/{endogenous_input}/'
                                  f'04_pre-trained_model_scaled_window{window_idx}.pickle', 'wb')
        pickle.dump(param_model, pretrained_model_f, protocol=pickle.HIGHEST_PROTOCOL)
        pretrained_model_f.close()

        confusion_matrix_chunks = pd.DataFrame(
            columns=['CHUNK_ID', 'SCALING', 'PARAMETER', 'MODEL', 'ENDOGENOUS', 'EXOGENOUS', 'FIRST_FORECAST',
                     'ALARM_TYPE', 'FP', 'TP', 'FN', 'TN', 'N_HIGH_ALARMS', 'N_LOW_ALARMS', 'N_ITERATIONS'])

        # Iterate chunk IDs we want to predict
        for chunk_id in pred_series.keys():

            print(f'\n##############################\nCurrent Chunk ID: {chunk_id}\n##############################\n',
                  file=sys.stderr)

            # Load original pre-trained model
            model_original_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/{parameter}/{endogenous_input}/'
                                    f'04_pre-trained_model_scaled_window{window_idx}.pickle', 'rb')
            model_for_iterations = pickle.load(model_original_f)
            model_original_f.close()

            # Create empty DataFrame for prediction result
            # Note: Have to use DataFrame because append() function of TimeSeries do not work
            final_pred = pd.DataFrame(columns=['Time', 'Value'])

            ########################
            # Hourly Predict Chunk #
            ########################

            # Do not iterate whole series-to-predict because of starting length of 12 (first prediction is for time 13)
            for iteration in range(len(pred_series[chunk_id]) - input_length):

                print(f'Iteration: {iteration}', file=sys.stderr)

                # Predict one measurement
                current_pred = model_for_iterations.predict(
                    n=output_length,
                    series=pred_series[chunk_id][:input_length + iteration])

                # Rescale predicted measurement
                current_pred = pred_scalers[chunk_id].inverse_transform(current_pred)

                # Add intermediate prediction result to DataFrame
                final_pred = final_pred.append({'Time': current_pred.start_time(),
                                                'Value': current_pred.first_value()},
                                               ignore_index=True)

            # Save final prediction of chunk as pickle file
            final_pred_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/{parameter}/{endogenous_input}/'
                                f'05_prediction_{chunk_id}_scaled_window{window_idx}.pickle', 'wb')
            pickle.dump(final_pred, final_pred_f, protocol=pickle.HIGHEST_PROTOCOL)
            final_pred_f.close()

            #####################################
            # Fill Chunk-level Confusion Matrix #
            #####################################

            # Extract original chunk
            original_chunk = resampled[resampled['CHUNK_ID_FILLED_TH'] == chunk_id].sort_values('CHARTTIME')
            original_chunk = original_chunk[input_length:].reset_index()

            # Add boolean indicating triggered high alarm for original value
            original_chunk['HIGH_ALARM_TRIGGERED'] = False
            original_chunk.loc[original_chunk[f'VITAL_PARAMTER_VALUE_{endogenous_input.upper()}_RESAMPLING']
                               > original_chunk['THRESHOLD_VALUE_HIGH'],
                               'HIGH_ALARM_TRIGGERED'] = True

            # Add boolean indicating triggered low alarm original value
            original_chunk['LOW_ALARM_TRIGGERED'] = False
            original_chunk.loc[original_chunk[f'VITAL_PARAMTER_VALUE_{endogenous_input.upper()}_RESAMPLING']
                               < original_chunk['THRESHOLD_VALUE_LOW'],
                               'LOW_ALARM_TRIGGERED'] = True

            # Add column with predicted value to chunk
            original_chunk['VALUE_PREDICTION'] = final_pred.Value

            # Add boolean indicating triggered high alarm for predicted value
            original_chunk['HIGH_ALARM_TRIGGERED_PREDICTION'] = False
            original_chunk.loc[original_chunk['VALUE_PREDICTION']
                               > original_chunk['THRESHOLD_VALUE_HIGH'],
                               'HIGH_ALARM_TRIGGERED_PREDICTION'] = True

            # Add boolean indicating triggered low alarm for predicted value
            original_chunk['LOW_ALARM_TRIGGERED_PREDICTION'] = False
            original_chunk.loc[original_chunk['VALUE_PREDICTION']
                               < original_chunk['THRESHOLD_VALUE_LOW'],
                               'LOW_ALARM_TRIGGERED_PREDICTION'] = True

            # Get indices where booleans are false or true for high alarms
            high_triggered = set(original_chunk.index[original_chunk['HIGH_ALARM_TRIGGERED']])
            high_triggered_pred = set(original_chunk.index[original_chunk['HIGH_ALARM_TRIGGERED_PREDICTION']])
            high_not_triggered = set(original_chunk.index[~original_chunk['HIGH_ALARM_TRIGGERED']])
            high_not_triggered_pred = set(original_chunk.index[~original_chunk['HIGH_ALARM_TRIGGERED_PREDICTION']])

            # Get indices where booleans are false or true for low alarms
            low_triggered = set(original_chunk.index[original_chunk['LOW_ALARM_TRIGGERED']])
            low_triggered_pred = set(original_chunk.index[original_chunk['LOW_ALARM_TRIGGERED_PREDICTION']])
            low_not_triggered = set(original_chunk.index[~original_chunk['LOW_ALARM_TRIGGERED']])
            low_not_triggered_pred = set(original_chunk.index[~original_chunk['LOW_ALARM_TRIGGERED_PREDICTION']])

            # Fill confusion matrix for high threshold analysis
            confusion_matrix_chunks = confusion_matrix_chunks.append({
                'CHUNK_ID': chunk_id,
                'SCALING': 'Min-Max',
                'PARAMETER': parameter.upper(),
                'MODEL': 'TCN',
                'ENDOGENOUS': endogenous_input,
                'EXOGENOUS': exogenous_input,
                'FIRST_FORECAST': input_length,
                'ALARM_TYPE': 'High',
                # Following 4 metrics look at how many indices are shared
                'TP': len(high_triggered.intersection(high_triggered_pred)),
                'FN': len(high_triggered.intersection(high_not_triggered_pred)),
                'FP': len(high_not_triggered.intersection(high_triggered_pred)),
                'TN': len(high_not_triggered.intersection(high_not_triggered_pred)),
                'N_HIGH_ALARMS': len(high_triggered),
                'N_LOW_ALARMS': len(low_triggered),
                'N_ITERATIONS': len(pred_series[chunk_id]) - input_length
            }, ignore_index=True)

            # Fill confusion matrix for low threshold analysis
            confusion_matrix_chunks = confusion_matrix_chunks.append({
                'CHUNK_ID': chunk_id,
                'SCALING': 'Min-Max',
                'PARAMETER': parameter.upper(),
                'MODEL': 'TCN',
                'ENDOGENOUS': endogenous_input,
                'EXOGENOUS': exogenous_input,
                'FIRST_FORECAST': input_length,
                'ALARM_TYPE': 'Low',
                # Following 4 metrics look at how many indices are shared
                'TP': len(low_triggered.intersection(low_triggered_pred)),
                'FN': len(low_triggered.intersection(low_not_triggered_pred)),
                'FP': len(low_not_triggered.intersection(low_triggered_pred)),
                'TN': len(low_not_triggered.intersection(low_not_triggered_pred)),
                'N_HIGH_ALARMS': len(high_triggered),
                'N_LOW_ALARMS': len(low_triggered),
                'N_ITERATIONS': len(pred_series[chunk_id]) - input_length
            }, ignore_index=True)

        # Save chunk-level confusion matrix after all chunks are processed
        confusion_matrix_chunks_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/confusion_matrix_chunks_'
                                         f'{parameter}_{endogenous_input}_scaled_window{window_idx}.pickle', 'wb')
        pickle.dump(confusion_matrix_chunks, confusion_matrix_chunks_f, protocol=pickle.HIGHEST_PROTOCOL)
        confusion_matrix_chunks_f.close()

    #####################################
    # Fill Model-level Confusion Matrix #
    #####################################

    # Collect chunk-level confusion matrices of all five windows
    confusion_matrix_chunks_concat = pd.DataFrame(
        columns=['CHUNK_ID', 'PARAMETER', 'MODEL', 'ENDOGENOUS', 'EXOGENOUS', 'FIRST_FORECAST', 'ALARM_TYPE', 'FP',
                 'TP', 'FN', 'TN', 'N_HIGH_ALARMS', 'N_LOW_ALARMS', 'N_ITERATIONS'])

    for file in os.listdir(f'./data/{approach}/{n_chunks}_chunks/{style}/'):
        if os.path.isfile(os.path.join(f'./data/{approach}/{n_chunks}_chunks/{style}/', file)) and \
                file.startswith(f'confusion_matrix_chunks_{parameter}_{endogenous_input}_scaled'):

            current_chunk_matrix_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/{file}', 'rb')
            current_chunk_matrix = pickle.load(current_chunk_matrix_f)
            current_chunk_matrix_f.close()

            confusion_matrix_chunks_concat = pd.concat([confusion_matrix_chunks_concat, current_chunk_matrix])

    confusion_matrix_chunks_concat.reset_index(inplace=True, drop=True)

    runtime = time.time() - start_time

    # Fill model-level confusion matrix per parameter and model type (HIGH alarm forecasting)
    confusion_matrix_chunks_concat_high = \
        confusion_matrix_chunks_concat[confusion_matrix_chunks_concat['ALARM_TYPE'] == 'High']

    confusion_matrix_models = confusion_matrix_models.append({
        # T = TCNModel, model_number = {01, ..., 04} and H = High
        'ID': f'{parameter.upper()}_T_03_H',
        'PARAMETER': parameter.upper(),
        'RUNTIME': runtime,
        'MODEL': 'TCN',
        'SCALING': 'Min-Max',
        'LIBRARY': 'darts',
        'ENDOGENOUS': endogenous_input,
        'EXOGENOUS': exogenous_input,
        'FIRST_FORECAST': input_length,
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

    # Fill model-level confusion matrix per parameter and model type (LOW alarm forecasting)
    confusion_matrix_chunks_concat_low = \
        confusion_matrix_chunks_concat[confusion_matrix_chunks_concat['ALARM_TYPE'] == 'Low']

    confusion_matrix_models = confusion_matrix_models.append({
        # T = TCNModel, model_number = {01, ..., 04} and L = Low
        'ID': f'{parameter.upper()}_T_03_L',
        'PARAMETER': parameter.upper(),
        'RUNTIME': runtime,
        'MODEL': 'TCN',
        'SCALING': 'Min-Max',
        'LIBRARY': 'darts',
        'ENDOGENOUS': endogenous_input,
        'EXOGENOUS': exogenous_input,
        'FIRST_FORECAST': input_length,
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

# Save model-level confusion matrix after all model types and parameters are processed
# Note: adjust path name if you want to execute this script in parallel with different parameters/ model types
confusion_matrix_models_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/confusion_matrix_models_scaled_'
                                 f'{endogenous_input}_s2.pickle', 'wb')
pickle.dump(confusion_matrix_models, confusion_matrix_models_f, protocol=pickle.HIGHEST_PROTOCOL)
confusion_matrix_models_f.close()

print('\nFinished.', file=sys.stderr)
sys.stderr.close()
