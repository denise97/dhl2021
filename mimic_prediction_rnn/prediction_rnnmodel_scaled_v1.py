"""
    PREDICTION WITH ALL RNN MODELS, ALL PARAMETERS, AND MEDIAN RESAMPLED CHUNKS WHICH ARE Z-SCALED

    This script assumes that there is already the subdirectory '/RNNModel' in the directory '/data'. If you want to
    adjust which input size is taken and what parameters and models are used for the prediction, have a look at the six
    variables from line 26 to 40.

    Lastly, you have to install some packages:
    pip3 install u8darts[torch] seaborn
"""

from darts.models import RNNModel

import numpy as np
import os
import pandas as pd
import pickle
import sys
import time


####################
# Adjust Variables #
####################

# Model type can be {'RNN', 'LSTM', 'GRU'}
model_types = ['RNN', 'LSTM', 'GRU']

# Parameter can be {'hr', 'bp', 'o2'}
parameters = ['hr', 'bp', 'o2']

# Number of chunks can be 1000, 2000 or 15000
n_chunks = 2000

# Style can be 'all' or '20_percent'
style = 'all'

# Define input length to imitate ARIMA training size (start after this length with forecast) and predict one data point
input_length = 12
output_length = 1

###########################################
# Create Folders, Init Variables & Models #
###########################################

approach = 'RNNModel'

# Prepare rescaling
means_f = open(f'./data/z_scaled/means_z_scaling.pickle', 'rb')
means = pickle.load(means_f)
means_f.close()

stds_f = open(f'./data/z_scaled/stds_z_scaling.pickle', 'rb')
stds = pickle.load(stds_f)
stds_f.close()

def z_retransform(scaled_series, parameter, window_idx):
    mean = means[f'{parameter}_{window_idx}_pred']
    std = stds[f'{parameter}_{window_idx}_pred']

    scaled_series_df = scaled_series.pd_dataframe()
    scaled_series_df.reset_index(level=0, inplace=True)
    scaled_series_df.columns = ['Time', 'Value_Scaled']

    scaled_series_df['Value'] = (scaled_series_df['Value_Scaled'] * std) + mean
    return scaled_series_df[['Time', 'Value']]

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

model_numbers = {
    ('RNN',     'Median'):  '01',
    ('LSTM',    'Median'):  '03',
    ('GRU',     'Median'):  '05'
}

if style == 'all':
    n_windows = 5
elif style == '20_percent':
    n_windows = 1
else:
    raise ValueError('The style has to be "all" or "20_percent".')

for model_type in model_types:
    print(f'\n##############################\nCurrent Model Type: {model_type}\n##############################\n',
          file=sys.stderr)

    # Create sub folder for each model type
    if not os.path.isdir(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}'):
        os.mkdir(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}')

    # Create model per model type
    model = RNNModel(model=model_type,
                     input_chunk_length=input_length,
                     output_chunk_length=output_length,
                     batch_size=input_length)  # batch_size must be <= input_length (bug fixed in Darts version 0.9.0)

    for parameter in parameters:
        print(f'\n##############################\nCurrent Parameter: {parameter.upper()}\n'
              f'##############################\n', file=sys.stderr)

        start_time = time.time()

        # Create sub folder for each parameter
        if not os.path.isdir(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}/{parameter}'):
            os.mkdir(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}/{parameter}')

        # Create sub folder for the input type (median as endogenous variable)
        if not os.path.isdir(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}/{parameter}/{endogenous_input}'):
            os.mkdir(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}/{parameter}/{endogenous_input}')

        ###############################
        # Preprocess Resampled Chunks #
        ###############################

        # Iterate five times different 20% of the chunks (= 5 windows) to predict all chunks
        for window_idx in range(n_windows):

            print(f'{window_idx}. window\n', file=sys.stderr)

            train_series_f = open(f'./data/z_scaled/{parameter}_{window_idx}_train_median.pickle', 'rb')
            train_series = pickle.load(train_series_f)
            train_series_f.close()

            pred_series_f = open(f'./data/z_scaled/{parameter}_{window_idx}_pred_median.pickle', 'rb')
            pred_series = pickle.load(pred_series_f)
            pred_series_f.close()

            print(f'#Chunks for training: {len(train_series)}', file=sys.stderr)
            print(f'#Chunks for prediction: {len(pred_series)}', file=sys.stderr)

            ###################
            # Pre-train Model #
            ###################

            print('Pre-train model...', file=sys.stderr)
            param_model = model

            # Pre-train with 80% of relevant series (steady training set)
            param_model.fit(series=list(train_series.values()),
                            verbose=True)

            # Save pre-trained model as pickle file
            pretrained_model_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}/{parameter}/'
                                      f'{endogenous_input}/04_pre-trained_model_scaled_v1_window{window_idx}.pickle', 'wb')
            pickle.dump(param_model, pretrained_model_f, protocol=pickle.HIGHEST_PROTOCOL)
            pretrained_model_f.close()

            confusion_matrix_chunks = pd.DataFrame(
                columns=['CHUNK_ID', 'SCALED', 'PARAMETER', 'MODEL', 'ENDOGENOUS', 'EXOGENOUS', 'FIRST_FORECAST',
                         'ALARM_TYPE', 'FP', 'TP', 'FN', 'TN', 'N_HIGH_ALARMS', 'N_LOW_ALARMS', 'N_ITERATIONS'])

            # Iterate chunk IDs we want to predict
            for chunk_id in pred_series.keys():

                print(f'\n##############################\nCurrent Chunk ID: {chunk_id}\n##############################\n',
                      file=sys.stderr)

                # Load original pre-trained model
                model_original_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}/{parameter}/'
                                        f'{endogenous_input}/04_pre-trained_model_scaled_v1_window{window_idx}.pickle',
                                        'rb')
                model_for_iterations = pickle.load(model_original_f)
                model_original_f.close()

                # Create empty DataFrame for prediction result
                # Note: Have to use DataFrame because append() function of TimeSeries do not work
                final_pred = pd.DataFrame(columns=['Time', 'Value'])

                ########################
                # Hourly Predict Chunk #
                ########################

                # Do not iterate whole series-to-predict because of start length of 12 (first prediction is for time 13)
                for iteration in range(len(pred_series[chunk_id]) - input_length):

                    print(f'Iteration: {iteration}', file=sys.stderr)

                    # Predict one measurement
                    current_pred = model_for_iterations.predict(
                        n=output_length,
                        series=pred_series[chunk_id][:input_length + iteration])

                    # Rescale predicted measurement (returned as DataFrame)
                    current_pred = z_retransform(current_pred, parameter, window_idx)

                    # Add intermediate prediction result to DataFrame
                    final_pred = pd.concat([final_pred, current_pred], axis=0, ignore_index=True)

                # Save final prediction of chunk as pickle file
                final_pred_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}/{parameter}/'
                                    f'{endogenous_input}/05_prediction_{chunk_id}_scaled_v1_window{window_idx}.pickle',
                                    'wb')
                pickle.dump(final_pred, final_pred_f, protocol=pickle.HIGHEST_PROTOCOL)
                final_pred_f.close()

                #####################################
                # Fill Chunk-level Confusion Matrix #
                #####################################

                # Extract original chunk
                resampled = pd.read_parquet(f'./data/resampling/resample_output_{parameter}_first{n_chunks}.parquet',
                                            engine='pyarrow')
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
                    'SCALED': True,
                    'PARAMETER': parameter.upper(),
                    'MODEL': model_type,
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
                    'SCALED': True,
                    'PARAMETER': parameter.upper(),
                    'MODEL': model_type,
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
                                             f'{model_type}_{parameter}_{endogenous_input}_scaled_window{window_idx}'
                                             f'_v1.pickle', 'wb')
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
                    file.startswith(f'confusion_matrix_chunks_{model_type}_{parameter}_{endogenous_input}_scaled') and \
                    file.endswith('_v1'):

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
            # R = RNNModel, model_number = {01, ..., 09} and H = High
            'ID': f'{parameter.upper()}_R_{model_numbers[model_type, endogenous_input]}_H_v1',
            'PARAMETER': parameter.upper(),
            'RUNTIME': runtime,
            'MODEL': model_type,
            'SCALED': True,
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
            # R = RNNModel, model_number = {01, ..., 12} and L = Low
            'ID': f'{parameter.upper()}_R_{model_numbers[model_type, endogenous_input]}_L_v1',
            'PARAMETER': parameter.upper(),
            'RUNTIME': runtime,
            'MODEL': model_type,
            'SCALED': True,
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
                                 f'{endogenous_input}_v1.pickle', 'wb')
pickle.dump(confusion_matrix_models, confusion_matrix_models_f, protocol=pickle.HIGHEST_PROTOCOL)
confusion_matrix_models_f.close()

print('\nFinished.', file=sys.stderr)
sys.stderr.close()
