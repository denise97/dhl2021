"""
    PREDICTION WITH ALL RNN MODELS, ALL PARAMETERS, AND MIN RESAMPLED CHUNKS

    This script assumes that there is already the subdirectory '/darts' in the directories '/plots' and '/data'. If you
    want to adjust which input size is taken and what parameters and models are used for the prediction, have a look at
    the three variables from line 27 to 39.

    Lastly, you have to install some packages:
    pip3 install u8darts[torch] seaborn
"""


from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel

import os
import pandas as pd
import pickle
import sys


####################
# Adjust Variables #
####################

# Model type can be {'RNN', 'LSTM', 'GRU'}
model_types = ['RNN', 'LSTM', 'GRU']

# Parameter can be {'hr', 'bp', 'o2'}
parameters = ['hr', 'bp']

# Number of chunks can be 1,000 or 15,000
# TODO: Execute with 15,000
n_chunks = 1000

# Take input of 12 data points to imitate ARIMA look-back and always predict one data point
input_length = 12
output_length = 1

##################
# Create Folders #
##################

# Create sub folders for this script
if not os.path.isdir(f'./data/darts/{n_chunks}_chunks'):
    os.mkdir(f'./data/darts/{n_chunks}_chunks')
if not os.path.isdir(f'./plots/darts/{n_chunks}_chunks'):
    os.mkdir(f'./plots/darts/{n_chunks}_chunks')

# Create model-level confusion matrix
confusion_matrix_models = pd.DataFrame(
    columns=['ID', 'PARAMETER', 'MODEL', 'ENDOGENOUS', 'EXOGENOUS', 'FORECAST_TYPE', 'FIRST_FORECAST', 'ALARM_TYPE',
             'FP', 'TP', 'FN', 'TN', 'N_CHUNKS', 'N_ITERATIONS'])

# Note: Not changeable, see other scripts for MAX and MIN
endogenous_input = 'MIN'
exogenous_input = 'MEDIAN'

for model_type in model_types:
    print(f'\n##############################\nCurrent Model Type: {model_type}\n##############################\n',
          file=sys.stderr)

    # Create sub folders for each model type
    if not os.path.isdir(f'./data/darts/{n_chunks}_chunks/{model_type}'):
        os.mkdir(f'./data/darts/{n_chunks}_chunks/{model_type}')
    if not os.path.isdir(f'./plots/darts/{n_chunks}_chunks/{model_type}'):
        os.mkdir(f'./plots/darts/{n_chunks}_chunks/{model_type}')

    # Create model per model type
    # TODO: try less / more epochs than 100
    model = RNNModel(model=model_type,
                     input_chunk_length=input_length,
                     output_chunk_length=output_length,
                     batch_size=input_length)  # batch_size must be <= input_length (bug of Darts)

    # TODO: NaN values in predictions of O2 series with all model types
    for parameter in parameters:
        print(
            f'##############################\nCurrent Parameter: {parameter.upper()}\n##############################\n',
            file=sys.stderr)

        # Create sub folders for each parameter
        if not os.path.isdir(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}'):
            os.mkdir(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}')
        if not os.path.isdir(f'./plots/darts/{n_chunks}_chunks/{model_type}/{parameter}'):
            os.mkdir(f'./plots/darts/{n_chunks}_chunks/{model_type}/{parameter}')

        # Create sub folder for input type (median, max, or min as endogenous variable)
        if not os.path.isdir(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}/{endogenous_input}'):
            os.mkdir(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}/{endogenous_input}')
        if not os.path.isdir(f'./plots/darts/{n_chunks}_chunks/{model_type}/{parameter}/{endogenous_input}'):
            os.mkdir(f'./plots/darts/{n_chunks}_chunks/{model_type}/{parameter}/{endogenous_input}')

        ###############################
        # Preprocess Resampled Chunks #
        ###############################

        print('Read resampled series and extract chunk IDs...', file=sys.stderr)

        # Extract first x=n_chunks resampled series
        first_x_resampled = pd.read_parquet(f'./data/resampling/resample_output_{parameter}_first{n_chunks}.parquet',
                                               engine='pyarrow')

        # Extract relevant chunk IDs
        relevant_chunk_ids = list()

        for chunk_id in pd.unique(first_x_resampled.CHUNK_ID_FILLED_TH):
            current_series = first_x_resampled[first_x_resampled['CHUNK_ID_FILLED_TH'] == chunk_id]

            # At least input_chunk_length + output_chunk_length = 12 + 1 = 13 data points are required
            if len(current_series) > input_length:
                relevant_chunk_ids.append(chunk_id)

        # Separate relevant chunk IDs into IDs for training (80%) and testing (20%)
        twenty_percent = int((20 * len(relevant_chunk_ids)) / 100)
        chunk_ids_train = relevant_chunk_ids[twenty_percent:]
        chunk_ids_pred = relevant_chunk_ids[:twenty_percent]

        # Create scaler for normalization of values between 0 and 1
        scaler = Scaler()

        # Create endogenous training series as {chunkID : TimeSeries} dict
        train_series = dict()

        for chunk_id in chunk_ids_train:
            current_series = first_x_resampled[first_x_resampled['CHUNK_ID_FILLED_TH'] == chunk_id]

            train_series[chunk_id] = scaler.fit_transform(TimeSeries.from_dataframe(
                df=current_series,
                time_col='CHARTTIME',
                value_cols=[f'VITAL_PARAMTER_VALUE_{endogenous_input}_RESAMPLING'],
                freq='H'))

        print(f'#Chunks for training (endo): {len(train_series)}', file=sys.stderr)

        # Save endogenous training dict as pickle file
        train_series_f = open(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}/{endogenous_input}/'
                              f'01_train_series_endo.pickle', 'wb')
        pickle.dump(train_series, train_series_f, protocol=pickle.HIGHEST_PROTOCOL)
        train_series_f.close()

        # Create exogenous training series as {chunkID : TimeSeries} dict
        train_series_exo = dict()

        for chunk_id in chunk_ids_train:
            current_series = first_x_resampled[first_x_resampled['CHUNK_ID_FILLED_TH'] == chunk_id]

            train_series_exo[chunk_id] = scaler.fit_transform(TimeSeries.from_dataframe(
                df=current_series,
                time_col='CHARTTIME',
                value_cols=[f'VITAL_PARAMTER_VALUE_{exogenous_input}_RESAMPLING'],
                freq='H'))

        print(f'#Chunks for training (exo): {len(train_series_exo)}', file=sys.stderr)

        # Save exogenous training dict as pickle file
        train_series_exo_f = open(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}/{endogenous_input}/'
                                  f'01_train_series_exo.pickle', 'wb')
        pickle.dump(train_series_exo, train_series_exo_f, protocol=pickle.HIGHEST_PROTOCOL)
        train_series_exo_f.close()

        # Create endogenous prediction series as {chunkID : TimeSeries} dict
        pred_series = dict()

        for chunk_id in chunk_ids_pred:
            current_series = first_x_resampled[first_x_resampled['CHUNK_ID_FILLED_TH'] == chunk_id]

            # Note: scaler was already fitted on set of training series
            pred_series[chunk_id] = scaler.transform(TimeSeries.from_dataframe(
                df=current_series,
                time_col='CHARTTIME',
                value_cols=[f'VITAL_PARAMTER_VALUE_{endogenous_input}_RESAMPLING'],
                freq='H'))

        print(f'#Chunks to predict (endo): {len(pred_series)}', file=sys.stderr)

        # Save endogenous prediction dict as pickle file
        pred_series_f = open(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}/{endogenous_input}/'
                             f'02_pred_series_endo.pickle', 'wb')
        pickle.dump(pred_series, pred_series_f, protocol=pickle.HIGHEST_PROTOCOL)
        pred_series_f.close()

        # Create exogenous prediction series as {chunkID : TimeSeries} dict
        pred_series_exo = dict()

        for chunk_id in chunk_ids_pred:
            current_series = first_x_resampled[first_x_resampled['CHUNK_ID_FILLED_TH'] == chunk_id]

            # Note: scaler was already fitted on set of training series
            pred_series_exo[chunk_id] = scaler.transform(TimeSeries.from_dataframe(
                df=current_series,
                time_col='CHARTTIME',
                value_cols=[f'VITAL_PARAMTER_VALUE_{exogenous_input}_RESAMPLING'],
                freq='H'))

        print(f'#Chunks to predict (exo): {len(pred_series_exo)}', file=sys.stderr)

        # Save exogenous prediction dict as pickle file
        pred_series_exo_f = open(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}/{endogenous_input}/'
                             f'02_pred_series_exo.pickle', 'wb')
        pickle.dump(pred_series_exo, pred_series_exo_f, protocol=pickle.HIGHEST_PROTOCOL)
        pred_series_exo_f.close()

        ###################
        # Pre-train Model #
        ###################

        print('Pre-train model...', file=sys.stderr)
        param_model = model

        # Pre-train with 80% of relevant series (steady prediction set)
        param_model.fit(series=list(train_series.values()),
                        covariates=list(train_series_exo.values()),
                        verbose=True)

        # Save pre-trained model as pickle file
        pretrained_model_f = open(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}/{endogenous_input}/'
                                  f'03_pre-trained_model.pickle', 'wb')
        pickle.dump(param_model, pretrained_model_f, protocol=pickle.HIGHEST_PROTOCOL)
        pretrained_model_f.close()

        # TODO: Add missing columns (see analysis script)
        confusion_matrix_chunks = pd.DataFrame(columns=['CHUNK_ID', 'ALARM_TYPE', 'N_ITERATIONS', 'FP', 'TP', 'FN', 'TN'])

        # Iterate (at most 20) chunk IDs we want to predict
        for chunk_id in pred_series.keys():

            print(f'\n##############################\nCurrent Chunk ID: {chunk_id}\n##############################\n',
                  file=sys.stderr)

            # Load original pre-trained model for first iteration
            model_original_f = open(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}/{endogenous_input}/'
                                    f'03_pre-trained_model.pickle', 'rb')
            model_for_iteration = pickle.load(model_original_f)
            model_original_f.close()

            # Create empty DataFrame for prediction result
            # Note: Have to use DataFrame because append() function of TimeSeries do not work
            final_pred = pd.DataFrame(columns=['Time', 'Value'])

            #########################################
            # Predict Chunk via Expanding Technique #
            #########################################

            # Do not iterate whole series-to-predict because of starting length of 12 (first prediction is for time 13)
            for iteration in range(len(pred_series[chunk_id]) - input_length):

                print(f'Iteration: {iteration}', file=sys.stderr)

                # Take last pre-trained model (or original pre-trained model in first iteration)
                if iteration > 0:
                    model_last_iteration_f = open(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}/{endogenous_input}/'
                                                  f'04_pre-trained_model_{chunk_id}_{iteration - 1}.pickle', 'rb')
                    model_for_iteration = pickle.load(model_last_iteration_f)
                    model_last_iteration_f.close()

                # Predict one measurement
                current_pred = model_for_iteration.predict(
                    n=output_length,
                    series=pred_series[chunk_id][:input_length + iteration],
                    covariates=pred_series_exo[chunk_id][:input_length + iteration])

                # Save model after each iteration
                extended_model_f = open(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}/{endogenous_input}/'
                                        f'04_pre-trained_model_{chunk_id}_{iteration}.pickle', 'wb')
                pickle.dump(model_for_iteration, extended_model_f, protocol=pickle.HIGHEST_PROTOCOL)
                extended_model_f.close()

                # Rescale predicted measurement
                current_pred = scaler.inverse_transform(current_pred)

                # Add intermediate prediction result to DataFrame
                final_pred = final_pred.append({'Time': current_pred.start_time(),
                                                'Value': current_pred.first_value()},
                                               ignore_index=True)

            # Save final prediction of chunk as pickle file
            final_pred_f = open(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}/{endogenous_input}/'
                                f'05_prediction_{chunk_id}.pickle', 'wb')
            pickle.dump(final_pred, final_pred_f, protocol=pickle.HIGHEST_PROTOCOL)
            final_pred_f.close()

            #####################################
            # Fill Chunk-level Confusion Matrix #
            #####################################

            # Extract original chunk
            original_chunk = first_x_resampled[first_x_resampled['CHUNK_ID_FILLED_TH'] == chunk_id].sort_values('CHARTTIME')
            original_chunk = original_chunk[input_length:].reset_index()

            # Add boolean indicating triggered low alarm original value
            original_chunk['LOW_ALARM_TRIGGERED'] = False
            original_chunk.loc[original_chunk[f'VITAL_PARAMTER_VALUE_{endogenous_input}_RESAMPLING']
                               < original_chunk['THRESHOLD_VALUE_LOW'],
                               'LOW_ALARM_TRIGGERED'] = True

            # Add column with predicted value to chunk infos
            original_chunk['VALUE_PREDICTION'] = final_pred.Value

            # Add boolean indicating triggered low alarm for predicted value
            original_chunk['LOW_ALARM_TRIGGERED_PREDICTION'] = False
            original_chunk.loc[original_chunk['VALUE_PREDICTION']
                               < original_chunk['THRESHOLD_VALUE_LOW'],
                               'LOW_ALARM_TRIGGERED_PREDICTION'] = True

            # Get indices where booleans are false or true for low alarms
            low_triggered = set(original_chunk.index[original_chunk['LOW_ALARM_TRIGGERED']])
            low_triggered_pred = set(original_chunk.index[original_chunk['LOW_ALARM_TRIGGERED_PREDICTION']])
            low_not_triggered = set(original_chunk.index[~original_chunk['LOW_ALARM_TRIGGERED']])
            low_not_triggered_pred = set(original_chunk.index[~original_chunk['LOW_ALARM_TRIGGERED_PREDICTION']])

            # Fill confusion matrix for low threshold analysis
            confusion_matrix_chunks = confusion_matrix_chunks.append({
                'CHUNK_ID': chunk_id,
                'ALARM_TYPE': 'Low',
                'N_ITERATIONS': len(pred_series[chunk_id]) - input_length,
                # Following 4 metrics look at how many indices are shared
                'TP': len(low_triggered.intersection(low_triggered_pred)),
                'FN': len(low_triggered.intersection(low_not_triggered_pred)),
                'FP': len(low_not_triggered.intersection(low_triggered_pred)),
                'TN': len(low_not_triggered.intersection(low_not_triggered_pred))
            }, ignore_index=True)

        # Save chunk-level confusion matrix after all chunks are processed
        confusion_matrix_chunks_f = open(f'./data/darts/{n_chunks}_chunks/confusion_matrix_chunks_{model_type}_'
                                         f'{parameter}_{endogenous_input}.pickle', 'wb')
        pickle.dump(confusion_matrix_chunks, confusion_matrix_chunks_f, protocol=pickle.HIGHEST_PROTOCOL)
        confusion_matrix_chunks_f.close()

        #####################################
        # Fill Model-level Confusion Matrix #
        #####################################

        # Fill model-level confusion matrix per parameter and model type
        FP_all = confusion_matrix_chunks['FP'].sum()
        TP_all = confusion_matrix_chunks['TP'].sum()
        FN_all = confusion_matrix_chunks['FN'].sum()
        TN_all = confusion_matrix_chunks['TN'].sum()
        n_chunks_all = len(confusion_matrix_chunks)
        n_iterations_all = confusion_matrix_chunks['N_ITERATIONS'].sum()

        confusion_matrix_models = confusion_matrix_models.append({
            'ID': f'{parameter.upper()}_R_03_L',  # R = RNNModel, 03 = MIN, and L = Low
            'PARAMETER': parameter.upper(),
            'MODEL': model_type,
            'ENDOGENOUS': endogenous_input,
            'EXOGENOUS': exogenous_input,
            # TODO: improve naming (expanding with RNNModel means hourly prediction) or remove column
            'FORECAST_TYPE': 'Expanding',
            'FIRST_FORECAST': input_length + output_length,
            'ALARM_TYPE': 'Low',
            'FP': FP_all,
            'TP': TP_all,
            'FN': FN_all,
            'TN': TN_all,
            'N_CHUNKS': n_chunks_all,
            'N_ITERATIONS': n_iterations_all
        }, ignore_index=True)

# Save model-level confusion matrix after all model types and parameter are processed
# Note: adjust path name if you want to execute this script in parallel with different parameters/ model types
confusion_matrix_models_f = open(f'./data/darts/{n_chunks}_chunks/confusion_matrix_models_{endogenous_input}.pickle', 'wb')
pickle.dump(confusion_matrix_models, confusion_matrix_models_f, protocol=pickle.HIGHEST_PROTOCOL)
confusion_matrix_models_f.close()

print('\nFinished.', file=sys.stderr)
sys.stderr.close()
