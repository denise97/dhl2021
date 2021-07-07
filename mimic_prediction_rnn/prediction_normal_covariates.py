"""
    PREDICTION WITH ALL RNN MODELS, ALL PARAMETERS, AND MAX OR MIN RESAMPLED CHUNKS

    This script assumes that there is already the subdirectory '/darts' in the directory '/data'. If you want to adjust
    which input is taken and what parameters and models are used for the prediction, have a look at the three variables
    from line 26 to 40.

    Lastly, you have to install some packages:
    pip3 install u8darts[torch] seaborn
"""


from darts import TimeSeries
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

# Endogenous input for prediction with covariates can be MIN or MAX
endogenous_input = 'MAX'

# Number of chunks can be 1,000 or 15,000
n_chunks = 1000

# Define input length to imitate ARIMA training size (start after this length with forecast) and predict one data point
input_length = 12
output_length = 1

##################
# Create Folders #
##################

# Create main folder for this script
if not os.path.isdir(f'./data/darts/{n_chunks}_chunks'):
    os.mkdir(f'./data/darts/{n_chunks}_chunks')

# Create model-level confusion matrix
confusion_matrix_models = pd.DataFrame(
    columns=['ID', 'PARAMETER', 'MODEL', 'ENDOGENOUS', 'EXOGENOUS', 'FORECAST_TYPE', 'FIRST_FORECAST', 'ALARM_TYPE',
             'FP', 'TP', 'FN', 'TN', 'N_CHUNKS', 'N_ITERATIONS'])

# Exogenous input is always median resampled for prediction with covariates
exogenous_input = 'MEDIAN'

model_numbers = {
    ('RNN',     'MAX'):     '11',
    ('RNN',     'MIN'):     '12',
    ('LSTM',    'MAX'):     '14',
    ('LSTM',    'MIN'):     '15',
    ('GRU',     'MAX'):     '17',
    ('GRU',     'MIN'):     '18'
}

for model_type in model_types:
    print(f'\n##############################\nCurrent Model Type: {model_type}\n##############################\n',
          file=sys.stderr)

    # Create sub folder for each model type
    if not os.path.isdir(f'./data/darts/{n_chunks}_chunks/{model_type}'):
        os.mkdir(f'./data/darts/{n_chunks}_chunks/{model_type}')

    # Create model per model type
    model = RNNModel(model=model_type,
                     input_chunk_length=input_length,
                     output_chunk_length=output_length,
                     batch_size=input_length)  # batch_size must be <= input_length (current bug in Darts)

    # TODO: NaN values in predictions of O2 series with all model types
    for parameter in parameters:
        print(f'\n##############################\nCurrent Parameter: {parameter.upper()}\n'
              f'##############################\n', file=sys.stderr)

        # Create sub folder for each parameter
        if not os.path.isdir(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}'):
            os.mkdir(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}')

        # Create sub folder for the input type (max or min as endogenous variable)
        if not os.path.isdir(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}/{endogenous_input}'):
            os.mkdir(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}/{endogenous_input}')

        ###############################
        # Preprocess Resampled Chunks #
        ###############################

        print('Read resampled series and extract chunks for training and prediction...', file=sys.stderr)

        # Extract first x=n_chunks resampled series
        resampled = pd.read_parquet(f'./data/resampling/resample_output_{parameter}_first{n_chunks}.parquet',
                                    engine='pyarrow')

        # Extract relevant chunks
        relevant_series_endo, relevant_series_exo = dict(), dict()

        # Collect all series with minimal length
        for chunk_id in pd.unique(resampled.CHUNK_ID_FILLED_TH):
            current_series = resampled[resampled['CHUNK_ID_FILLED_TH'] == chunk_id]

            # At least input_chunk_length + output_chunk_length = 12 + 1 = 13 data points are required
            if len(current_series) > 12:
                relevant_series_endo[chunk_id] = TimeSeries.from_dataframe(
                    df=current_series,
                    time_col='CHARTTIME',
                    value_cols=[f'VITAL_PARAMTER_VALUE_{endogenous_input}_RESAMPLING'],
                    freq='H')

                relevant_series_exo[chunk_id] = TimeSeries.from_dataframe(
                    df=current_series,
                    time_col='CHARTTIME',
                    value_cols=[f'VITAL_PARAMTER_VALUE_{exogenous_input}_RESAMPLING'],
                    freq='H')

        # Note: dict with relevant exogenous series contains same IDs
        relevant_chunk_ids = list(relevant_series_endo.keys())

        # Calculate number of chunks corresponding to 20% of chunks
        twenty_percent = int((20 * len(relevant_chunk_ids)) / 100)

        # Extract first 20% of endogenous and exogenous series for prediction
        pred_series_endo = {k: relevant_series_endo[k] for k in list(relevant_series_endo)[:twenty_percent]}
        pred_series_exo = {k: relevant_series_exo[k] for k in list(relevant_series_exo)[:twenty_percent]}

        # Extract last 80% of endogenous and exogenous series for training
        train_series_endo = {k: relevant_series_endo[k] for k in list(relevant_series_endo)[twenty_percent:]}
        train_series_exo = {k: relevant_series_exo[k] for k in list(relevant_series_exo)[twenty_percent:]}

        # Note: dicts with training and prediction chunks of exogenous series have the same lengths
        print(f'#Chunks for training: {len(train_series_endo)}', file=sys.stderr)
        print(f'#Chunks to prediction: {len(pred_series_endo)}', file=sys.stderr)

        # Save exogenous training dict as pickle file
        train_series_exo_f = open(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}/{endogenous_input}/'
                                  f'01_train_series_exo_normal.pickle', 'wb')
        pickle.dump(train_series_exo, train_series_exo_f, protocol=pickle.HIGHEST_PROTOCOL)
        train_series_exo_f.close()

        # Save exogenous prediction dict as pickle file
        pred_series_exo_f = open(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}/{endogenous_input}/'
                                 f'02_pred_series_exo_normal.pickle', 'wb')
        pickle.dump(pred_series_exo, pred_series_exo_f, protocol=pickle.HIGHEST_PROTOCOL)
        pred_series_exo_f.close()

        # Save endogenous training dict as pickle file
        train_series_f = open(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}/{endogenous_input}/'
                              f'01_train_series_endo_normal.pickle', 'wb')
        pickle.dump(train_series_endo, train_series_f, protocol=pickle.HIGHEST_PROTOCOL)
        train_series_f.close()

        # Save endogenous prediction dict as pickle file
        pred_series_f = open(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}/{endogenous_input}/'
                             f'02_pred_series_endo_normal.pickle', 'wb')
        pickle.dump(pred_series_endo, pred_series_f, protocol=pickle.HIGHEST_PROTOCOL)
        pred_series_f.close()

        ###################
        # Pre-train Model #
        ###################

        print('Pre-train model...', file=sys.stderr)
        param_model = model

        # Pre-train with 80% of relevant series (steady training set)
        param_model.fit(series=list(train_series_endo.values()),
                        covariates=list(train_series_exo.values()),
                        verbose=True)

        # Save pre-trained model as pickle file
        pretrained_model_f = open(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}/{endogenous_input}/'
                                  f'03_pre-trained_model_normal.pickle', 'wb')
        pickle.dump(param_model, pretrained_model_f, protocol=pickle.HIGHEST_PROTOCOL)
        pretrained_model_f.close()

        confusion_matrix_chunks = pd.DataFrame(
            columns=['CHUNK_ID', 'PARAMETER', 'MODEL', 'ENDOGENOUS', 'EXOGENOUS', 'FIRST_FORECAST', 'ALARM_TYPE', 'FP',
                     'TP', 'FN', 'TN', 'N_HIGH_ALARMS', 'N_LOW_ALARMS', 'N_ITERATIONS', ])

        # Iterate chunk IDs we want to predict
        for chunk_id in pred_series_endo.keys():

            print(f'\n##############################\nCurrent Chunk ID: {chunk_id}\n##############################\n',
                  file=sys.stderr)

            # Load original pre-trained model for first iteration
            model_original_f = open(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}/{endogenous_input}/'
                                    f'03_pre-trained_model_normal.pickle', 'rb')
            model_for_iteration = pickle.load(model_original_f)
            model_original_f.close()

            # Create empty DataFrame for prediction result
            # Note: Have to use DataFrame because append() function of TimeSeries do not work
            final_pred = pd.DataFrame(columns=['Time', 'Value'])

            #########################################
            # Predict Chunk via Expanding Technique #
            #########################################

            # Do not iterate whole series-to-predict because of starting length of 12 (first prediction is for time 13)
            for iteration in range(len(pred_series_endo[chunk_id]) - input_length):

                print(f'Iteration: {iteration}', file=sys.stderr)

                # Take last pre-trained model (or original pre-trained model in first iteration)
                if iteration > 0:
                    model_last_iteration_f = open(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}/'
                                                  f'{endogenous_input}/04_pre-trained_model_{chunk_id}_{iteration - 1}'
                                                  f'_normal.pickle', 'rb')
                    model_for_iteration = pickle.load(model_last_iteration_f)
                    model_last_iteration_f.close()

                # Predict one measurement
                current_pred = model_for_iteration.predict(
                    n=output_length,
                    series=pred_series_endo[chunk_id][:input_length + iteration],
                    covariates=pred_series_exo[chunk_id][:input_length + iteration])

                # Save model after each iteration
                extended_model_f = open(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}/{endogenous_input}/'
                                        f'04_pre-trained_model_{chunk_id}_{iteration}_normal.pickle', 'wb')
                pickle.dump(model_for_iteration, extended_model_f, protocol=pickle.HIGHEST_PROTOCOL)
                extended_model_f.close()

                # Add intermediate prediction result to DataFrame
                final_pred = final_pred.append({'Time': current_pred.start_time(),
                                                'Value': current_pred.first_value()},
                                               ignore_index=True)

            # Save final prediction of chunk as pickle file
            final_pred_f = open(f'./data/darts/{n_chunks}_chunks/{model_type}/{parameter}/{endogenous_input}/'
                                f'05_prediction_{chunk_id}_normal.pickle', 'wb')
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
            original_chunk.loc[original_chunk[f'VITAL_PARAMTER_VALUE_{endogenous_input}_RESAMPLING']
                               > original_chunk['THRESHOLD_VALUE_HIGH'],
                               'HIGH_ALARM_TRIGGERED'] = True

            # Add boolean indicating triggered low alarm original value
            original_chunk['LOW_ALARM_TRIGGERED'] = False
            original_chunk.loc[original_chunk[f'VITAL_PARAMTER_VALUE_{endogenous_input}_RESAMPLING']
                               < original_chunk['THRESHOLD_VALUE_LOW'],
                               'LOW_ALARM_TRIGGERED'] = True

            # Add column with predicted value to chunk infos
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
                'PARAMETER': parameter.upper(),
                'MODEL': model_type,
                'ENDOGENOUS': endogenous_input,
                'EXOGENOUS': exogenous_input,
                'FIRST_FORECAST': input_length + output_length,
                'ALARM_TYPE': 'High',
                # Following 4 metrics look at how many indices are shared
                'TP': len(high_triggered.intersection(high_triggered_pred)),
                'FN': len(high_triggered.intersection(high_not_triggered_pred)),
                'FP': len(high_not_triggered.intersection(high_triggered_pred)),
                'TN': len(high_not_triggered.intersection(high_not_triggered_pred)),
                'N_HIGH_ALARMS': len(high_triggered),
                'N_LOW_ALARMS': len(low_triggered),
                'N_ITERATIONS': len(pred_series_endo[chunk_id]) - input_length
            }, ignore_index=True)

            # Fill confusion matrix for low threshold analysis
            confusion_matrix_chunks = confusion_matrix_chunks.append({
                'CHUNK_ID': chunk_id,
                'PARAMETER': parameter.upper(),
                'MODEL': model_type,
                'ENDOGENOUS': endogenous_input,
                'EXOGENOUS': exogenous_input,
                'FIRST_FORECAST': input_length + output_length,
                'ALARM_TYPE': 'Low',
                # Following 4 metrics look at how many indices are shared
                'TP': len(low_triggered.intersection(low_triggered_pred)),
                'FN': len(low_triggered.intersection(low_not_triggered_pred)),
                'FP': len(low_not_triggered.intersection(low_triggered_pred)),
                'TN': len(low_not_triggered.intersection(low_not_triggered_pred)),
                'N_HIGH_ALARMS': len(high_triggered),
                'N_LOW_ALARMS': len(low_triggered),
                'N_ITERATIONS': len(pred_series_endo[chunk_id]) - input_length
            }, ignore_index=True)

        # Save chunk-level confusion matrix after all chunks are processed
        confusion_matrix_chunks_f = open(f'./data/darts/{n_chunks}_chunks/confusion_matrix_chunks_{model_type}_'
                                         f'{parameter}_{endogenous_input}_normal.pickle', 'wb')
        pickle.dump(confusion_matrix_chunks, confusion_matrix_chunks_f, protocol=pickle.HIGHEST_PROTOCOL)
        confusion_matrix_chunks_f.close()

        #####################################
        # Fill Model-level Confusion Matrix #
        #####################################

        confusion_matrix_chunks_high = confusion_matrix_chunks[confusion_matrix_chunks['ALARM_TYPE'] == 'HIGH']

        confusion_matrix_models = confusion_matrix_models.append({
            # R = RNNModel, model_number = {10, ..., 19} and H = High
            'ID': f'{parameter.upper()}_R_{model_numbers[model_type, endogenous_input]}_H',
            'PARAMETER': parameter.upper(),
            'MODEL': model_type,
            'ENDOGENOUS': endogenous_input,
            'EXOGENOUS': exogenous_input,
            'FIRST_FORECAST': input_length + output_length,
            'ALARM_TYPE': 'High',
            'FP': confusion_matrix_chunks_high['FP'].sum(),
            'TP': confusion_matrix_chunks_high['TP'].sum(),
            'FN': confusion_matrix_chunks_high['FN'].sum(),
            'TN': confusion_matrix_chunks_high['TN'].sum(),
            'N_HIGH_ALARMS': confusion_matrix_chunks_high['N_HIGH_ALARMS'].sum(),
            'N_LOW_ALARMS': confusion_matrix_chunks_high['N_LOW_ALARMS'].sum(),
            'N_CHUNKS': len(confusion_matrix_chunks_high),
            'N_ITERATIONS': confusion_matrix_chunks_high['N_ITERATIONS'].sum()
        }, ignore_index=True)

        # Fill model-level confusion matrix per parameter and model type (LOW alarm forecasting)
        confusion_matrix_chunks_low = confusion_matrix_chunks[confusion_matrix_chunks['ALARM_TYPE'] == 'LOW']

        confusion_matrix_models = confusion_matrix_models.append({
            # R = RNNModel, model_number = {10, ..., 19} and L = Low
            'ID': f'{parameter.upper()}_R_{model_numbers[model_type, endogenous_input]}_L',
            'PARAMETER': parameter.upper(),
            'MODEL': model_type,
            'ENDOGENOUS': endogenous_input,
            'EXOGENOUS': exogenous_input,
            'FIRST_FORECAST': input_length + output_length,
            'ALARM_TYPE': 'Low',
            'FP': confusion_matrix_chunks_low['FP'].sum(),
            'TP': confusion_matrix_chunks_low['TP'].sum(),
            'FN': confusion_matrix_chunks_low['FN'].sum(),
            'TN': confusion_matrix_chunks_low['TN'].sum(),
            'N_HIGH_ALARMS': confusion_matrix_chunks_low['N_HIGH_ALARMS'].sum(),
            'N_LOW_ALARMS': confusion_matrix_chunks_low['N_LOW_ALARMS'].sum(),
            'N_CHUNKS': len(confusion_matrix_chunks_low),
            'N_ITERATIONS': confusion_matrix_chunks_low['N_ITERATIONS'].sum()
        }, ignore_index=True)

# Save model-level confusion matrix after all model types and parameter are processed
# Note: adjust path name if you want to execute this script in parallel with different parameters/ model types
confusion_matrix_models_f = open(f'./data/darts/{n_chunks}_chunks/confusion_matrix_models_normal_{endogenous_input}'
                                 f'.pickle', 'wb')
pickle.dump(confusion_matrix_models, confusion_matrix_models_f, protocol=pickle.HIGHEST_PROTOCOL)
confusion_matrix_models_f.close()

print('\nFinished.', file=sys.stderr)
sys.stderr.close()
