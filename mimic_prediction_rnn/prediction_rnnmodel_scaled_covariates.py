"""
    PREDICTION WITH ALL RNN MODELS, ALL PARAMETERS, AND MAX AND MIN RESAMPLED CHUNKS WHICH ARE SCALED

    This script assumes that there is already the subdirectory '/RNNModel' in the directory '/data'. If you want to
    adjust which input size is taken and what parameters and models are used for the prediction, have a look at the six
    variables from line 27 to 41.

    Lastly, you have to install some packages:
    pip3 install u8darts[torch] seaborn
"""


from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
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
parameters = ['hr', 'bp', 'o2']

# Number of chunks can be 1,000 or 15,000
n_chunks = 1000

# Style can be 'all' or '20_percent'
style = 'all'

# Define input length to imitate ARIMA training size (start after this length with forecast) and predict one data point
input_length = 12
output_length = 1

###########################################
# Create Folders, Init Variables & Models #
###########################################

approach = 'RNNModel'

# Create main folder for this script
if not os.path.isdir(f'./data/{approach}/{n_chunks}_chunks'):
    os.mkdir(f'./data/{approach}/{n_chunks}_chunks')
if not os.path.isdir(f'./data/{approach}/{n_chunks}_chunks/{style}'):
    os.mkdir(f'./data/{approach}/{n_chunks}_chunks/{style}')

# Create model-level confusion matrix
confusion_matrix_models = pd.DataFrame(
    columns=['ID', 'PARAMETER', 'MODEL', 'ENDOGENOUS', 'EXOGENOUS', 'FIRST_FORECAST', 'ALARM_TYPE',
             'FP', 'TP', 'FN', 'TN', 'N_HIGH_ALARMS', 'N_LOW_ALARMS', 'N_CHUNKS', 'N_ITERATIONS'])

endogenous_input_high = 'MAX'
endogenous_input_low = 'MIN'
endogenous_input = endogenous_input_high + '_' + endogenous_input_low
exogenous_input = 'MEDIAN'

model_numbers = {
    ('RNN',     'MAX_MIN'):     '02',
    ('LSTM',    'MAX_MIN'):     '04',
    ('GRU',     'MAX_MIN'):     '06'
}

if style == 'all':
    n_windows = 5
elif style == '20_percent':
    n_windows = 1
else:
    raise ValueError('The style has to be "all" or "20_percent".')

# Note: Only use filler for now, remove after resampling script is fixed
filler = MissingValuesFiller()

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
                     batch_size=input_length)  # batch_size must be <= input_length (current bug in Darts)

    for parameter in parameters:
        print(f'\n##############################\nCurrent Parameter: {parameter.upper()}\n'
              f'##############################\n', file=sys.stderr)

        # Create sub folder for each parameter
        if not os.path.isdir(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}/{parameter}'):
            os.mkdir(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}/{parameter}')

        # Create sub folder for the input type (max or min as endogenous variable)
        if not os.path.isdir(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}/{parameter}/{endogenous_input}'):
            os.mkdir(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}/{parameter}/{endogenous_input}')

        ###############################
        # Preprocess Resampled Chunks #
        ###############################

        print('Read resampled series and extract chunks for training and prediction...', file=sys.stderr)

        # Extract first n_chunks resampled series
        resampled = pd.read_parquet(f'./data/resampling/resample_output_{parameter}_first{n_chunks}.parquet',
                                    engine='pyarrow')

        # Extract relevant chunks
        relevant_series_endo_high, relevant_series_endo_low, relevant_series_exo = dict(), dict(), dict()

        # Collect all series with minimal length
        for chunk_id in pd.unique(resampled.CHUNK_ID_FILLED_TH):
            current_series = resampled[resampled['CHUNK_ID_FILLED_TH'] == chunk_id]

            # At least input_chunk_length + output_chunk_length = 12 + 1 = 13 data points are required
            if len(current_series) > 12:
                relevant_series_endo_high[chunk_id] = filler.transform(TimeSeries.from_dataframe(
                    df=current_series,
                    time_col='CHARTTIME',
                    value_cols=[f'VITAL_PARAMTER_VALUE_{endogenous_input_high}_RESAMPLING'],
                    freq='H'))

                relevant_series_endo_low[chunk_id] = filler.transform(TimeSeries.from_dataframe(
                    df=current_series,
                    time_col='CHARTTIME',
                    value_cols=[f'VITAL_PARAMTER_VALUE_{endogenous_input_low}_RESAMPLING'],
                    freq='H'))

                relevant_series_exo[chunk_id] = filler.transform(TimeSeries.from_dataframe(
                    df=current_series,
                    time_col='CHARTTIME',
                    value_cols=[f'VITAL_PARAMTER_VALUE_{exogenous_input}_RESAMPLING'],
                    freq='H'))

        # Note: dict with relevant exogenous series contains same IDs
        relevant_chunk_ids = list(relevant_series_endo_high.keys())

        # Calculate number of chunks corresponding to 20% of chunks
        twenty_percent = int((20 * len(relevant_chunk_ids)) / 100)

        # Iterate five times different 20% of the chunks (= 5 windows) to predict all chunks
        for window_idx in range(n_windows):

            print(f'{window_idx + 1}. window\n', file=sys.stderr)

            # Extract 20% of series for prediction and catch last window to avoid ignoring chunks
            if window_idx == 4:
                pred_series_endo_high = {chunk_id: relevant_series_endo_high[chunk_id]
                                         for chunk_id in list(relevant_series_endo_high)[twenty_percent * window_idx:]}
                pred_series_endo_low = {chunk_id: relevant_series_endo_low[chunk_id]
                                        for chunk_id in list(relevant_series_endo_low)[twenty_percent * window_idx:]}
                pred_series_exo = {chunk_id: relevant_series_exo[chunk_id]
                                   for chunk_id in list(relevant_series_exo)[twenty_percent * window_idx:]}
            else:
                pred_series_endo_high = {chunk_id: relevant_series_endo_high[chunk_id]
                                         for chunk_id in list(relevant_series_endo_high)[
                                                   twenty_percent * window_idx:twenty_percent * (window_idx + 1)]}
                pred_series_endo_low = {chunk_id: relevant_series_endo_low[chunk_id]
                                        for chunk_id in list(relevant_series_endo_low)[
                                                  twenty_percent * window_idx:twenty_percent * (window_idx + 1)]}
                pred_series_exo = {chunk_id: relevant_series_exo[chunk_id]
                                   for chunk_id in list(relevant_series_exo)[
                                             twenty_percent * window_idx:twenty_percent * (window_idx + 1)]}

            # Extract 80% of series for training
            train_series_endo_high = {chunk_id: relevant_series_endo_high[chunk_id] for chunk_id in relevant_chunk_ids
                                      if chunk_id not in list(pred_series_endo_high.keys())}
            train_series_endo_low = {chunk_id: relevant_series_endo_low[chunk_id] for chunk_id in relevant_chunk_ids
                                     if chunk_id not in list(pred_series_endo_low.keys())}
            train_series_exo = {chunk_id: relevant_series_exo[chunk_id] for chunk_id in relevant_chunk_ids
                                if chunk_id not in list(pred_series_exo.keys())}

            # Define and fit scalers for training and prediction set
            train_scaler, pred_scaler = Scaler(), Scaler()

            # Fit both scalers
            train_scaler = train_scaler.fit(list(train_series_endo_high.values()), list(train_series_endo_low.values()),
                                            list(train_series_exo.values()))
            pred_scaler = pred_scaler.fit(list(pred_series_endo_high.values()), list(pred_series_endo_low.values()),
                                          list(pred_series_exo.values()))

            # Normalize values
            for chunk_id in train_series_endo_high.keys():
                train_series_endo_high[chunk_id] = train_scaler.transform(train_series_endo_high[chunk_id])
                train_series_endo_low[chunk_id] = train_scaler.transform(train_series_endo_low[chunk_id])
                train_series_exo[chunk_id] = train_scaler.transform(train_series_exo[chunk_id])

            for chunk_id in pred_series_endo_high.keys():
                pred_series_endo_high[chunk_id] = pred_scaler.transform(pred_series_endo_high[chunk_id])
                pred_series_endo_low[chunk_id] = pred_scaler.transform(pred_series_endo_low[chunk_id])
                pred_series_exo[chunk_id] = pred_scaler.transform(pred_series_exo[chunk_id])

            # Note: dicts with training and prediction chunks of other series have the same lengths
            print(f'#Chunks for training: {len(train_series_endo_high)}', file=sys.stderr)
            print(f'#Chunks for prediction: {len(pred_series_endo_high)}', file=sys.stderr)

            # Save endogenous training dicts as pickle files
            train_series_endo_high_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}/{parameter}/'
                                            f'{endogenous_input}/01_train_series_endo_high_scaled_window{window_idx}'
                                            f'.pickle', 'wb')
            pickle.dump(train_series_endo_high, train_series_endo_high_f, protocol=pickle.HIGHEST_PROTOCOL)
            train_series_endo_high_f.close()

            train_series_endo_low_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}/{parameter}/'
                                           f'{endogenous_input}/01_train_series_endo_low_scaled_window{window_idx}'
                                           f'.pickle', 'wb')
            pickle.dump(train_series_endo_low, train_series_endo_low_f, protocol=pickle.HIGHEST_PROTOCOL)
            train_series_endo_low_f.close()

            # Save exogenous training dict as pickle file
            train_series_exo_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}/{parameter}/'
                                      f'{endogenous_input}/01_train_series_exo_scaled_window{window_idx}.pickle', 'wb')
            pickle.dump(train_series_exo, train_series_exo_f, protocol=pickle.HIGHEST_PROTOCOL)
            train_series_exo_f.close()

            # Save endogenous prediction dicts as pickle files
            pred_series_endo_high_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}/{parameter}/'
                                           f'{endogenous_input}/02_pred_series_endo_high_scaled_window{window_idx}'
                                           f'.pickle', 'wb')
            pickle.dump(pred_series_endo_high, pred_series_endo_high_f, protocol=pickle.HIGHEST_PROTOCOL)
            pred_series_endo_high_f.close()

            pred_series_endo_low_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}/{parameter}/'
                                          f'{endogenous_input}/02_pred_series_endo_low_scaled_window{window_idx}'
                                          f'.pickle', 'wb')
            pickle.dump(pred_series_endo_low, pred_series_endo_low_f, protocol=pickle.HIGHEST_PROTOCOL)
            pred_series_endo_low_f.close()

            # Save exogenous prediction dict as pickle file
            pred_series_exo_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}/{parameter}/'
                                     f'{endogenous_input}/02_pred_series_exo_scaled_window{window_idx}.pickle', 'wb')
            pickle.dump(pred_series_exo, pred_series_exo_f, protocol=pickle.HIGHEST_PROTOCOL)
            pred_series_exo_f.close()

            # Save scaler for chunks to predict as pickle file
            pred_scaler_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}/{parameter}/'
                                 f'{endogenous_input}/03_pred_scaler_window{window_idx}.pickle', 'wb')
            pickle.dump(pred_scaler, pred_scaler_f, protocol=pickle.HIGHEST_PROTOCOL)
            pred_scaler_f.close()

            ###################
            # Pre-train Model #
            ###################

            print('Pre-train model for high alarm forecasting...', file=sys.stderr)

            # Pre-train with 80% of relevant MAX series (steady training set)
            param_model_high = model
            param_model_high.fit(series=list(train_series_endo_high.values()),
                                 covariates=list(train_series_exo.values()),
                                 verbose=True)

            # Save pre-trained model as pickle file
            param_model_high_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}/{parameter}/'
                                      f'{endogenous_input}/04_pre-trained_model_high_scaled_window{window_idx}.pickle',
                                      'wb')
            pickle.dump(param_model_high, param_model_high_f, protocol=pickle.HIGHEST_PROTOCOL)
            param_model_high_f.close()

            print('Pre-train model for low alarm forecasting...', file=sys.stderr)

            # Pre-train with 80% of relevant MIN series (steady training set)
            param_model_low = model
            param_model_low.fit(series=list(train_series_endo_low.values()),
                                covariates=list(train_series_exo.values()),
                                verbose=True)

            # Save pre-trained model as pickle file
            param_model_low_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}/{parameter}/'
                                     f'{endogenous_input}/04_pre-trained_model_low_scaled_window{window_idx}.pickle',
                                     'wb')
            pickle.dump(param_model_low, param_model_low_f, protocol=pickle.HIGHEST_PROTOCOL)
            param_model_low_f.close()

            confusion_matrix_chunks = pd.DataFrame(
                columns=['CHUNK_ID', 'PARAMETER', 'MODEL', 'ENDOGENOUS', 'EXOGENOUS', 'FIRST_FORECAST', 'ALARM_TYPE',
                         'FP', 'TP', 'FN', 'TN', 'N_HIGH_ALARMS', 'N_LOW_ALARMS', 'N_ITERATIONS', ])

            # Iterate chunk IDs we want to predict
            for chunk_id in pred_series_endo_high.keys():

                print(f'\n##############################\nCurrent Chunk ID: {chunk_id}\n##############################\n',
                      file=sys.stderr)

                ##############################
                # Hourly Predict Chunk (MAX) #
                ##############################

                print(f'High alarm forecasting:\n', file=sys.stderr)

                # Load original pre-trained model
                model_original_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}/{parameter}/'
                                        f'{endogenous_input}/04_pre-trained_model_high_scaled_window{window_idx}'
                                        f'.pickle', 'rb')
                model_for_iterations_high = pickle.load(model_original_f)
                model_original_f.close()

                # Create empty DataFrame for prediction result
                # Note: Have to use DataFrame because append() function of TimeSeries do not work
                final_pred_high = pd.DataFrame(columns=['Time', 'Value'])

                # Do not iterate whole series-to-predict because of start length of 12 (first prediction is for time 13)
                for iteration in range(len(pred_series_endo_high[chunk_id]) - input_length):

                    print(f'Iteration: {iteration}', file=sys.stderr)

                    # Predict one measurement
                    current_pred_high = model_for_iterations_high.predict(
                        n=output_length,
                        series=pred_series_endo_high[chunk_id][:input_length + iteration],
                        covariates=pred_series_exo[chunk_id][:input_length + iteration])

                    # Rescale predicted measurement
                    current_pred_high = pred_scaler.inverse_transform(current_pred_high)

                    # Add intermediate prediction result to DataFrame
                    final_pred_high = final_pred_high.append({'Time': current_pred_high.start_time(),
                                                              'Value': current_pred_high.first_value()},
                                                             ignore_index=True)

                # Save final prediction of chunk as pickle file
                final_pred_high_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}/{parameter}/'
                                         f'{endogenous_input}/05_prediction_{chunk_id}_high_scaled_window{window_idx}'
                                         f'.pickle', 'wb')
                pickle.dump(final_pred_high, final_pred_high_f, protocol=pickle.HIGHEST_PROTOCOL)
                final_pred_high_f.close()

                ##############################
                # Hourly Predict Chunk (MIN) #
                ##############################

                print(f'Low alarm forecasting:\n', file=sys.stderr)

                # Load original pre-trained model
                model_original_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}/{parameter}/'
                                        f'{endogenous_input}/04_pre-trained_model_low_scaled_window{window_idx}.pickle',
                                        'rb')
                model_for_iterations_low = pickle.load(model_original_f)
                model_original_f.close()

                # Create empty DataFrame for prediction result
                # Note: Have to use DataFrame because append() function of TimeSeries do not work
                final_pred_low = pd.DataFrame(columns=['Time', 'Value'])

                # Do not iterate whole series-to-predict because of start length of 12 (first prediction is for time 13)
                for iteration in range(len(pred_series_endo_low[chunk_id]) - input_length):

                    print(f'Iteration: {iteration}', file=sys.stderr)

                    # Predict one measurement
                    current_pred_low = model_for_iterations_low.predict(
                        n=output_length,
                        series=pred_series_endo_low[chunk_id][:input_length + iteration],
                        covariates=pred_series_exo[chunk_id][:input_length + iteration])

                    # Rescale predicted measurement
                    current_pred_low = pred_scaler.inverse_transform(current_pred_low)

                    # Add intermediate prediction result to DataFrame
                    final_pred_low = final_pred_low.append({'Time': current_pred_low.start_time(),
                                                            'Value': current_pred_low.first_value()},
                                                           ignore_index=True)

                # Save final prediction of chunk as pickle file
                final_pred_low_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/{model_type}/{parameter}/'
                                        f'{endogenous_input}/05_prediction_{chunk_id}_low_scaled_window{window_idx}'
                                        f'.pickle', 'wb')
                pickle.dump(final_pred_low, final_pred_low_f, protocol=pickle.HIGHEST_PROTOCOL)
                final_pred_low_f.close()

                #####################################
                # Fill Chunk-level Confusion Matrix #
                #####################################

                # Extract original chunk
                original_chunk = resampled[resampled['CHUNK_ID_FILLED_TH'] == chunk_id].sort_values('CHARTTIME')
                original_chunk = original_chunk[input_length:].reset_index()

                # Add boolean indicating triggered high alarm for original value
                original_chunk['HIGH_ALARM_TRIGGERED'] = False
                original_chunk.loc[original_chunk[f'VITAL_PARAMTER_VALUE_{endogenous_input_high}_RESAMPLING']
                                   > original_chunk['THRESHOLD_VALUE_HIGH'],
                                   'HIGH_ALARM_TRIGGERED'] = True

                # Add boolean indicating triggered low alarm original value
                original_chunk['LOW_ALARM_TRIGGERED'] = False
                original_chunk.loc[original_chunk[f'VITAL_PARAMTER_VALUE_{endogenous_input_low}_RESAMPLING']
                                   < original_chunk['THRESHOLD_VALUE_LOW'],
                                   'LOW_ALARM_TRIGGERED'] = True

                # Add columns with predicted values to chunk
                original_chunk['VALUE_PREDICTION_HIGH'] = final_pred_high.Value
                original_chunk['VALUE_PREDICTION_LOW'] = final_pred_low.Value

                # Add boolean indicating triggered high alarm for predicted value
                original_chunk['HIGH_ALARM_TRIGGERED_PREDICTION'] = False
                original_chunk.loc[original_chunk['VALUE_PREDICTION_HIGH']
                                   > original_chunk['THRESHOLD_VALUE_HIGH'],
                                   'HIGH_ALARM_TRIGGERED_PREDICTION'] = True

                # Add boolean indicating triggered low alarm for predicted value
                original_chunk['LOW_ALARM_TRIGGERED_PREDICTION'] = False
                original_chunk.loc[original_chunk['VALUE_PREDICTION_LOW']
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
                    'VERSION': 'scaled',
                    'PARAMETER': parameter.upper(),
                    'MODEL': model_type,
                    'ENDOGENOUS': endogenous_input_high,
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
                    'N_ITERATIONS': len(pred_series_endo_high[chunk_id]) - input_length
                }, ignore_index=True)

                # Fill confusion matrix for low threshold analysis
                confusion_matrix_chunks = confusion_matrix_chunks.append({
                    'CHUNK_ID': chunk_id,
                    'VERSION': 'scaled',
                    'PARAMETER': parameter.upper(),
                    'MODEL': model_type,
                    'ENDOGENOUS': endogenous_input_low,
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
                    'N_ITERATIONS': len(pred_series_endo_low[chunk_id]) - input_length
                }, ignore_index=True)

            # Save chunk-level confusion matrix after all chunks are processed
            confusion_matrix_chunks_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/confusion_matrix_chunks_'
                                             f'{model_type}_{parameter}_{endogenous_input}_scaled_window{window_idx}'
                                             f'.pickle', 'wb')
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
                    file.startswith(f'confusion_matrix_chunks_{model_type}_{parameter}_{endogenous_input}_scaled'):
                current_chunk_matrix_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/{file}', 'rb')
                current_chunk_matrix = pickle.load(current_chunk_matrix_f)
                current_chunk_matrix_f.close()

                confusion_matrix_chunks_concat = pd.concat([confusion_matrix_chunks_concat, current_chunk_matrix])

        confusion_matrix_chunks_concat.reset_index(inplace=True, drop=True)

        # Fill model-level confusion matrix per parameter and model type (HIGH alarm forecasting)
        confusion_matrix_chunks_concat_high = \
            confusion_matrix_chunks_concat[confusion_matrix_chunks_concat['ALARM_TYPE'] == 'High']

        confusion_matrix_models = confusion_matrix_models.append({
            # R = RNNModel, model_numbers = {01, ..., 12} and H = High
            'ID': f'{parameter.upper()}_R_{model_numbers[model_type, endogenous_input]}_H',
            'VERSION': 'scaled',
            'PARAMETER': parameter.upper(),
            'MODEL': model_type,
            'ENDOGENOUS': endogenous_input_high,
            'EXOGENOUS': exogenous_input,
            'FIRST_FORECAST': input_length + output_length,
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
            # R = RNNModel, model_numbers = {01, ..., 12} and L = Low
            'ID': f'{parameter.upper()}_R_{model_numbers[model_type, endogenous_input]}_L',
            'VERSION': 'scaled',
            'PARAMETER': parameter.upper(),
            'MODEL': model_type,
            'ENDOGENOUS': endogenous_input_low,
            'EXOGENOUS': exogenous_input,
            'FIRST_FORECAST': input_length + output_length,
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

# Save model-level confusion matrix after all model types and parameter are processed
# Note: adjust path name if you want to execute this script in parallel with different parameters/ model types
confusion_matrix_models_f = open(f'./data/{approach}/{n_chunks}_chunks/{style}/confusion_matrix_models_scaled_'
                                 f'{endogenous_input}.pickle', 'wb')
pickle.dump(confusion_matrix_models, confusion_matrix_models_f, protocol=pickle.HIGHEST_PROTOCOL)
confusion_matrix_models_f.close()

print('\nFinished.', file=sys.stderr)
sys.stderr.close()
