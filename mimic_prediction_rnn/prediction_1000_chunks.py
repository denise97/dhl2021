"""
    PREDICTION WITH ALL 3 RNN MODELS AND MEDIAN VALUE FOR FIRST 1000 RESAMPLED CHUNKS OF HR AND O2 PARAMETER SERIES

    This script assumes that there is already the subdirectory '/darts' in the directories '/plots' and '/data'.

    Lastly, you have to install some packages:
    pip3 install u8darts[torch] seaborn
"""


from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mse
from darts.models import RNNModel

import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import seaborn as sns
import sys


# Create sub folders
if not os.path.isdir('./data/darts/1000_chunks'):
    os.mkdir('./data/darts/1000_chunks')
if not os.path.isdir('./plots/darts/1000_chunks'):
    os.mkdir('./plots/darts/1000_chunks')

# Create scaler for normalization of values between 0 and 1
scaler = Scaler()

for model_type in ['RNN', 'LSTM', 'GRU']:
    print(f'#################################\nCurrent Model Type: {model_type}\n#################################',
          file=sys.stderr)

    # Create sub folders
    if not os.path.isdir(f'./data/darts/1000_chunks/{model_type}'):
        os.mkdir(f'./data/darts/1000_chunks/{model_type}')
    if not os.path.isdir(f'./plots/darts/1000_chunks/{model_type}'):
        os.mkdir(f'./plots/darts/1000_chunks/{model_type}')

    # Create model per model type
    model = RNNModel(model=model_type,
                     input_chunk_length=12,
                     output_chunk_length=1)

    # TODO: division by zero error for NBPs when fitting (happens with all 3 model types)
    for parameter in ['hr', 'o2']:
        print(f'#################################\nCurrent Parameter: {parameter.upper()}\n#################################',
              file=sys.stderr)

        # Create sub folders
        if not os.path.isdir(f'./data/darts/1000_chunks/{model_type}/{parameter}'):
            os.mkdir(f'./data/darts/1000_chunks/{model_type}/{parameter}')
        if not os.path.isdir(f'./plots/darts/1000_chunks/{model_type}/{parameter}'):
            os.mkdir(f'./plots/darts/1000_chunks/{model_type}/{parameter}')

        print('Read resampled series and extract chunk IDs...', file=sys.stderr)

        # Extract first 1000 resampled series
        first_1000_resampled = pd.read_parquet(f'./data/resampling/resample_output_{parameter}_first1000.parquet',
                                               engine='pyarrow')

        # Extract chunk IDs for prediction (20) and training (980)
        chunk_ids_pred = pd.unique(first_1000_resampled.CHUNK_ID_FILLED_TH)[:20]
        chunk_ids_train = pd.unique(first_1000_resampled.CHUNK_ID_FILLED_TH)[20:]

        # Create training series as {chunkID : TimeSeries} dict
        train_series = dict()

        for chunk_id in chunk_ids_train:
            current_series = first_1000_resampled[first_1000_resampled['CHUNK_ID_FILLED_TH'] == chunk_id]

            # At least input_chunk_length + output_chunk_length = 12 + 1 = 13 data points are required
            if len(current_series) < 13:
                continue

            train_series[chunk_id] = scaler.fit_transform(TimeSeries.from_dataframe(
                df=current_series,
                time_col='CHARTTIME',
                value_cols=['VITAL_PARAMTER_VALUE_MEDIAN_RESAMPLING'],
                freq='H'))

        print(f'#Chunks for training: {len(train_series)}', file=sys.stderr)

        # Save training dict as pickle file
        train_series_f = open(f'./data/darts/1000_chunks/{model_type}/{parameter}/01_train_series.pickle', 'wb')
        pickle.dump(train_series, train_series_f, protocol=pickle.HIGHEST_PROTOCOL)
        train_series_f.close()

        # Create prediction series as {chunkID : TimeSeries} dict
        pred_series = dict()

        for chunk_id in chunk_ids_pred:
            current_series = first_1000_resampled[first_1000_resampled['CHUNK_ID_FILLED_TH'] == chunk_id]

            # At least input_chunk_length + output_chunk_length = 12 + 1 = 13 data points are required
            if len(current_series) < 13:
                continue

            pred_series[chunk_id] = scaler.fit_transform(TimeSeries.from_dataframe(
                df=current_series,
                time_col='CHARTTIME',
                value_cols=['VITAL_PARAMTER_VALUE_MEDIAN_RESAMPLING'],
                freq='H'))

        print(f'#Chunks to predict: {len(pred_series)}', file=sys.stderr)

        # Save prediction dict as pickle file
        pred_series_f = open(f'./data/darts/1000_chunks/{model_type}/{parameter}/02_pred_series.pickle', 'wb')
        pickle.dump(pred_series, pred_series_f, protocol=pickle.HIGHEST_PROTOCOL)
        pred_series_f.close()

        print('Pre-train model...', file=sys.stderr)
        param_model = model

        # Pre-train with at most 980 series
        param_model.fit(series=list(pred_series.values()))

        # Save pre-trained model as pickle file
        pretrained_model_f = open(f'./data/darts/1000_chunks/{model_type}/{parameter}/03_pre-trained_model.pickle', 'wb')
        pickle.dump(param_model, pretrained_model_f, protocol=pickle.HIGHEST_PROTOCOL)
        pretrained_model_f.close()

        # Iterate (at most 20) chunk IDs we want to predict
        for chunk_id in pred_series.keys():

            print(f'#################################\nCurrent Chunk ID: {chunk_id}\n#################################',
                  file=sys.stderr)

            # Load original pre-trained model for first iteration
            model_original_f = open(f'./data/darts/1000_chunks/{model_type}/{parameter}/03_pre-trained_model.pickle', 'rb')
            model_for_iteration = pickle.load(model_original_f)
            model_original_f.close()

            # Create empty DataFrame for prediction result
            # Note: Have to use DataFrame because append() function of TimeSeries do not work
            final_pred = pd.DataFrame(columns=['Time', 'Value'])

            # Do not iterate whole series-to-predict because of starting length of 12 (first prediction is for time 13)
            for iteration in range(len(pred_series[chunk_id]) - 12):

                print(f'Iteration: {iteration}', file=sys.stderr)

                # Take last pre-trained model (or original pre-trained model in first iteration)
                if iteration > 0:
                    model_last_iteration_f = open(f'./data/darts/1000_chunks/{model_type}/{parameter}/04_pre'
                                                  f'-trained_model_{chunk_id}_{iteration-1}.pickle', 'rb')
                    model_for_iteration = pickle.load(model_last_iteration_f)
                    model_last_iteration_f.close()

                # Predict one measurement
                current_pred = model_for_iteration.predict(
                    n=1,
                    series=pred_series[chunk_id][:12+iteration])

                # Save model after each iteration
                extended_model_f = open(f'./data/darts/1000_chunks/{model_type}/{parameter}/04_pre-trained_model_{chunk_id}_{iteration}.pickle', 'wb')
                pickle.dump(model_for_iteration, extended_model_f, protocol=pickle.HIGHEST_PROTOCOL)
                extended_model_f.close()

                # Add intermediate prediction result to DataFrame
                final_pred = final_pred.append({'Time': current_pred.start_time(),
                                                'Value': current_pred.first_value()},
                                               ignore_index=True)

                # Save intermediate prediction of chunk as pickle file
                final_pred_f = open(f'./data/darts/1000_chunks/{model_type}/{parameter}/05_prediction_{chunk_id}_{iteration}.pickle', 'wb')
                pickle.dump(final_pred, final_pred_f, protocol=pickle.HIGHEST_PROTOCOL)
                final_pred_f.close()

            # Convert DataFrame to TimeSeries
            final_pred = TimeSeries.from_dataframe(
                df=final_pred,
                time_col='Time',
                value_cols=['Value'],
                freq='H')

            # Calculate MSE
            mse_for_chunk = mse(pred_series[chunk_id][12:], final_pred)
            print(f'MSE: {round(mse_for_chunk, 2)}', file=sys.stderr)

            # Plot predictions
            sns.set_style('whitegrid')
            plt.figure(figsize=(8, 5))
            pred_series[chunk_id].plot(label=f'{parameter.upper()} - actual')
            final_pred.plot(label=f'{parameter.upper()} - predicted')

            # Adjust texts of plot
            plt.legend()
            plt.suptitle(f'Prediction of {parameter.upper()} with 1000 Chunk IDs and {model_type} model', fontweight='bold')
            plt.title(f'MSE: {round(mse_for_chunk, 2)}%')
            plt.xlabel('Time')
            plt.ylabel('Scaled Value')

            plt.savefig(f'./plots/darts/1000_chunks/{model_type}/{parameter}/prediction_{chunk_id}.png', dpi=1200)

print('\nFinished.', file=sys.stderr)
sys.stderr.close()

# FUTURE TODO: try less/ more epochs
# FUTURE TODO: run with min/ max/ mean values
# FUTURE TODO: run with all chunks
