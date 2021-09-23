import numpy as np


# Model types list is subset of {'RNN', 'LSTM', 'GRU'}
model_types = ['RNN', 'LSTM', 'GRU']

# Parameter list is subset of {'hr', 'bp', 'o2'}
parameters = ['hr', 'bp', 'o2']

# Define (in hours) input length to imitate ARIMA(X) training size (start after this length with forecast)
input_length = 12

# Define (in hours) how many data points should be predicted at once
output_length = 1

# Define (in percent) how big testing dataset should be per window (rest is used for training)
prediction_data_size = 20

# Number of chunks can be 1000, 2000 or 15000
n_chunks = 2000

# Number of windows defines how many chunks are predicted at once
# Overall, prediction_data_size * n_windows percent of all chunks are predicted
n_windows = 5

# Define if exogenous input is feeded into ML models
# If so, MIN/MAX resampled chunks are used as endogenous and MEDIAN resampled chunks as exogenous input
# If not, only MEDIAN resampled chunks are used as endogenous input
with_exogenous_input = True

# Scaling method can be np.nan, 'standard' (s1) or 'min-max' (s2)
# If it is NaN, no scaling will be applied
scaling_method = 'standard'
