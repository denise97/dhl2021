# path_to_data = '/hpi/fs00/share/MPSS2021BA1/data/'
path_to_data = '../data/'
import pandas as pd
import pyarrow as pa
import time

starttime = time.time()
print('Start reading the input file.')

# Read chartevents_resampled from parquet file to pandas data frame
#chartevents_resampled = pd.read_parquet(str(path_to_data)+'resampling/resample_running_output_hr.parquet', engine='pyarrow')
#chartevents_resampled = pd.read_parquet(str(path_to_data)+'resampling/resample_output_bp.parquet', engine='pyarrow')
#chartevents_resampled = pd.read_parquet(str(path_to_data)+'resampling/resample_output_o2.parquet', engine='pyarrow')
#chartevents_resampled = pd.read_parquet(str(path_to_data)+'resampling/resample_output_hr_first1000.parquet', engine='pyarrow')
#chartevents_resampled = pd.read_parquet(str(path_to_data)+'resampling/resample_output_bp_first1000.parquet', engine='pyarrow')
#chartevents_resampled = pd.read_parquet(str(path_to_data)+'resampling/resample_output_o2_first1000.parquet', engine='pyarrow')

endtime = round(((time.time() - starttime) / 60), 5)
print('Reading of the input file completed after '+str(endtime)+' minutes.')

# Sampling rate of 1 data point per hour - Test for different values in the future - e.g. longer training set
TRAIN = 12 # 12 * 1 h = 12 hour training period
TEST = 1 # 1 * 1 h = 2 hours testing period
STEP = 1 # move 1 * 1 h = 1 hour per step
# Filter for chunks that have sufficient values to be used for training and testing the model
all_chunks_value_count = chartevents_resampled.CHUNK_ID_FILLED_TH.value_counts()
chunkid_filter = all_chunks_value_count[all_chunks_value_count >= (TRAIN + TEST)].index
arima_data = chartevents_resampled[chartevents_resampled.CHUNK_ID_FILLED_TH.isin(chunkid_filter)].copy()

# Create new HOURS_SINCE_FIRST_RECORD column containing the time difference that has passed since the first timestamp of the measurement series.
import numpy as np
# arima_data['MINUTES_SINCE_FIRST_RECORD'] = arima_data.groupby('CHUNK_ID_FILLED_TH')#['CHARTTIME'].transform(lambda x: (x - x.min())/np.timedelta64(1,'m'))
# Alternative for hours instead of minutes
arima_data['HOURS_SINCE_FIRST_RECORD'] = arima_data.groupby('CHUNK_ID_FILLED_TH')['CHARTTIME'].transform(lambda x: (x - x.min())/np.timedelta64(1,'h'))

# Create dictionary with chunk id as key and a dataframe as value.
# This dataframe contains of three columns the vital parameter values, the high thresholds and the low thresholds.
# As the index of these three list is the same and can be referenced back to the "HOURS_SINCE_FIRST_RECORD", we keep the time related information.
# Example:
# dict_of_chunk_series = {
#     "<chunkid_A>" : { | vital_parameter_series | threshold_high_series | threshold_low_series
#                     0 |                   95.0 |                   120 |                   60
#                     1 |                   90.5 |                   120 |                   60
#                     2 |                   91.0 |                   120 |                   60
#                    },
#     "<chunkid_B>" : { | vital_parameter_series | threshold_high_series | threshold_low_series
#                     0 |                   88.0 |                   110 |                   50
#                     1 |                   78.5 |                   110 |                   50
#                     2 |                   73.0 |                   120 |                   60
#                    }
#  }
#
# Example with additional vital parameter series which can be used as an exogenous variable for ARIMAX
# dict_of_chunk_series_with_test_and_train = {
#     "<chunkid_A>" : | vital_parameter_series_median | vital_parameter_series_mean | vital_parameter_series_min |threshold_high_series | threshold_low_series (+max)
#                   0 |                          95.0 |                        98.0 |                          80|                  120 |                   60
#                   1 |                          90.5 |                        96.0 |                          79|                  120 |                   60
#                   2 |                          91.0 |                        94.0 |                          83|                  120 |                   60
#  }

runningtime = round(((time.time() - starttime) / 60), 5)
print('Starting setting up chunk based dictionary (First). Running time '+str(runningtime)+' min.')

dict_of_chunk_series = {}

for chunkid in chunkid_filter:
    
    chunk_data = arima_data[arima_data.CHUNK_ID_FILLED_TH == chunkid].copy()

    # vital parameter series - median resampling
    chunk_value_series_median = pd.Series(chunk_data['VITAL_PARAMTER_VALUE_MEDIAN_RESAMPLING'],name="vital_parameter_series_median")
    chunk_value_series_median = chunk_value_series_median.reset_index(drop=True)
    chunk_value_series_median.index = list(chunk_value_series_median.index)

    # vital parameter series - mean resampling
    chunk_value_series_mean = pd.Series(chunk_data['VITAL_PARAMTER_VALUE_MEAN_RESAMPLING'],name="vital_parameter_series_mean")
    chunk_value_series_mean = chunk_value_series_mean.reset_index(drop=True)
    chunk_value_series_mean.index = list(chunk_value_series_mean.index)

    # vital parameter series - min resampling
    chunk_value_series_min = pd.Series(chunk_data['VITAL_PARAMTER_VALUE_MIN_RESAMPLING'],name="vital_parameter_series_min")
    chunk_value_series_min = chunk_value_series_min.reset_index(drop=True)
    chunk_value_series_min.index = list(chunk_value_series_min.index)

    # vital parameter series - max resampling
    chunk_value_series_max = pd.Series(chunk_data['VITAL_PARAMTER_VALUE_MAX_RESAMPLING'],name="vital_parameter_series_max")
    chunk_value_series_max = chunk_value_series_max.reset_index(drop=True)
    chunk_value_series_max.index = list(chunk_value_series_max.index)  
    

    # threshold series high
    chunk_threshold_high_series = pd.Series(chunk_data['THRESHOLD_VALUE_HIGH'],name="threshold_high_series")
    chunk_threshold_high_series = chunk_threshold_high_series.reset_index(drop=True)
    chunk_threshold_high_series.index = list(chunk_threshold_high_series.index)

    # threshold series low
    chunk_threshold_low_series = pd.Series(chunk_data['THRESHOLD_VALUE_LOW'],name="threshold_low_series")
    chunk_threshold_low_series = chunk_threshold_low_series.reset_index(drop=True)
    chunk_threshold_low_series.index = list(chunk_threshold_low_series.index)

    # Append series with key (CHUNK_ID) into dictionary
    vital_parameter_and_thresholds_for_chunkid = pd.concat([chunk_value_series_median,chunk_value_series_mean,chunk_value_series_min,chunk_value_series_max,chunk_threshold_high_series,chunk_threshold_low_series],axis=1)
    dict_of_chunk_series[chunkid] = vital_parameter_and_thresholds_for_chunkid

runningtime = round(((time.time() - starttime) / 60), 5)
print('Finished setting up chunk based dictionary (First). Running time '+str(runningtime)+' min.')

# Create multiple test & train sets for each chunk to iteratively predict the next x measurements
# Create nested dictionary that holds the CHUNK_ID as first key.
# This key holds one dictionary for each iteration over this chunk. This depends on the TEST, TRAIN, and STEP.
# For each iteration we create another dictionary, whereby the last index of the train list acts as key.
# This key holds again one dictionary for the train list and one for the test list.
# Example:
# dict_of_chunk_series_with_test_and_train = {
#     "<chunkid_A>" : {
#         "<last_index_of_training_list_of_first_chunkid_A_iteration>" : {
#             "TRAIN_LIST_MEDIAN" : train_list_median,
#             "TEST_LIST_MEDIAN" : test_list_median,
#             "TRAIN_LIST_MEAN" : train_list_mean,
#             "TEST_LIST_MEAN" : test_list_mean,
#             "TRAIN_LIST_MIN" : train_list_min,
#             "TEST_LIST_MIN" : test_list_min,
#             "TRAIN_LIST_MAX" : train_list_max,
#             "TEST_LIST_MAX" : test_list_max,
#             "THRESHOLD_HIGH_FOR_TRAIN_LIST" : threshold_high_for_train_list ,
#             "THRESHOLD_LOW_FOR_TEST_LIST" : threshold_low_for_test_list
#         },
#         "<last_index_of_training_list_of_second_chunkid_A_iteration>" : {
#             "TRAIN_LIST_MEDIAN" : train_list_median,
#             "TEST_LIST_MEDIAN" : test_list_median,
#             "TRAIN_LIST_MEAN" : train_list_mean,
#             "TEST_LIST_MEAN" : test_list_mean,
#             "TRAIN_LIST_MIN" : train_list_min,
#             "TEST_LIST_MIN" : test_list_min,
#             "TRAIN_LIST_MAX" : train_list_max,
#             "TEST_LIST_MAX" : test_list_max,
#             "THRESHOLD_HIGH_FOR_TRAIN_LIST" : threshold_high_for_train_list ,
#             "THRESHOLD_LOW_FOR_TEST_LIST" : threshold_low_for_test_list
#         },
#     }
# }
import copy

runningtime = round(((time.time() - starttime) / 60), 5)
print('Starting setting up chunk iteration based dictionary for steady train size (Second). Running time '+str(runningtime)+' min.')

dict_of_chunk_series_with_test_and_train = {}

for i, chunk in enumerate(dict_of_chunk_series):
    # acces dataframe of current chunk
    chunk_series_for_chunk = copy.deepcopy(dict_of_chunk_series[chunk])

    # access vital_parameter_series_median of current chunk
    chunk_value_series_for_chunk_median = chunk_series_for_chunk["vital_parameter_series_median"]
    # access vital_parameter_series_mean of current chunk
    chunk_value_series_for_chunk_mean = chunk_series_for_chunk["vital_parameter_series_mean"]
    # access vital_parameter_series_min of current chunk
    chunk_value_series_for_chunk_min = chunk_series_for_chunk["vital_parameter_series_min"]
    # access vital_parameter_series_max of current chunk
    chunk_value_series_for_chunk_max = chunk_series_for_chunk["vital_parameter_series_max"]
        
    # access threshold_high_series of current chunk
    chunk_threshold_high_series_for_chunk = chunk_series_for_chunk["threshold_high_series"]
    # access threshold_low_series of current chunk
    chunk_threshold_low_series_for_chunk = chunk_series_for_chunk["threshold_low_series"]

    # create an empty dictionary for the key of the current chunk
    dict_of_chunk_series_with_test_and_train[chunk] = {}

    # create multiple test and train lists for that chunk
    for start in range(0, len(chunk_value_series_for_chunk_median) - (TRAIN + TEST)+1, STEP):
        
        # vital_parameter_series_median
        train_list_median = pd.Series(chunk_value_series_for_chunk_median[start : start+TRAIN], name="train_list_median")
        test_list_median = pd.Series(chunk_value_series_for_chunk_median[start+TRAIN : start+TRAIN+TEST], name="test_list_median")
        # vital_parameter_series_mean
        train_list_mean = pd.Series(chunk_value_series_for_chunk_mean[start : start+TRAIN], name="train_list_mean")
        test_list_mean = pd.Series(chunk_value_series_for_chunk_mean[start+TRAIN : start+TRAIN+TEST], name="test_list_mean")
        # vital_parameter_series_min
        train_list_min = pd.Series(chunk_value_series_for_chunk_min[start : start+TRAIN], name="train_list_min")
        test_list_min = pd.Series(chunk_value_series_for_chunk_min[start+TRAIN : start+TRAIN+TEST], name="test_list_min")
        # vital_parameter_series_max
        train_list_max = pd.Series(chunk_value_series_for_chunk_max[start : start+TRAIN], name="train_list_max")
        test_list_max = pd.Series(chunk_value_series_for_chunk_max[start+TRAIN : start+TRAIN+TEST], name="test_list_max")
        
        #threshold series high & low
        threshold_high_for_test_list = pd.Series(chunk_threshold_high_series_for_chunk[start+TRAIN : start+TRAIN+TEST],name="threshold_high_for_test_list")
        threshold_low_for_test_list = pd.Series(chunk_threshold_low_series_for_chunk[start+TRAIN : start+TRAIN+TEST],name="threshold_low_for_test_list")
        
        # For each iteration over the current chunk, we will create a dictionary that holds again the test and train list as dictionary
        # We use the last index of the current train list (which currently refers to the difference to first measurement) as second key
        second_key = train_list_median.index.max()
        dict_of_chunk_series_with_test_and_train[chunk][second_key] = {}
        # Assign the train and test list to the current chunk iteration      
        dict_of_chunk_series_with_test_and_train[chunk][second_key]["TRAIN_LIST_MEDIAN"] = train_list_median
        dict_of_chunk_series_with_test_and_train[chunk][second_key]["TRAIN_LIST_MEAN"] = train_list_mean
        dict_of_chunk_series_with_test_and_train[chunk][second_key]["TRAIN_LIST_MIN"] = train_list_min
        dict_of_chunk_series_with_test_and_train[chunk][second_key]["TRAIN_LIST_MAX"] = train_list_max

        dict_of_chunk_series_with_test_and_train[chunk][second_key]["TEST_LIST_MEDIAN"] = test_list_median
        dict_of_chunk_series_with_test_and_train[chunk][second_key]["TEST_LIST_MEAN"] = test_list_median
        dict_of_chunk_series_with_test_and_train[chunk][second_key]["TEST_LIST_MIN"] = test_list_min
        dict_of_chunk_series_with_test_and_train[chunk][second_key]["TEST_LIST_MAX"] = test_list_max

        dict_of_chunk_series_with_test_and_train[chunk][second_key]["THRESHOLD_HIGH_FOR_TEST_LIST"] = threshold_high_for_test_list
        dict_of_chunk_series_with_test_and_train[chunk][second_key]["THRESHOLD_LOW_FOR_TEST_LIST"] = threshold_low_for_test_list

runningtime = round(((time.time() - starttime) / 60), 5)
print('Finished setting up chunk iteration based dictionary for steady train size (Second). Running time '+str(runningtime)+' min.')

import pickle

runningtime = round(((time.time() - starttime) / 60), 5)
print('Started writing chunk iteration based dictionary for steady train size (Second) to pickle. Running time '+str(runningtime)+' min.')

# Write dictionaries with expanding train and fix test size to pickle file

#output_file = open(str(path_to_data)+'arima_preprocessing/dict_of_chunk_iterations_with_steady_train_'+str(TRAIN)+'_hr.pickle', 'wb')
#output_file = open(str(path_to_data)+'arima_preprocessing/dict_of_chunk_iterations_with_steady_train_'+str(TRAIN)+'_bp.pickle', 'wb')
#output_file = open(str(path_to_data)+'arima_preprocessing/dict_of_chunk_iterations_with_steady_train_'+str(TRAIN)+'_o2.pickle', 'wb')
#output_file = open(str(path_to_data)+'arima_preprocessing/dict_of_chunk_iterations_with_steady_train_'+str(TRAIN)+'_hr_first1000.pickle', 'wb')
#output_file = open(str(path_to_data)+'arima_preprocessing/dict_of_chunk_iterations_with_steady_train_'+str(TRAIN)+'_bp_first1000.pickle', 'wb')
#output_file = open(str(path_to_data)+'arima_preprocessing/dict_of_chunk_iterations_with_steady_train_'+str(TRAIN)+'_o2_first1000.pickle', 'wb')

pickle.dump(dict_of_chunk_series_with_test_and_train,output_file)
output_file.close()

runningtime = round(((time.time() - starttime) / 60), 5)
print('Finished writing chunk iteration based dictionary for steady train size (Second) to pickle. Running time '+str(runningtime)+' min.')
endtime = round(((time.time() - starttime) / 60), 5)
print('DONE')
print('Completed in '+str(endtime)+' minutes.')
