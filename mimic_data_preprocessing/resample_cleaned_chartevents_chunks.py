import numpy as np
import pandas as pd
import pyarrow as pa
import time

starttime = time.time()

path_to_data = '/hpi/fs00/share/MPSS2021BA1/data/resampling/'

print('Start reading the input file.')
chartevents = pd.read_parquet(str(path_to_data+'resample_input_hr.parquet'), engine='pyarrow')
# chartevents = pd.read_parquet(str(path_to_data+'resample_input_bp.parquet'), engine='pyarrow')
# chartevents = pd.read_parquet(str(path_to_data+'resample_input_o2.parquet'), engine='pyarrow')
# chartevents = pd.read_parquet(str(path_to_data+'resample_input_hr_first1000.parquet'), engine='pyarrow')
# chartevents = pd.read_parquet(str(path_to_data+'resample_input_bp_first1000.parquet'), engine='pyarrow')
# chartevents = pd.read_parquet(str(path_to_data+'resample_input_o2_first1000.parquet'), engine='pyarrow')
print('Reading of the input file completed.')

unique_chunkids = chartevents.CHUNK_ID_FILLED_TH.unique()

# Create data frame with the vital parameter ITEMIDs and associated alarm threshold ITEMIDs
parameters = pd.DataFrame({
    'VITAL_PARAMETER_LABEL':                    ['HR',      'NBPs',     'SpO2'],
    'VITAL_PARAMETER_ITEMID_VALUE':             [220045,    220179,     220277],
    'VITAL_PARAMETER_ITEMID_THRESHOLD_HIGH':    [220046,    223751,     223769],
    'VITAL_PARAMETER_ITEMID_THRESHOLD_LOW':     [220047,    223752,     223770]})

# Dictionaries are used to organize the data within the for loops (all_chunks_dict, current_chunk_dict, current_chunk_parameter_dict).
# After an iteration, the nested dictionary is usually transformed into a data frame using pd.concat(), so that eventually a flat table can be stored instead of a nested dictionary.

all_chunks_dict = dict()
chunkno = 1

for chunkid in unique_chunkids:

    current_chunk_dict = dict()

    for i, parameter in parameters.iterrows():

        current_chunk_parameter_dict = dict()

        # Get vital parameter value series for current chunk/ vital parameter combination and ...
        # ... resample the vital parameter value series using different methods when downsampling (median, mean, max, min)

        # Resampling of VALUENUM_CLEAN with a frequency of 60 min (1 hour), using the median of the values when downsampling.
        current_chunk_parameter_dict['VITAL_PARAMTER_VALUE_MEDIAN_RESAMPLING'] = chartevents[
            (chartevents['CHUNK_ID_FILLED_TH'] == chunkid) & (chartevents['ITEMID'] == parameter['VITAL_PARAMETER_ITEMID_VALUE'])][
            ['CHARTTIME','VALUENUM_CLEAN']
            ].sort_values(by=['CHARTTIME']).set_index('CHARTTIME').squeeze(axis=1).rename(parameter['VITAL_PARAMETER_ITEMID_VALUE']).resample('1H').median()
        
        # Resampling of VALUENUM_CLEAN with a frequency of 60 min (1 hour), using the mean of the values when downsampling.
        current_chunk_parameter_dict['VITAL_PARAMTER_VALUE_MEAN_RESAMPLING'] = chartevents[
            (chartevents['CHUNK_ID_FILLED_TH'] == chunkid) & (chartevents['ITEMID'] == parameter['VITAL_PARAMETER_ITEMID_VALUE'])][
            ['CHARTTIME','VALUENUM_CLEAN']
            ].sort_values(by=['CHARTTIME']).set_index('CHARTTIME').squeeze(axis=1).rename(parameter['VITAL_PARAMETER_ITEMID_VALUE']).resample('1H').mean()
        
        # Resampling of VALUENUM_CLEAN with a frequency of 60 min (1 hour), using the maximum value when downsampling.
        current_chunk_parameter_dict['VITAL_PARAMTER_VALUE_MAX_RESAMPLING'] = chartevents[
            (chartevents['CHUNK_ID_FILLED_TH'] == chunkid) & (chartevents['ITEMID'] == parameter['VITAL_PARAMETER_ITEMID_VALUE'])][
            ['CHARTTIME','VALUENUM_CLEAN']
            ].sort_values(by=['CHARTTIME']).set_index('CHARTTIME').squeeze(axis=1).rename(parameter['VITAL_PARAMETER_ITEMID_VALUE']).resample('1H').max()
        
        # Resampling of VALUENUM_CLEAN with a frequency of 60 min (1 hour), using the minimum value when downsampling.
        current_chunk_parameter_dict['VITAL_PARAMTER_VALUE_MIN_RESAMPLING'] = chartevents[
            (chartevents['CHUNK_ID_FILLED_TH'] == chunkid) & (chartevents['ITEMID'] == parameter['VITAL_PARAMETER_ITEMID_VALUE'])][
            ['CHARTTIME','VALUENUM_CLEAN']
            ].sort_values(by=['CHARTTIME']).set_index('CHARTTIME').squeeze(axis=1).rename(parameter['VITAL_PARAMETER_ITEMID_VALUE']).resample('1H').min()
        
        # Get alarm threshold value series for current chunk/ vital parameter combination
                
        current_chunk_parameter_dict['THRESHOLD_VALUE_HIGH'] = chartevents[
            (chartevents['CHUNK_ID_FILLED_TH'] == chunkid) & (chartevents['ITEMID'] == parameter['VITAL_PARAMETER_ITEMID_THRESHOLD_HIGH'])][
            ['CHARTTIME','VALUENUM_CLEAN']
            ].sort_values(by=['CHARTTIME']).set_index('CHARTTIME').squeeze(axis=1).rename(parameter['VITAL_PARAMETER_ITEMID_THRESHOLD_HIGH'])
        
        current_chunk_parameter_dict['THRESHOLD_VALUE_LOW'] = chartevents[
            (chartevents['CHUNK_ID_FILLED_TH'] == chunkid) & (chartevents['ITEMID'] == parameter['VITAL_PARAMETER_ITEMID_THRESHOLD_LOW'])][
            ['CHARTTIME','VALUENUM_CLEAN']
            ].sort_values(by=['CHARTTIME']).set_index('CHARTTIME').squeeze(axis=1).rename(parameter['VITAL_PARAMETER_ITEMID_THRESHOLD_LOW'])
        
        # Merge resampled vital parameter value series with associated alarm threshold value series into new data frame current_chunk_parameter_df
        current_chunk_parameter_df = pd.concat(current_chunk_parameter_dict, axis=1)

        # Interpolate missing values for alarm threshold value series using the last available value (also called forward fill).
        # If there is no previous value available, no value will be inserted during the interpolation. The value remains NaN.
        current_chunk_parameter_df['THRESHOLD_VALUE_HIGH'].interpolate('pad', inplace=True)
        current_chunk_parameter_df['THRESHOLD_VALUE_LOW'].interpolate('pad', inplace=True)

        # Filter for rows where the vital parameter value series are not NaN.
        # This removes the rows with an irregular timestamp originating from the merge with the alarm threshold series.
        # This step must not be performed earlier, because those rows are needed for the preceding interpolation of alarm threshold value series.
        current_chunk_parameter_df = current_chunk_parameter_df[current_chunk_parameter_df['VITAL_PARAMTER_VALUE_MEDIAN_RESAMPLING'].notna()]
        
        current_chunk_dict[parameter['VITAL_PARAMETER_LABEL']] = current_chunk_parameter_df.reset_index()
        
    else:
        None
    
    # Transform nested dictionary current_chunk_dict into a data frame using pd.concat()
    all_chunks_dict[chunkid] = pd.concat(current_chunk_dict, axis=0).reset_index(level=0).rename(columns={'level_0':'VITAL_PARAMETER_NAME'})

    # Save as parquet file
    # all_chunks_dict[chunkid].to_parquet(str(path_to_data+'resample_output_test_'+str(chunkid)+'.parquet'), engine='pyarrow')
    chartevents_resampled_running = pd.concat(all_chunks_dict, axis=0).reset_index(level=0).rename(columns={'level_0':'CHUNK_ID_FILLED_TH'})
    chartevents_resampled_running.to_parquet(str(path_to_data+'resample_output_running_hr.parquet'), engine='pyarrow')
    # chartevents_resampled_running.to_parquet(str(path_to_data+'resample_output_running_bp.parquet'), engine='pyarrow')
    # chartevents_resampled_running.to_parquet(str(path_to_data+'resample_output_running_o2.parquet'), engine='pyarrow')
    # chartevents_resampled_running.to_parquet(str(path_to_data+'resample_output_running_hr_first1000.parquet'), engine='pyarrow')
    # chartevents_resampled_running.to_parquet(str(path_to_data+'resample_output_running_bp_first1000.parquet'), engine='pyarrow')
    # chartevents_resampled_running.to_parquet(str(path_to_data+'resample_output_running_o2_first1000.parquet'), engine='pyarrow')
    runningtime = round(((time.time() - starttime) / 60), 5)
    print('Completed chunk number '+str(chunkno)+', running time in minutes: '+str(runningtime))
    chunkno = chunkno+1

# Transform nested dictionary all_chunks_dict into a data frame using pd.concat()
chartevents_resampled = pd.concat(all_chunks_dict, axis=0).reset_index(level=0).rename(columns={'level_0':'CHUNK_ID_FILLED_TH'})

# Save as parquet file
chartevents_resampled.to_parquet(str(path_to_data+'resample_output_hr.parquet'), engine='pyarrow')
# chartevents_resampled.to_parquet(str(path_to_data+'resample_output_bp.parquet'), engine='pyarrow')
# chartevents_resampled.to_parquet(str(path_to_data+'resample_output_o2.parquet'), engine='pyarrow')
# chartevents_resampled.to_parquet(str(path_to_data+'resample_output_hr_first1000.parquet'), engine='pyarrow')
# chartevents_resampled.to_parquet(str(path_to_data+'resample_output_bp_first1000.parquet'), engine='pyarrow')
# chartevents_resampled.to_parquet(str(path_to_data+'resample_output_o2_first1000.parquet'), engine='pyarrow')

endtime = round(((time.time() - starttime) / 60), 5)
print('DONE')
print('Completed '+str(chunkno)+' chunks in '+str(endtime)+' minutes')
