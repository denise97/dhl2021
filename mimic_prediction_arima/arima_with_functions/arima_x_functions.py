import config
from darts.models import AutoARIMA

# Adjust minimum train length for darts AutoARIMA() 
class ModifiedAutoARIMA(AutoARIMA):

    def __init__(self, *autoarima_args, **autoarima_kwarg):
        super().__init__(*autoarima_args, **autoarima_kwarg)

    @property
    def min_train_series_length(self) -> int:
        return config.glob_train_size


# Valid Inputs:
# arima_library: pmd, darts
# arima_mode: arima,arimax
# train_size: 12, 30
# test_size: (for later)
# train_size_modus : steady, expanding
# parameter: hr, bp, o2
# chunk_amount: first1000, first15000, first2000
# resampling_methods_endog: example: [MAX,MIN] - high,low
# resampling_methods_exog: example: [MEDIAN,MEDIAN] - high,low
def save_user_input(arima_library,arima_mode,train_size,test_size,train_size_modus,parameter,chunk_amount,resampling_methods_endog,resampling_methods_exog):
    
    config.glob_arima_library = arima_library
    config.glob_arima_mode = arima_mode    
    config.glob_train_size = train_size     
    config.glob_test_size = test_size    
    config.glob_train_size_modus = train_size_modus    
    config.glob_parameter = parameter     
    config.glob_chunk_amount = chunk_amount
    config.glob_resampling_methods_endog = resampling_methods_endog
    config.glob_resampling_methods_exog = resampling_methods_exog


def read_preprocessed_dict(path_to_data):
    
    import time
    import pickle 
    

    starttime = time.time()
    print('Start reading the input file.')

    input_file = open(str(path_to_data) + 'dict_of_chunk_iterations_with_'+str(config.glob_train_size_modus)+ '_train_' + str(config.glob_train_size)+ '_'+str(config.glob_parameter)+ '_' +str(config.glob_chunk_amount)+'.pickle','rb')

    dict_of_chunk_series_with_test_and_train = pickle.load(input_file)
    input_file.close()

    endtime = round(((time.time() - starttime) / 60), 5)
    print('Reading of the input file completed after '+str(endtime)+' minutes.')

    return dict_of_chunk_series_with_test_and_train   
    
def write_accuracy_dictionary(path_to_data,accuracy_dict_for_chunk_iterations):
    import pickle

    print('Starting saving accuracy dictionary.')
    output_file = open(str(path_to_data)+'accuracy_dict_for_chunk_iterations_'+str(config.glob_arima_library)+'_'+str(config.glob_arima_mode)+'_'+str(config.glob_train_size_modus)+'_'+str(config.glob_train_size)+ '_'+str(config.glob_parameter)+ '_' +str(config.glob_chunk_amount)+'.pickle', 'wb')
    pickle.dump(accuracy_dict_for_chunk_iterations, output_file)
    output_file.close()
    print('Completed saving accuracy dictionary.')


def write_chunk_iterations_with_value_errors(path_to_data,chunk_iterations_with_value_error):
    import pyarrow as pa
    
    chunk_iterations_with_value_error.to_parquet(str(path_to_data)+'chunk_iterations_with_value_error_'+str(config.glob_arima_library)+'_'+str(config.glob_arima_mode)+'_'+str(config.glob_train_size_modus)+'_'+str(config.glob_train_size)+ '_'+str(config.glob_parameter)+ '_' +str(config.glob_chunk_amount)+'.parquet', engine='pyarrow')


def write_dict_of_chunk_series_with_forecast_df(path_to_data,dict_of_chunk_series_with_forecast_df):
    import pickle
    print('Starting saving forecast dictionary.')
    output_file = open(str(path_to_data)+'dict_of_chunk_series_with_forecast_df_'+str(config.glob_arima_library)+'_'+str(config.glob_arima_mode)+'_'+str(config.glob_train_size_modus)+'_'+str(config.glob_train_size)+ '_'+str(config.glob_parameter)+ '_' +str(config.glob_chunk_amount)+'.pickle', 'wb')
    pickle.dump(dict_of_chunk_series_with_forecast_df,output_file)
    output_file.close()
    print('Completed saving forecast dictionary.')

def add_triggered_alarms_info(all_dict_lists_as_df):
    import numpy as np

    resampling_method_for_high = str.lower(config.glob_resampling_methods_endog[0])
    resampling_method_for_low = str.lower(config.glob_resampling_methods_endog[1])

    df_for_chunk_iteration = all_dict_lists_as_df
            
    # True alarms
    tmp_median_test = df_for_chunk_iteration['test_list_high_'+str(resampling_method_for_high)]
    df_for_chunk_iteration['high_alarm_triggered'] = np.where(df_for_chunk_iteration['test_list_high_'+str(resampling_method_for_high)] > df_for_chunk_iteration['threshold_high_for_test_list'] ,1,0)
    df_for_chunk_iteration['low_alarm_triggered'] = np.where(df_for_chunk_iteration['test_list_low_'+str(resampling_method_for_low)] < df_for_chunk_iteration['threshold_low_for_test_list'] ,1,0)
            
    # forecast
    df_for_chunk_iteration['high_alarm_triggered_forecast_'+str(config.glob_arima_mode)] = np.where(df_for_chunk_iteration['forecast_list_'+str(config.glob_arima_mode)+'_endog_high'] > df_for_chunk_iteration['threshold_high_for_test_list'],1,0)
    df_for_chunk_iteration['low_alarm_triggered_forecast_'+str(config.glob_arima_mode)] = np.where(df_for_chunk_iteration['forecast_list_'+str(config.glob_arima_mode)+'_endog_low'] < df_for_chunk_iteration['threshold_low_for_test_list'],1,0)
    return df_for_chunk_iteration

def calculate_confusion_matrix_high_alarms(df_for_chunk_iteration):
    import pandas as pd
    tp, tn, fp, fn = 0, 0, 0, 0
    # select true high alarms triggered
    column_index_of_high_alarm_triggered = df_for_chunk_iteration.columns.get_loc('high_alarm_triggered')

    # select predicted high alarms
    column_index_of_high_alarm_triggered_forecast = df_for_chunk_iteration.columns.get_loc('high_alarm_triggered_forecast_'+str(config.glob_arima_mode))
    
    # create df with both as column
    high_alarms = df_for_chunk_iteration.iloc[0:,[column_index_of_high_alarm_triggered,column_index_of_high_alarm_triggered_forecast]]
    
    for row_in_high_alarms in high_alarms.iterrows():

        if row_in_high_alarms[1][0] and row_in_high_alarms[1][1]:
            tp +=1
            
        if row_in_high_alarms[1][0] and not row_in_high_alarms[1][1]:
            fn +=1
            
        if not row_in_high_alarms[1][0] and row_in_high_alarms[1][1]:
            fp +=1
            
        if not row_in_high_alarms[1][0] and not row_in_high_alarms[1][1]:
            tn +=1
            
    
    a_new_row = {"TP":tp,"FN":fn,"FP":fp,"TN":tn}
    a_new_row_series = pd.Series(a_new_row,name='accuracy_high_alarms_'+str(config.glob_arima_mode))

    return a_new_row_series

def calculate_confusion_matrix_low_alarms(df_for_chunk_iteration):
    import pandas as pd
    tp, tn, fp, fn = 0, 0, 0, 0
    # select true low alarms triggered
    column_index_of_low_alarm_triggered = df_for_chunk_iteration.columns.get_loc('low_alarm_triggered')

    # select predicted low alarms
    column_index_of_low_alarm_triggered_forecast = df_for_chunk_iteration.columns.get_loc('low_alarm_triggered_forecast_'+str(config.glob_arima_mode))
    
    # create df with both as column
    low_alarms = df_for_chunk_iteration.iloc[0:,[column_index_of_low_alarm_triggered,column_index_of_low_alarm_triggered_forecast]]
    
    for row_in_low_alarms in low_alarms.iterrows():

        if row_in_low_alarms[1][0] and row_in_low_alarms[1][1]:
            tp +=1
            
        if row_in_low_alarms[1][0] and not row_in_low_alarms[1][1]:
            fn +=1
            
        if not row_in_low_alarms[1][0] and row_in_low_alarms[1][1]:
            fp +=1
            
        if not row_in_low_alarms[1][0] and not row_in_low_alarms[1][1]:
            tn +=1
            
    
    a_new_row = {"TP":tp,"FN":fn,"FP":fp,"TN":tn}
    a_new_row_series = pd.Series(a_new_row,name='accuracy_low_alarms_'+str(config.glob_arima_mode))

    return a_new_row_series

def convert_list_to_TimeSeries(selected_list):
    import pandas as pd
    from darts import TimeSeries

    selected_list_as_df = selected_list.to_frame()
    selected_list_as_df.reset_index(level=0,inplace=True)
    start = 'Jan 1, 1970 00:00'
    selected_list_as_df['timestamp'] = pd.to_datetime(selected_list_as_df.index, origin=start, unit='h')
    selected_list_as_TS = TimeSeries.from_dataframe(selected_list_as_df, 'timestamp', selected_list.name, freq='H')
    return selected_list_as_TS

def convert_TimeSeries_to_list_for_steady_train(selected_TS,TRAIN,TEST,i,variable_type,alarm_type):
    import pandas as pd
    from darts import TimeSeries

    selected_TS_as_df = selected_TS.pd_dataframe()
    selected_TS_as_df['index_column'] = [*range(i+TRAIN,i+TRAIN+TEST,1)]
    selected_TS_as_df.set_index('index_column', inplace=True)
    selected_TS_as_series = pd.Series(selected_TS_as_df['0'],name="forecast_list_"+str(config.glob_arima_mode)+"_"+str(variable_type)+"_"+str(alarm_type))
    return selected_TS_as_series





    










