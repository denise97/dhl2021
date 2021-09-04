import config
from arima_x_functions import *

def perform_alarm_forecast_with_arima_steady(dict_of_chunk_series_with_test_and_train):

    import pandas as pd
    import pmdarima as pm
    import numpy as np
    import copy
    import time
    import warnings
    import pyarrow as pa
    import pickle
    from darts.models import AutoARIMA
    from darts import TimeSeries 
    
    starttime = time.time()

    runningtime = round(((time.time() - starttime) / 60), 5)
    print('Starting setting up dictionaries. Running time '+str(runningtime)+' min.')

    dict_of_chunk_series_with_test_and_train_and_forecast = copy.deepcopy(dict_of_chunk_series_with_test_and_train)
    dict_of_chunk_series_with_forecast_df = {}
    accuracy_dict_for_chunk_iterations = {}
    chunk_iterations_with_value_error = pd.DataFrame(columns=["CHUNK_ID_FILLED_TH","ITERATION","ERROR_MSG"]) 

    runningtime = round(((time.time() - starttime) / 60), 5)
    print('Completed setting up dictionaries. Running time '+str(runningtime)+' min.')

    for j, chunk in enumerate(dict_of_chunk_series_with_test_and_train_and_forecast):
        dict_of_chunk_series_with_forecast_df[chunk] = {}
        accuracy_dict_for_chunk_iterations[chunk] = {}

        runningtime = round(((time.time() - starttime) / 60), 5)
        print('Chunk '+str(j)+' (ID: '+str(chunk)+'): START. Running time '+str(runningtime)+' min.')
        
        for i, chunk_iteration in enumerate(dict_of_chunk_series_with_test_and_train_and_forecast[chunk]):
            
            TRAIN = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TRAIN_LIST_MEDIAN"].size
            TEST = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TEST_LIST_MEDIAN"].size
            
            accurracy_matrix_df_for_chunk_iteration = pd.DataFrame(columns=["TP","FN","FP","TN"])            

            try:

                if config.glob_arima_library == 'pmd':
                #call method for pmd arima
                    dict_of_chunk_series_with_forecast_df[chunk][chunk_iteration] = perform_pmd_arima_steady(dict_of_chunk_series_with_test_and_train_and_forecast,TRAIN,TEST,i,chunk,chunk_iteration)
                    
                elif config.glob_arima_library == 'darts':
                #call method for darts arima
                    dict_of_chunk_series_with_forecast_df[chunk][chunk_iteration] = perform_darts_arima_steady(dict_of_chunk_series_with_test_and_train_and_forecast,TRAIN,TEST,i,chunk,chunk_iteration)
                
                else:
                    print("Invalid User Input for arima_library! Valid Inputs: 'pmd' or 'darts'.")
                    break              
                                        
               
                df_for_chunk_iteration = dict_of_chunk_series_with_forecast_df[chunk][chunk_iteration]

                # add triggered alarms

                dict_of_chunk_series_with_forecast_df[chunk][chunk_iteration] = add_triggered_alarms_info(df_for_chunk_iteration)

                #calculate confusion matrix

                df_for_chunk_iteration_with_alarms = dict_of_chunk_series_with_forecast_df[chunk][chunk_iteration]

                #high
                confusion_matrix_for_high_alarms = calculate_confusion_matrix_high_alarms(df_for_chunk_iteration_with_alarms)
                accurracy_matrix_df_for_chunk_iteration = accurracy_matrix_df_for_chunk_iteration.append(confusion_matrix_for_high_alarms)
                print('Chunk '+str(j)+' (ID: '+str(chunk)+') iteration '+str(chunk_iteration)+': Completed Confusion Matrix - High Alarms.')

                #low
                confusion_matrix_for_low_alarms = calculate_confusion_matrix_low_alarms(df_for_chunk_iteration_with_alarms)
                accurracy_matrix_df_for_chunk_iteration = accurracy_matrix_df_for_chunk_iteration.append(confusion_matrix_for_low_alarms)                
  
                print('Chunk '+str(j)+' (ID: '+str(chunk)+') iteration '+str(chunk_iteration)+': Completed Confusion Matrix - Low Alarms.')
                
                # Add confusion matrix to dictionary
                accuracy_dict_for_chunk_iterations[chunk][chunk_iteration] = accurracy_matrix_df_for_chunk_iteration
            
            except ValueError as ve:

                ve_string = str(ve)
                a_new_row = {"CHUNK_ID_FILLED_TH":chunk,"ITERATION":chunk_iteration,"ERROR_MSG":ve_string}
                a_new_row_series = pd.Series(a_new_row)
                chunk_iterations_with_value_error = chunk_iterations_with_value_error.append(a_new_row_series, ignore_index = True)                           
                print("VALUE ERROR DETECTED:")
                print(a_new_row_series)   
                                   
   
        runningtime = round(((time.time() - starttime) / 60), 5)
        print('Chunk '+str(j)+' (ID: '+str(chunk)+'): Completed chunk. Running time '+str(runningtime)+' min.')
        print('--------------------')


    endtime = round(((time.time() - starttime) / 60), 5)
    print('DONE WITH ' + str(config.glob_arima_library)+ ' ARIMA STEADY')
    print('Completed in '+str(endtime)+' minutes.')  

    return accuracy_dict_for_chunk_iterations, dict_of_chunk_series_with_forecast_df, chunk_iterations_with_value_error



def perform_pmd_arima_steady(dict_of_chunk_series_with_test_and_train_and_forecast,TRAIN,TEST,i,chunk,chunk_iteration):
    import pmdarima as pm
    import pandas as pd
    import numpy as np

    resampling_method_for_high = config.glob_resampling_methods_endog[0]
    resampling_method_for_low = config.glob_resampling_methods_endog[1]

    np.seterr(all='ignore')

    ########################
    # ARIMA for High Alarms
    ########################

    current_train_list_high = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TRAIN_LIST_"+str(resampling_method_for_high)]
    current_test_list_high = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TEST_LIST_"+str(resampling_method_for_high)] 


    arima_high = pm.auto_arima(current_train_list_high, seasonal=False, suppress_warnings=True, error_action='ignore')
    forecast_arima_high = pd.Series(arima_high.predict(TEST), index=[*range(i+TRAIN,i+TRAIN+TEST,1)], name="forecast_list_arima_endog_high")

    print('Chunk (ID: '+str(chunk)+') iteration '+str(chunk_iteration)+': Completed ARIMA - High Alarms.')

    ########################
    # ARIMA for Low Alarms
    ########################
    
    current_train_list_low = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TRAIN_LIST_"+str(resampling_method_for_low)]
    current_test_list_low = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TEST_LIST_"+str(resampling_method_for_low)] 


    arima_low = pm.auto_arima(current_train_list_low, seasonal=False, suppress_warnings=True, error_action='ignore')
    forecast_arima_low = pd.Series(arima_low.predict(TEST), index=[*range(i+TRAIN,i+TRAIN+TEST,1)], name="forecast_list_arima_endog_low")

    print('Chunk (ID: '+str(chunk)+') iteration '+str(chunk_iteration)+': Completed ARIMA - Low Alarms.')
    
    
    # extract threshold series 
    threshold_high_for_test_list = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["THRESHOLD_HIGH_FOR_TEST_LIST"]
    threshold_low_for_test_list = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["THRESHOLD_LOW_FOR_TEST_LIST"]
                
    # write to dict_of_chunk_series_with_forecast_df dataframe
    # give test lists new name in case resampling input for high and low is the same
    resampling_method_for_high_lower = str.lower(resampling_method_for_high)
    resampling_method_for_low_lower = str.lower(resampling_method_for_low)
    current_test_list_high_renamed = pd.Series(current_test_list_high, index=[*range(i+TRAIN,i+TRAIN+TEST,1)], name="test_list_high_"+str(resampling_method_for_high_lower))
    current_test_list_low_renamed = pd.Series(current_test_list_low, index=[*range(i+TRAIN,i+TRAIN+TEST,1)], name="test_list_low_"+str(resampling_method_for_low_lower))
    all_dict_lists_as_df = pd.concat([current_test_list_high_renamed,forecast_arima_high,threshold_high_for_test_list,current_test_list_low_renamed,forecast_arima_low,threshold_low_for_test_list],axis=1)    
    return all_dict_lists_as_df


def perform_darts_arima_steady(dict_of_chunk_series_with_test_and_train_and_forecast,TRAIN,TEST,i,chunk,chunk_iteration):
    #from darts.models import AutoARIMA
    #from darts import TimeSeries
    import pandas as pd
    import numpy as np

    resampling_method_for_high = config.glob_resampling_methods_endog[0]
    resampling_method_for_low = config.glob_resampling_methods_endog[1]

    np.seterr(all='ignore')

    ########################
    # ARIMA for High Alarms
    ########################

    current_train_list_high = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TRAIN_LIST_"+str(resampling_method_for_high)]
    current_test_list_high = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TEST_LIST_"+str(resampling_method_for_high)] 

    #adapt for darts - convert to TimeSeries 
    current_train_list_timeseries_high = convert_list_to_TimeSeries(current_train_list_high)


    arima_high = ModifiedAutoARIMA(seasonal=False, suppress_warnings=True, error_action='ignore')
    arima_high.fit(current_train_list_timeseries_high)
    forecast_arima_high = arima_high.predict(TEST)

    #reconvert TimeSeries Forecast to Series with the first index being 'last index of train' +1
    forecast_arima_high_series = convert_TimeSeries_to_list_for_steady_train(forecast_arima_high,TRAIN,TEST,i,'endog','high')

    ########################
    # ARIMA for Low Alarms
    ########################

    current_train_list_low = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TRAIN_LIST_"+str(resampling_method_for_low)]
    current_test_list_low = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TEST_LIST_"+str(resampling_method_for_low)] 

    #adapt for darts - convert to TimeSeries 
    current_train_list_timeseries_low = convert_list_to_TimeSeries(current_train_list_low)


    arima_low = ModifiedAutoARIMA(seasonal=False, suppress_warnings=True, error_action='ignore')
    arima_low.fit(current_train_list_timeseries_low)
    forecast_arima_low = arima_low.predict(TEST)

    #reconvert TimeSeries Forecast to Series with the first index being 'last index of train' +1
    forecast_arima_low_series = convert_TimeSeries_to_list_for_steady_train(forecast_arima_low,TRAIN,TEST,i,'endog','low')


    # extract threshold series 
    threshold_high_for_test_list = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["THRESHOLD_HIGH_FOR_TEST_LIST"]
    threshold_low_for_test_list = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["THRESHOLD_LOW_FOR_TEST_LIST"]

    # write to dict_of_chunk_series_with_forecast_df dataframe
    # give test lists new name in case resampling input for high and low is the same
    resampling_method_for_high_lower = str.lower(resampling_method_for_high)
    resampling_method_for_low_lower = str.lower(resampling_method_for_low)
    current_test_list_high_renamed = pd.Series(current_test_list_high, index=[*range(i+TRAIN,i+TRAIN+TEST,1)], name="test_list_high_"+str(resampling_method_for_high_lower))
    current_test_list_low_renamed = pd.Series(current_test_list_low, index=[*range(i+TRAIN,i+TRAIN+TEST,1)], name="test_list_low_"+str(resampling_method_for_low_lower))
    all_dict_lists_as_df = pd.concat([current_test_list_high_renamed,forecast_arima_high_series,threshold_high_for_test_list,current_test_list_low_renamed,forecast_arima_low_series,threshold_low_for_test_list],axis=1)
    return all_dict_lists_as_df







    










