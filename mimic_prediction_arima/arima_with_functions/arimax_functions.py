import config
from arima_x_functions import *

def perform_alarm_forecast_with_arimax_steady(dict_of_chunk_series_with_test_and_train):

    import pandas as pd
    import numpy as np
    import copy
    import time

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
                    dict_of_chunk_series_with_forecast_df[chunk][chunk_iteration] = perform_pmd_arimax_steady(dict_of_chunk_series_with_test_and_train_and_forecast,TRAIN,TEST,i,chunk,chunk_iteration)
                    
                elif config.glob_arima_library == 'darts':
                #call method for darts arima
                    dict_of_chunk_series_with_forecast_df[chunk][chunk_iteration] = perform_darts_arimax_steady(dict_of_chunk_series_with_test_and_train_and_forecast,TRAIN,TEST,i,chunk,chunk_iteration)
                
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
                
                runningtime = round(((time.time() - starttime) / 60), 5)
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
    print('DONE WITH ' + str(config.glob_arima_library)+ ' ARIMAX STEADY')
    print('Completed in '+str(endtime)+' minutes.')  

    return accuracy_dict_for_chunk_iterations, dict_of_chunk_series_with_forecast_df, chunk_iterations_with_value_error




def perform_pmd_arimax_steady(dict_of_chunk_series_with_test_and_train_and_forecast,TRAIN,TEST,i,chunk,chunk_iteration):

    import pmdarima as pm
    import pandas as pd
    import numpy as np

    np.seterr(all='ignore')

    
    resampling_method_for_endog_high = config.glob_resampling_methods_endog[0]
    resampling_method_for_endog_low = config.glob_resampling_methods_endog[1]
    resampling_method_for_exog_high = config.glob_resampling_methods_exog[0]
    resampling_method_for_exog_low = config.glob_resampling_methods_exog[1]

    ################################
    # ARIMA for Exog. Variable High
    ################################

    # we can not simply add the 'current_test_list_exog_high' as exog in arimax.predict as we do not have that info at the current point in time
    # we need to first predict (with e.g. common arima) the value for current_test_list_exog_high and add this as exog

    # We only need one model for predicting the test value of the exog. variable as with the current configuration, we use the same exog. variable for high and low alarms

    current_train_list_exog_high = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TRAIN_LIST_"+str(resampling_method_for_exog_high)].values.reshape(-1, 1)
    current_test_list_exog_high = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TEST_LIST_"+str(resampling_method_for_exog_high)]

    #predict exog value for TEST
    arima_for_exog_high = pm.auto_arima(current_train_list_exog_high, seasonal=False, suppress_warnings=True, error_action='ignore')
    forecast_arima_exog_high = pd.Series(arima_for_exog_high.predict(TEST), index=[*range(i+TRAIN,i+TRAIN+TEST,1)], name="forecast_list_arima_exog_high")
    forecast_arima_exog_high_np = forecast_arima_exog_high.values.reshape(-1, 1)

    print('Chunk (ID: '+str(chunk)+') iteration '+str(chunk_iteration)+': Completed ARIMA for Exog - High Alarms.')

    
    ########################
    # ARIMAX for High Alarms
    ########################


    current_train_list_endog_high = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TRAIN_LIST_"+str(resampling_method_for_endog_high)] 
    current_test_list_endog_high = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TEST_LIST_"+str(resampling_method_for_endog_high)]
    

    #predict endog value for TEST with help of exog prediction    
    arimax_for_endog_high = pm.auto_arima(current_train_list_endog_high, X=current_train_list_exog_high, seasonal=False, suppress_warnings=True, error_action='ignore')
    forecast_arimax_endog_high = pd.Series(arimax_for_endog_high.predict(TEST, X=forecast_arima_exog_high_np), index=[*range(i+TRAIN,i+TRAIN+TEST,1)], name="forecast_list_arimax_endog_high")
    
    print('Chunk (ID: '+str(chunk)+') iteration '+str(chunk_iteration)+': Completed ARIMAX for Endog - High Alarms.')

    ################################
    # ARIMA for Exog. Variable Low
    ################################

    current_train_list_exog_low = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TRAIN_LIST_"+str(resampling_method_for_exog_low)].values.reshape(-1, 1)
    current_test_list_exog_low = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TEST_LIST_"+str(resampling_method_for_exog_low)]

    # we can not simply add the 'current_test_list_exog_low' as exog in arimax.predict as we do not have that info at the current point in time
    # we need to first predict (with e.g. common arima) the value for current_test_list_exog_low and add this as exog

    #predict exog value for TEST
    arima_for_exog_low = pm.auto_arima(current_train_list_exog_low, seasonal=False, suppress_warnings=True, error_action='ignore')
    forecast_arima_exog_low = pd.Series(arima_for_exog_low.predict(TEST), index=[*range(i+TRAIN,i+TRAIN+TEST,1)], name="forecast_list_arima_exog_low")
    forecast_arima_exog_low_np = forecast_arima_exog_low.values.reshape(-1, 1)
    print('Chunk (ID: '+str(chunk)+') iteration '+str(chunk_iteration)+': Completed ARIMA for Exog - Low Alarms.')

    ########################
    # ARIMAX for Low Alarms
    ########################
    
    current_train_list_endog_low = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TRAIN_LIST_"+str(resampling_method_for_endog_low)] 
    current_test_list_endog_low = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TEST_LIST_"+str(resampling_method_for_endog_low)] 

    
    #predict endog value for TEST with help of exog prediction    
    arimax_for_endog_low = pm.auto_arima(current_train_list_endog_low, X=current_train_list_exog_low, seasonal=False, suppress_warnings=True, error_action='ignore')
    forecast_arimax_endog_low = pd.Series(arimax_for_endog_low.predict(TEST, X=forecast_arima_exog_low_np), index=[*range(i+TRAIN,i+TRAIN+TEST,1)], name="forecast_list_arimax_endog_low")
    print('Chunk (ID: '+str(chunk)+') iteration '+str(chunk_iteration)+': Completed ARIMAX for Endog - Low Alarms.')

    # extract threshold series 
    threshold_high_for_test_list = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["THRESHOLD_HIGH_FOR_TEST_LIST"]
    threshold_low_for_test_list = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["THRESHOLD_LOW_FOR_TEST_LIST"]
    
    # write to dict_of_chunk_series_with_forecast_df dataframe
    # give test lists new name in case resampling input for high and low is the same
    resampling_method_for_high_lower = str.lower(resampling_method_for_endog_high)
    resampling_method_for_low_lower = str.lower(resampling_method_for_endog_low)
    current_test_list_endog_high_renamed = pd.Series(current_test_list_endog_high, index=[*range(i+TRAIN,i+TRAIN+TEST,1)], name="test_list_high_"+str(resampling_method_for_high_lower))
    current_test_list_endog_low_renamed = pd.Series(current_test_list_endog_low, index=[*range(i+TRAIN,i+TRAIN+TEST,1)], name="test_list_low_"+str(resampling_method_for_low_lower))
    all_dict_lists_as_df = pd.concat([current_test_list_endog_high_renamed,forecast_arimax_endog_high,current_test_list_exog_high,forecast_arima_exog_high,threshold_high_for_test_list,current_test_list_endog_low_renamed,forecast_arimax_endog_low,current_test_list_exog_low,forecast_arima_exog_low,threshold_low_for_test_list],axis=1)

    return all_dict_lists_as_df


def perform_darts_arimax_steady(dict_of_chunk_series_with_test_and_train_and_forecast,TRAIN,TEST,i,chunk,chunk_iteration):
    from darts.models import AutoARIMA
    from darts import TimeSeries
    import pandas as pd
    import numpy as np

    np.seterr(all='ignore')
    
    resampling_method_for_endog_high = config.glob_resampling_methods_endog[0]
    resampling_method_for_endog_low = config.glob_resampling_methods_endog[1]
    resampling_method_for_exog_high = config.glob_resampling_methods_exog[0]
    resampling_method_for_exog_low = config.glob_resampling_methods_exog[1]

    #####################################
    # DARTS ARIMA for Exog. Variable High
    #####################################

    current_train_list_exog_high = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TRAIN_LIST_"+str(resampling_method_for_exog_high)]
    current_test_list_exog_high = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TEST_LIST_"+str(resampling_method_for_exog_high)]

    #adapt for darts - convert to TimeSeries - Train List Exog - High
    current_train_list_timeseries_exog_high = convert_list_to_TimeSeries(current_train_list_exog_high)

    #adapt for darts - convert to TimeSeries - Test List Exog - High
    current_test_list_timeseries_exog_high = convert_list_to_TimeSeries(current_test_list_exog_high) 

    # we can not simply add the 'current_test_list_exog_low' as exog in arimax.predict as we do not have that info at the current point in time
    # we need to first predict (with e.g. common arima) the value for current_test_list_exog_low and add this as exog

    #predict exog value for TEST
    arima_for_exog_high = ModifiedAutoARIMA(seasonal=False, suppress_warnings=True, error_action='ignore')
    arima_for_exog_high.fit(current_train_list_timeseries_exog_high)
    forecast_arima_exog_high = arima_for_exog_high.predict(TEST)

    #reconvert TimeSeries Forecast to Series with the first index being 'last index of train' +1
    forecast_arima_exog_high_series = convert_TimeSeries_to_list_for_steady_train(forecast_arima_exog_high,TRAIN,TEST,i,'exog','high')
    print('Chunk (ID: '+str(chunk)+') iteration '+str(chunk_iteration)+': Completed ARIMA for Exog - High Alarms.')

    ###############################
    # DARTS ARIMAX for High Alarms
    ###############################

    current_train_list_endog_high = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TRAIN_LIST_"+str(resampling_method_for_endog_high)] 
    current_test_list_endog_high = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TEST_LIST_"+str(resampling_method_for_endog_high)]
    
    #adapt for darts - convert to TimeSeries - Train List - High
    current_train_list_timeseries_endog_high = convert_list_to_TimeSeries(current_train_list_endog_high)

    #perform darts arimax high
    arimax_for_endog_high = ModifiedAutoARIMA(seasonal=False, suppress_warnings=True, error_action='ignore')
    arimax_for_endog_high.fit(current_train_list_timeseries_endog_high, exog=current_train_list_timeseries_exog_high)
    forecast_arimax_endog_high = arimax_for_endog_high.predict(TEST, exog=forecast_arima_exog_high)
    
    #reconvert TimeSeries Forecast to Series with the first index being 'last index of train' +1
    forecast_arimax_endog_high_series = convert_TimeSeries_to_list_for_steady_train(forecast_arimax_endog_high,TRAIN,TEST,i,'endog','high')

    print('Chunk (ID: '+str(chunk)+') iteration '+str(chunk_iteration)+': Completed ARIMA for Endog - High Alarms.')

    #####################################
    # DARTS ARIMA for Exog. Variable Low
    #####################################

    current_train_list_exog_low = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TRAIN_LIST_"+str(resampling_method_for_exog_low)]
    current_test_list_exog_low = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TEST_LIST_"+str(resampling_method_for_exog_low)]

    #adapt for darts - convert to TimeSeries - Train List Exog - low
    current_train_list_timeseries_exog_low = convert_list_to_TimeSeries(current_train_list_exog_low)

    # we can not simply add the 'current_test_list_exog_low' as exog in arimax.predict as we do not have that info at the current point in time
    # we need to first predict (with e.g. common arima) the value for current_test_list_exog_low and add this as exog

    #predict exog value for TEST
    arima_for_exog_low = ModifiedAutoARIMA(seasonal=False, suppress_warnings=True, error_action='ignore')
    arima_for_exog_low.fit(current_train_list_timeseries_exog_low)
    forecast_arima_exog_low = arima_for_exog_low.predict(TEST)

    #reconvert TimeSeries Forecast to Series with the first index being 'last index of train' +1
    forecast_arima_exog_low_series = convert_TimeSeries_to_list_for_steady_train(forecast_arima_exog_low,TRAIN,TEST,i,'exog','low')

    print('Chunk (ID: '+str(chunk)+') iteration '+str(chunk_iteration)+': Completed ARIMA for Exog - Low Alarms.')

        
    ###############################
    # DARTS ARIMAX for Low Alarms
    ###############################

    current_train_list_endog_low = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TRAIN_LIST_"+str(resampling_method_for_endog_low)] 
    current_test_list_endog_low = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TEST_LIST_"+str(resampling_method_for_endog_low)]
    

    #adapt for darts - convert to TimeSeries - Train List - low
    current_train_list_timeseries_endog_low = convert_list_to_TimeSeries(current_train_list_endog_low)

    #perform darts arimax low
    arimax_for_endog_low = ModifiedAutoARIMA(seasonal=False, suppress_warnings=True, error_action='ignore')
    arimax_for_endog_low.fit(current_train_list_timeseries_endog_low, exog=current_train_list_timeseries_exog_low)
    forecast_arimax_endog_low = arimax_for_endog_low.predict(TEST, exog=forecast_arima_exog_low)

    #reconvert TimeSeries Forecast to Series with the first index being 'last index of train' +1
    forecast_arimax_endog_low_series = convert_TimeSeries_to_list_for_steady_train(forecast_arimax_endog_low,TRAIN,TEST,i,'endog','low')

    print('Chunk (ID: '+str(chunk)+') iteration '+str(chunk_iteration)+': Completed ARIMA for Exog - Low Alarms.')
                
    # extract threshold series 
    threshold_high_for_test_list = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["THRESHOLD_HIGH_FOR_TEST_LIST"]
    threshold_low_for_test_list = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["THRESHOLD_LOW_FOR_TEST_LIST"]
    
    # write to dict_of_chunk_series_with_forecast_df dataframe
       # give test lists new name in case resampling input for high and low is the same
    resampling_method_for_high_lower = str.lower(resampling_method_for_endog_high)
    resampling_method_for_low_lower = str.lower(resampling_method_for_endog_low)
    current_test_list_endog_high_renamed = pd.Series(current_test_list_endog_high, index=[*range(i+TRAIN,i+TRAIN+TEST,1)], name="test_list_high_"+str(resampling_method_for_high_lower))
    current_test_list_endog_low_renamed = pd.Series(current_test_list_endog_low, index=[*range(i+TRAIN,i+TRAIN+TEST,1)], name="test_list_low_"+str(resampling_method_for_low_lower))
    all_dict_lists_as_df = pd.concat([current_test_list_endog_high_renamed,forecast_arimax_endog_high_series,current_test_list_exog_high,forecast_arima_exog_high_series,threshold_high_for_test_list,current_test_list_endog_low_renamed,forecast_arimax_endog_low_series,current_test_list_exog_low,forecast_arima_exog_low_series,threshold_low_for_test_list],axis=1)
    
    return all_dict_lists_as_df
    
    







    










