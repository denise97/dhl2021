import pandas as pd
import pmdarima as pm
import numpy as np
import copy
import pickle
import time

path_to_data = '/hpi/fs00/share/MPSS2021BA1/data/'

starttime = time.time()
print('Start reading the input file.')

TRAIN=12
# Run for the one or the other
# read dict where train size is TRAIN
input_file = open(str(path_to_data) + 'dict_of_chunk_series_with_test_and_train_' + str(TRAIN) + '.pickle', 'rb')
dict_of_chunk_series_with_test_and_train = pickle.load(input_file)
input_file.close()

# read dict where train size is expanding
# second_input_file = open(str(path_to_data) + 'dict_of_chunk_series_with_expanding_test_and_steady_train.pickle', 'rb')
# dict_of_chunk_series_with_test_and_train = pickle.load(second_input_file)
# second_input_file.close()

endtime = round(((time.time() - starttime) / 60), 5)
print('Reading of the input file completed after '+str(endtime)+' minutes.')

# Expand the previously created dictionary (dict_of_chunk_series_with_test_and_train) to also hold the prediction series next to the train and the test series (and threshold values for test)

runningtime = round(((time.time() - starttime) / 60), 5)
print('Starting setting up dictionaries. Running time '+str(runningtime)+' min.')

dict_of_chunk_series_with_test_and_train_and_forecast = copy.deepcopy(dict_of_chunk_series_with_test_and_train)
dict_of_chunk_series_with_forecast_df = {}
accuracy_dict_for_chunk_iterations = {}

runningtime = round(((time.time() - starttime) / 60), 5)
print('Completed setting up dictionaries. Running time '+str(runningtime)+' min.')

for i, chunk in enumerate(dict_of_chunk_series_with_test_and_train_and_forecast):
    dict_of_chunk_series_with_forecast_df[chunk] = {}
    accuracy_dict_for_chunk_iterations[chunk] = {}

    runningtime = round(((time.time() - starttime) / 60), 5)
    print('CHUNK '+str(i)+': START. Running time '+str(runningtime)+' min.')
    
    for i, chunk_iteration in enumerate(dict_of_chunk_series_with_test_and_train_and_forecast[chunk]):
        
        TEST = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TEST_LIST_MEDIAN"].size
        tp, tn, fp, fn = 0, 0, 0, 0
        accurracy_matrix_df_for_chunk_iteration = pd.DataFrame(columns=["TP","FN","FP","TN"])

        ########################
        # ARIMAX for High Alarms
        ########################

        current_train_list_high = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TRAIN_LIST_MAX"] 
        current_test_list_high = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TEST_LIST_MAX"]
        current_train_list_exog_high = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TRAIN_LIST_MEDIAN"].values.reshape(-1, 1)
        current_test_list_exog_high = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TEST_LIST_MEDIAN"].values.reshape(-1, 1)
        arimax_high = pm.auto_arima(current_train_list_high, X=current_train_list_exog_high, suppress_warnings=True, error_action='ignore')
        forecast_arimax_high = pd.Series(arimax_high.predict(TEST, X=current_test_list_exog_high), index=[*range(i+TRAIN,i+TRAIN+TEST,1)], name="forecast_list_arimax_high")
        dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["FORECAST_LIST_ARIMAX_HIGH"] = forecast_arimax_high
        
        runningtime = round(((time.time() - starttime) / 60), 5)
        print('Chunk iteration '+str(i)+': Completed ARIMAX - High Alarms. Running time '+str(runningtime)+' min.')

        ########################
        # ARIMAX for Low Alarms
        ########################
        
        current_train_list_low = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TRAIN_LIST_MIN"] 
        current_test_list_low = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TEST_LIST_MIN"] 
        current_train_list_exog_low = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TRAIN_LIST_MEDIAN"].values.reshape(-1, 1)
        current_test_list_exog_low = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TEST_LIST_MEDIAN"].values.reshape(-1, 1)
        arimax_low = pm.auto_arima(current_train_list_low, X=current_train_list_exog_low, suppress_warnings=True, error_action='ignore')
        forecast_arimax_low = pd.Series(arimax_low.predict(TEST, X=current_test_list_exog_low), index=[*range(i+TRAIN,i+TRAIN+TEST,1)], name="forecast_list_arimax_low")
        dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["FORECAST_LIST_ARIMAX_LOW"] = forecast_arimax_low
        
        runningtime = round(((time.time() - starttime) / 60), 5)
        print('Chunk iteration '+str(i)+': Completed ARIMAX - Low Alarms. Running time '+str(runningtime)+' min.')

        # extract threshold series 
        threshold_high_for_test_list = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["THRESHOLD_HIGH_FOR_TEST_LIST"]
        threshold_low_for_test_list = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["THRESHOLD_LOW_FOR_TEST_LIST"]
        
        # write to dict_of_chunk_series_with_forecast_df dataframe
        all_dict_lists_as_df = pd.concat([current_test_list_high,forecast_arimax_high,threshold_high_for_test_list,current_test_list_low,forecast_arimax_low,threshold_low_for_test_list],axis=1)
        dict_of_chunk_series_with_forecast_df[chunk][chunk_iteration] = all_dict_lists_as_df

        ##############################################
        # Add information whether alarm was triggered
        ##############################################

        df_for_chunk_iteration = dict_of_chunk_series_with_forecast_df[chunk][chunk_iteration]
        
        # True alarms
        df_for_chunk_iteration['high_alarm_triggered'] = np.where(df_for_chunk_iteration['test_list_max'] > df_for_chunk_iteration['threshold_high_for_test_list'] ,1,0)
        df_for_chunk_iteration['low_alarm_triggered'] = np.where(df_for_chunk_iteration['test_list_min'] < df_for_chunk_iteration['threshold_low_for_test_list'] ,1,0)
                
        # ARIMAX forecast
        df_for_chunk_iteration['high_alarm_triggered_forecast_arimax'] = np.where(df_for_chunk_iteration['forecast_list_arimax_high'] > df_for_chunk_iteration['threshold_high_for_test_list'],1,0)
        df_for_chunk_iteration['low_alarm_triggered_forecast_arimax'] = np.where(df_for_chunk_iteration['forecast_list_arimax_low'] < df_for_chunk_iteration['threshold_low_for_test_list'],1,0)
        #write to dict_of_chunk_series_with_forecast_and_alarm_df dataframe
        dict_of_chunk_series_with_forecast_df[chunk][chunk_iteration] = df_for_chunk_iteration
        
        runningtime = round(((time.time() - starttime) / 60), 5)
        print('Chunk iteration '+str(i)+': Completed Alarm Identification. Running time '+str(runningtime)+' min.')

        ##########################################
        # Calculate Confusion Matrix - High Alarms
        ##########################################

        # select true high alarms triggered
        column_index_of_high_alarm_triggered = df_for_chunk_iteration.columns.get_loc("high_alarm_triggered")

        # select predicted high alarms
        column_index_of_high_alarm_triggered_forecast_arimax = df_for_chunk_iteration.columns.get_loc("high_alarm_triggered_forecast_arimax")
        
        # create df with bot as column
        high_alarms = df_for_chunk_iteration.iloc[0:,[column_index_of_high_alarm_triggered,column_index_of_high_alarm_triggered_forecast_arimax]]
        
        for row_in_high_alarms in high_alarms.iterrows():

            if row_in_high_alarms[1][0] and row_in_high_alarms[1][1]:
                tp +=1
                # print("tp", tp)
            if row_in_high_alarms[1][0] and not row_in_high_alarms[1][1]:
                fn +=1
                # print("fn", fn)
            if not row_in_high_alarms[1][0] and row_in_high_alarms[1][1]:
                fp +=1
                # print("fp", fp)
            if not row_in_high_alarms[1][0] and not row_in_high_alarms[1][1]:
                tn +=1
                # print("tn",tn)
        
        a_new_row = {"TP":tp,"FN":fn,"FP":fp,"TN":tn}
        a_new_row_series = pd.Series(a_new_row,name="accuracy_high_alarms_arimax")

        accurracy_matrix_df_for_chunk_iteration = accurracy_matrix_df_for_chunk_iteration.append(a_new_row_series)

        runningtime = round(((time.time() - starttime) / 60), 5)
        print('Chunk iteration '+str(i)+': Completed Confusion Matrix - High Alarms. Running time '+str(runningtime)+' min.')

        #########################################
        # Calculate Confusion Matrix - Low Alarms
        ##########################################

        # Reset tp, tn, fp, fn
        tp, tn, fp, fn = 0, 0, 0, 0

        # select true low alarms triggered
        column_index_of_low_alarm_triggered = df_for_chunk_iteration.columns.get_loc("low_alarm_triggered")
        
        # select predicted low alarms
        column_index_of_low_alarm_triggered_forecast_arimax = df_for_chunk_iteration.columns.get_loc("low_alarm_triggered_forecast_arimax")
        
        # create df with bot as column 
        low_alarms = df_for_chunk_iteration.iloc[0:,[column_index_of_low_alarm_triggered,column_index_of_low_alarm_triggered_forecast_arimax]]
        
        for row_in_low_alarms in low_alarms.iterrows():

            if row_in_low_alarms[1][0] and row_in_low_alarms[1][1]:
                tp +=1
                # print("tp", tp)
            if row_in_low_alarms[1][0] and not row_in_low_alarms[1][1]:
                fn +=1
                # print("fn", fn)
            if not row_in_low_alarms[1][0] and row_in_low_alarms[1][1]:
                fp +=1
                # print("fp", fp)
            if not row_in_low_alarms[1][0] and not row_in_low_alarms[1][1]:
                tn +=1
                # print("tn",tn)
        
        a_new_row = {"TP":tp,"FN":fn,"FP":fp,"TN":tn}
        a_new_row_series = pd.Series(a_new_row,name="accuracy_low_alarms_arimax")
        
        accurracy_matrix_df_for_chunk_iteration = accurracy_matrix_df_for_chunk_iteration.append(a_new_row_series)

        runningtime = round(((time.time() - starttime) / 60), 5)
        print('Chunk iteration '+str(i)+': Completed Confusion Matrix - Low Alarms. Running time '+str(runningtime)+' min.')
        
        # Write confusion matrix into dictionary
        accuracy_dict_for_chunk_iterations[chunk][chunk_iteration] = accurracy_matrix_df_for_chunk_iteration

    runningtime = round(((time.time() - starttime) / 60), 5)
    print('CHUNK '+str(i)+': Completed chunk. Running time '+str(runningtime)+' min.')
    print('--------------------')
        
endtime = round(((time.time() - starttime) / 60), 5)
print('DONE')
print('Completed in '+str(endtime)+' minutes.')
