import pickle
import time
import pandas as pd
import pmdarima as pm
import numpy as np
import copy
import time
import warnings
import pyarrow as pa

path_to_data = '/hpi/fs00/share/MPSS2021BA1/data/'

starttime = time.time()
print('Start reading the input file.')

# read dict where train size is expanding
# input_file = open(str(path_to_data) + 'arima_preprocessing/dict_of_chunk_iterations_with_expanding_train_12_hr.pickle', 'rb')
# input_file = open(str(path_to_data) + 'arima_preprocessing/dict_of_chunk_iterations_with_expanding_train_12_bp.pickle', 'rb')
# input_file = open(str(path_to_data) + 'arima_preprocessing/dict_of_chunk_iterations_with_expanding_train_12_o2.pickle', 'rb')
input_file = open(str(path_to_data) + 'arima_preprocessing/dict_of_chunk_iterations_with_expanding_train_12_hr_first1000.pickle', 'rb')
# input_file = open(str(path_to_data) + 'arima_preprocessing/dict_of_chunk_iterations_with_expanding_train_12_bp_first1000.pickle', 'rb')
# input_file = open(str(path_to_data) + 'arima_preprocessing/dict_of_chunk_iterations_with_expanding_train_12_o2_first1000.pickle', 'rb')
dict_of_chunk_series_with_test_and_train = pickle.load(input_file)
input_file.close()

endtime = round(((time.time() - starttime) / 60), 5)
print('Reading of the input file completed after '+str(endtime)+' minutes.')

starttime = time.time()

# Expand the previously created dictionary (dict_of_chunk_series_with_test_and_train) to also hold the prediction series next to the train and the test series (and threshold values for test)

runningtime = round(((time.time() - starttime) / 60), 5)
print('Starting setting up dictionaries. Running time '+str(runningtime)+' min.')

dict_of_chunk_series_with_test_and_train_and_forecast = copy.deepcopy(dict_of_chunk_series_with_test_and_train)
dict_of_chunk_series_with_forecast_df = {}
accuracy_dict_for_chunk_iterations = {}
chunk_iterations_with_runtime_warning = pd.DataFrame(columns=["CHUNK_ID_FILLED_TH","ITERATION","WARNING_MSG"])

# Convert warnings to exceptions
warnings.filterwarnings('error', category=RuntimeWarning)
np.seterr(all='warn')

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
        
        tp, tn, fp, fn = 0, 0, 0, 0
        accurracy_matrix_df_for_chunk_iteration = pd.DataFrame(columns=["TP","FN","FP","TN"])

        ########################
        # ARIMA
        ########################

        current_train_list = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TRAIN_LIST_MEDIAN"] 
        current_test_list = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["TEST_LIST_MEDIAN"]

        try:
            arima = pm.auto_arima(current_train_list, seasonal=False, suppress_warnings=True, error_action='ignore')
            forecast_arima = pd.Series(arima.predict(TEST), index=[*range(TRAIN,TRAIN+TEST,1)], name="forecast_list_arima")
            dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["FORECAST_LIST_ARIMA"] = forecast_arima
            
            runningtime = round(((time.time() - starttime) / 60), 5)
            print('Chunk '+str(j)+' (ID: '+str(chunk)+') iteration '+str(chunk_iteration)+': Completed ARIMA. Running time '+str(runningtime)+' min.')

            # extract threshold series 
            threshold_high_for_test_list = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["THRESHOLD_HIGH_FOR_TEST_LIST"]
            threshold_low_for_test_list = dict_of_chunk_series_with_test_and_train_and_forecast[chunk][chunk_iteration]["THRESHOLD_LOW_FOR_TEST_LIST"]
            
            # write to dict_of_chunk_series_with_forecast_df dataframe
            all_dict_lists_as_df = pd.concat([current_test_list,threshold_high_for_test_list,threshold_low_for_test_list,forecast_arima],axis=1)
            dict_of_chunk_series_with_forecast_df[chunk][chunk_iteration] = all_dict_lists_as_df

            ##############################################
            # Add information whether alarm was triggered
            ##############################################

            df_for_chunk_iteration = dict_of_chunk_series_with_forecast_df[chunk][chunk_iteration]
            
            # True alarms
            df_for_chunk_iteration['high_alarm_triggered'] = np.where(df_for_chunk_iteration['test_list_median'] > df_for_chunk_iteration['threshold_high_for_test_list'] ,1,0)
            df_for_chunk_iteration['low_alarm_triggered'] = np.where(df_for_chunk_iteration['test_list_median'] < df_for_chunk_iteration['threshold_low_for_test_list'] ,1,0)
                    
            # ARIMA forecast
            df_for_chunk_iteration['high_alarm_triggered_forecast_arima'] = np.where(df_for_chunk_iteration['forecast_list_arima'] > df_for_chunk_iteration['threshold_high_for_test_list'],1,0)
            df_for_chunk_iteration['low_alarm_triggered_forecast_arima'] = np.where(df_for_chunk_iteration['forecast_list_arima'] < df_for_chunk_iteration['threshold_low_for_test_list'],1,0)
            # write to dict_of_chunk_series_with_forecast_and_alarm_df dataframe
            dict_of_chunk_series_with_forecast_df[chunk][chunk_iteration] = df_for_chunk_iteration
            
            runningtime = round(((time.time() - starttime) / 60), 5)
            print('Chunk '+str(j)+' (ID: '+str(chunk)+') iteration '+str(chunk_iteration)+': Completed Alarm Identification. Running time '+str(runningtime)+' min.')

            ##########################################
            # Calculate Confusion Matrix - High Alarms
            ##########################################

            # select true high alarms triggered
            column_index_of_high_alarm_triggered = df_for_chunk_iteration.columns.get_loc("high_alarm_triggered")

            # select predicted high alarms
            column_index_of_high_alarm_triggered_forecast_arima = df_for_chunk_iteration.columns.get_loc("high_alarm_triggered_forecast_arima")
            
            # create df with bot as column
            high_alarms = df_for_chunk_iteration.iloc[0:,[column_index_of_high_alarm_triggered,column_index_of_high_alarm_triggered_forecast_arima]]
            
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
            a_new_row_series = pd.Series(a_new_row,name="accuracy_high_alarms_arima")

            accurracy_matrix_df_for_chunk_iteration = accurracy_matrix_df_for_chunk_iteration.append(a_new_row_series)

            runningtime = round(((time.time() - starttime) / 60), 5)
            print('Chunk '+str(j)+' (ID: '+str(chunk)+') iteration '+str(chunk_iteration)+': Completed Confusion Matrix - High Alarms. Running time '+str(runningtime)+' min.')

            #########################################
            # Calculate Confusion Matrix - Low Alarms
            #########################################

            # Reset tp, tn, fp, fn
            tp, tn, fp, fn = 0, 0, 0, 0

            # select true low alarms triggered
            column_index_of_low_alarm_triggered = df_for_chunk_iteration.columns.get_loc("low_alarm_triggered")
            
            # select predicted low alarms
            column_index_of_low_alarm_triggered_forecast_arima = df_for_chunk_iteration.columns.get_loc("low_alarm_triggered_forecast_arima")
            
            # create df with bot as column 
            low_alarms = df_for_chunk_iteration.iloc[0:,[column_index_of_low_alarm_triggered,column_index_of_low_alarm_triggered_forecast_arima]]
            
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
            a_new_row_series = pd.Series(a_new_row,name="accuracy_low_alarms_arima")
            
            accurracy_matrix_df_for_chunk_iteration = accurracy_matrix_df_for_chunk_iteration.append(a_new_row_series)

            runningtime = round(((time.time() - starttime) / 60), 5)
            print('Chunk '+str(j)+' (ID: '+str(chunk)+') iteration '+str(chunk_iteration)+': Completed Confusion Matrix - Low Alarms. Running time '+str(runningtime)+' min.')
            
            # Write confusion matrix into dictionary
            accuracy_dict_for_chunk_iterations[chunk][chunk_iteration] = accurracy_matrix_df_for_chunk_iteration
            
        except RuntimeWarning as rw:
            
            rw_string = str(rw)
            a_new_row = {"CHUNK_ID_FILLED_TH":chunk,"ITERATION":chunk_iteration,"WARNING_MSG":rw_string}
            a_new_row_series = pd.Series(a_new_row)
            chunk_iterations_with_runtime_warning = chunk_iterations_with_runtime_warning.append(a_new_row_series, ignore_index = True)
            #chunk_iterations_with_runtime_warning.to_parquet(str(path_to_data)+'chunk_iterations_with_runtime_warning_for_arima_expanding_12_hr.parquet', engine='pyarrow')
            #chunk_iterations_with_runtime_warning.to_parquet(str(path_to_data)+'chunk_iterations_with_runtime_warning_for_arima_expanding_12_bp.parquet', engine='pyarrow')
            #chunk_iterations_with_runtime_warning.to_parquet(str(path_to_data)+'chunk_iterations_with_runtime_warning_for_arima_expanding_12_o2.parquet', engine='pyarrow')
            chunk_iterations_with_runtime_warning.to_parquet(str(path_to_data)+'chunk_iterations_with_runtime_warning_for_arima_expanding_12_hr_first1000.parquet', engine='pyarrow')
            #chunk_iterations_with_runtime_warning.to_parquet(str(path_to_data)+'chunk_iterations_with_runtime_warning_for_arima_expanding_12_bp_first1000.parquet', engine='pyarrow')
            #chunk_iterations_with_runtime_warning.to_parquet(str(path_to_data)+'chunk_iterations_with_runtime_warning_for_arima_expanding_12_o2_first1000.parquet', engine='pyarrow')
            print("RUNTIME WARNING DETECTED:")
            print(a_new_row_series)
            
    runningtime = round(((time.time() - starttime) / 60), 5)
    print('Chunk '+str(j)+' (ID: '+str(chunk)+'): Completed chunk. Running time '+str(runningtime)+' min.')
    print('--------------------')

endtime = round(((time.time() - starttime) / 60), 5)
print('DONE')
print('Completed in '+str(endtime)+' minutes.')

print('Starting saving dictionary.')
# output_file = open(str(path_to_data)+'accuracy_dict_for_chunk_iterations_arima_expanding_12_hr.pickle', 'wb')
# output_file = open(str(path_to_data)+'accuracy_dict_for_chunk_iterations_arima_expanding_12_bp.pickle', 'wb')
# output_file = open(str(path_to_data)+'accuracy_dict_for_chunk_iterations_arima_expanding_12_o2.pickle', 'wb')
output_file = open(str(path_to_data)+'accuracy_dict_for_chunk_iterations_arima_expanding_12_hr_first1000.pickle', 'wb')
# output_file = open(str(path_to_data)+'accuracy_dict_for_chunk_iterations_arima_expanding_12_bp_first1000.pickle', 'wb')
# output_file = open(str(path_to_data)+'accuracy_dict_for_chunk_iterations_arima_expanding_12_o2_first1000.pickle', 'wb')
pickle.dump(accuracy_dict_for_chunk_iterations, output_file)
output_file.close()
print('Completed saving dictionary.')
