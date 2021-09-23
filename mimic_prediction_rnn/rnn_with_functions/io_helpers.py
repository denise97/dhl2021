import numpy as np
import os
import pickle5 as pickle


# Checks validity of configuration parameters from configs.py
def verify_configs(configs):
    if len(configs.model_types) <= 0:
        raise ValueError('No model types are configured')

    configs.model_types = [model_type.upper() for model_type in configs.model_types]
    if not set(configs.model_types) <= {'RNN', 'LSTM', 'GRU'}:
        raise ValueError('Configured model types are not valid')

    if len(configs.parameters) <= 0:
        raise ValueError('No parameters are configured')

    configs.parameters = [parameter.lower() for parameter in configs.parameters]
    if not set(configs.parameters) <= {'hr', 'bp', 'o2'}:
        raise ValueError('Configured parameters are not valid')

    if not configs.scaling_method:
        raise ValueError('No scaling method is configured')

    if configs.scaling_method != np.nan:
        configs.scaling_method = configs.scaling_method.lower()
    if configs.scaling_method not in [np.nan, 'standard', 'min-max']:
        raise ValueError('Configured scaling method is not valid')

    if not (1 <= configs.prediction_data_size <= 99):
        raise ValueError('Configured prediction data set size should be between 1 and 99 percent')

    if not (isinstance(configs.n_chunks, int) and isinstance(configs.n_windows, int) and
            isinstance(configs.input_length, int) and isinstance(configs.output_length, int)):
        raise TypeError('Number of configured chunks, windows and input/output lengths must be integers')

    if configs.n_chunks <= 0 or configs.n_windows <= 0 or configs.output_length <= 0:
        raise ValueError('Number of configured chunks, windows and output length must be at least 1')

    if not isinstance(configs.with_exogenous_input, bool):
        raise TypeError('Config "with_exogenous_input" must be a boolean')


def get_script_path(configs):
    if configs.with_exogenous_input:
        folder_suffix = 'withExo'
    else:
        folder_suffix = 'onlyEndo'

    if configs.scaling_method == 'standard':
        scaling = 's1'
    elif configs.scaling_method == 'min-max':
        scaling = 's2'
    else:
        scaling = 'n'

    return f'data/{configs.n_chunks}chunks_{configs.n_windows}windows_{configs.input_length}input_' \
           f'{configs.output_length}output_{configs.prediction_data_size}percent_{scaling}_{folder_suffix}'


def create_folder(folder_path):
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)


def create_folders(configs):
    path = get_script_path(configs)
    create_folder(path)

    if configs.scaling_method == 'standard':
        create_folder(f'{path}/metrics')

    if configs.scaling_method == 'min-max':
        create_folder(f'{path}/scalers')

    for folder in ['chunk_ids', 'time_series', 'training', 'prediction', 'confusion_matrices']:
        create_folder(f'{path}/{folder}')


def write_pickle_file(file_path, obj):
    file = open(file_path, 'wb')
    pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()


def read_pickle_file(file_path):
    file = open(file_path, 'rb')
    obj = pickle.load(file)
    file.close()

    return obj
