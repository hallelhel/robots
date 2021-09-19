import os
import re
import numpy as np
import pandas as pd
from scipy import signal


def compute_velocity(x, y, z):
    """
    usually calculating using wrist signal
    :param x: x time series signal
    :param y: y time series signal
    :param z: z time series signal
    :return: velocity time series vector with the same length as input
    """

    b, a = signal.butter(1, 20 / 120)
    del_t = 1 / 120

    x_filtered = signal.filtfilt(b, a, x)
    y_filtered = signal.filtfilt(b, a, y)
    z_filtered = signal.filtfilt(b, a, z)

    x_filtered = np.abs(np.gradient(x_filtered, del_t))
    y_filtered = np.abs(np.gradient(y_filtered, del_t))
    z_filtered = np.abs(np.gradient(z_filtered, del_t))

    velocity = np.sqrt(np.square(x_filtered) + np.square(y_filtered) + np.square(z_filtered))

    return velocity


def read_matlab_time_cut(time_series_folder, path_to_data):
    """
    read th1 - th4 time cuts. if needed for the algorithm - should be implemented
    :return: indices to cut each frame
    """
    agg_matlab = pd.read_excel(f'{time_series_folder}\\matlab_features_full.xlsx')

    agg_matlab['height'] = agg_matlab['height'].replace({1: 'H', 2: 'M', 3: 'L'})
    agg_matlab['empty'] = agg_matlab['empty'].replace({1: 'E', 2: 'F'})
    agg_matlab['key'] = agg_matlab['Subjects'] + '_' + agg_matlab['height'] + agg_matlab["moveNum"].astype(
        str) + agg_matlab['empty']
    time_features = ['th1F', 'th4F']

    new_time_featurs = []
    for feature in time_features:
        agg_matlab[feature[:-1] + 'V'] = ((agg_matlab[feature] -1) * (1.2)).astype(int)
        new_time_featurs += [feature[:-1] + 'V']
    return agg_matlab[['key'] + new_time_featurs]


def read_stroke_data(task_column, time_series_folder='Soroka\\Patients\\organized_train_data',
                     path_to_data='',
                     with_controls=False):
    """
   read raw data into list of all the time series data df
    :param task_column:
    :param with_controls: whether to use controllers data
    :param path_to_data:
    :param time_series_folder: folder with raw data
    :return: list of df for each event
    """
    subjects_folders = os.listdir(f'{time_series_folder}')
    subjects = [folder for folder in subjects_folders if 'P' in folder]

    if with_controls:
        subjects += [folder for folder in subjects_folders if 'C' in folder]

    # # filter time series between th1 to th4
    # start_end_time_map = read_matlab_time_cut(time_series_folder, path_to_data)

    dic = {}
    index_trip = 0

    for subject in subjects:
        path_raw_data = f'{time_series_folder}\\{subject}'
        subject_files = [i for i in os.listdir(path_raw_data) if os.path.isfile(os.path.join(path_raw_data, i))]
        for file in subject_files:
            data = pd.read_csv(f'{path_raw_data}\\{file}')
            data = data.apply(pd.to_numeric, errors='coerce')
            data = data.dropna(axis=0, how='all')

            data[task_column] = subject + '_' + ''.join([x[0] if not x.isdigit() else str(int(x))
                                                          for x in re.findall(f'{subject}_(.+).csv',
                                                                              file)[0].split('_')]).upper()

            # computes velocity if needed
            # data['velocity'] = compute_velocity(data['wrist_X'], data['wrist_Y'], data['wrist_Z'])

            # # filter time series between th1 to th4
            # start_end_by_key = start_end_time_map.loc[start_end_time_map['key'] == data.loc[0, event_column]]
            # data = data.iloc[start_end_by_key['th1V'].values[0]:start_end_by_key['th4V'].values[0]]

            data['idx'] = index_trip
            data['PatientID'] = subject
            dic[index_trip] = data
            index_trip += 1

    return list(dic.values())


def missing_values(X, linear_filling):
    """
    fill missing values by linear method
    :param X: Data set
    :param linear_filling: list of columns to fill by linear filling
    :return:  X after linear filling
    """
    linear_filling = [filling for filling in linear_filling if filling in X.columns]
    # filling continues ( straight line between two point)
    X[linear_filling] = X[linear_filling].interpolate(method='linear', limit_direction='both', axis=0)

    return X


def handle_nan_vals(X):
    """
    fill missing values by mean value
    :param X: Data
    :return: X with filled missing values
    """
    X = X.apply(lambda x: x.fillna(x.mean()))  # fill nan with mean val

    return X


def rm_outliers(final_df, column_list):
    """
    Remove outliers from the data by IQR computation
    :param final_df: Data set
    :param column_list: list of columns to find outliers in them
    :return: final_df without outliers
    """
    for column in column_list:
        if len(final_df.loc[:, column].dropna().unique()) > 10:
            q3 = np.nanpercentile(final_df.loc[:, column], 75)
            q1 = np.nanpercentile(final_df.loc[:, column], 25)

            max = q3 + (q3 - q1)*1.5
            min = q1 - (q3 - q1)*1.5

            final_df.loc[final_df.loc[:, column] > max, column] = max
            final_df.loc[final_df.loc[:, column] < min, column] = min

    return final_df


def preprocess(column_list, label, task, handle_nan=False, time_series_folder='',
               with_controls=False):
    """
    preprocess the data before feature creation
    :param time_series_folder: path to data
    :param task: Unique label of the specific task
    :param column_list: list of columns to extract
    :param label: patient number label
    :param handle_nan: boolean if True - it will handle missing values
    :param with_controls: boolean. if true- will also use control group data.
    :return: data frame after preprocessing
    """

    final_df = pd.DataFrame()
    x_train = read_stroke_data(time_series_folder=time_series_folder,
                               task_column=task, with_controls=with_controls)

    for i in range(len(x_train)):
        if handle_nan:
            x_train[i] = missing_values(x_train[i], column_list)
        final_df = final_df.append(x_train[i], ignore_index=True)

    final_df = final_df[column_list + [task, label, 'idx']]
    if handle_nan:
        # final_df = rm_outliers(final_df, column_list)
        final_df.loc[:, column_list] = handle_nan_vals(final_df.loc[:, column_list])
        # final_df.loc[:, column_list] = final_df.loc[:, column_list].fillna(0)

    final_df.dropna(axis=1, how='all', inplace=True)

    return final_df[column_list + [label, task, 'idx']]



