import os
import csv
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

choice = ['wrist', 'index', 'thumb', 'data_cup', 'shoulder', 'elbow', 's1', 's2', 'hum', 'for', 'rad',
          'velocity', 'force']
list_columns = ['wrist', 'index', 'thumb', 'Marker1', 'shoulder', 'elbow', 'sternum 1', 'sternum 2', 'humerus',
                'forearm', 'radial']


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


def read_data(data_path, output_path):
    """
    read raw data and transforme it to convinient csv files. the input and the output is one csv file for each experiment.
    :param data_path: path to raw data
    :param output_path: path to output directory. if not exist it will be created.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    count_raws = []
    count = 0
    subjects = [name for name in os.listdir(data_path) if
                os.path.isdir(os.path.join(data_path, name)) and (name.startswith('P') or name.startswith('C'))]

    for name_subject in subjects:  # loop over all subjects
        path_raw_data = f'{data_path}\\{name_subject}\\raw_data'

        if not os.path.exists(path_raw_data):
            path_raw_data = f'{data_path}\\{name_subject}'
        all_motions_per_subject = [i for i in os.listdir(path_raw_data) if os.path.isfile(os.path.join(path_raw_data, i))]

        # loop over all motions of the i'th subject
        for j in range(len(all_motions_per_subject)):
            patient_df = pd.DataFrame()
            curr_motion = all_motions_per_subject[j]  # the name of the current motion
            path_motion = os.path.join(path_raw_data, curr_motion)
            with open(path_motion) as csv_file:
                file = csv.reader(csv_file, delimiter=',')
                for i, row in enumerate(file):
                    if i == 0:
                        # take all  marker names from xls file of the current motion (with the same order)
                        marker_names = row
                        break
            data = pd.read_csv(f'{path_motion}', skiprows=[0, 1, 2, 3, 4, 5], index_col=0)

            data = data.rename(columns={'Time (Seconds)': 'Time'})
            patient_df = pd.concat([patient_df, data['Time']], axis=1)
            count_raws.append(data.shape[0])
            count += 1
            for idx, movement in enumerate(list_columns):
                range_motion = [i - 1 for i, x in enumerate(marker_names) if movement in x]

                if len(range_motion) == 0 and choice[idx] == 'data_cup':
                    movement = movement[:10] + '2' + movement[11:]
                    range_motion = [i - 1 for i, x in enumerate(marker_names) if movement == x]

                if len(range_motion) == 0:
                    print(movement, name_subject)
                    continue

                if len(range_motion) > 4:
                    print(movement, name_subject, 'note: more then 3 columns')

                sub_data = data.iloc[:, range_motion[:3]]
                sub_data.columns = [choice[idx] + '_X', choice[idx] + '_Y', choice[idx] + '_Z']
                patient_df = pd.concat([patient_df, sub_data], axis=1)

                if not os.path.exists(os.path.join(output_path, name_subject)):
                    os.makedirs(os.path.join(output_path, name_subject))

            name_csv_file = os.path.join(output_path, name_subject,
                                         f'{curr_motion}')

            patient_df.to_csv(f'{name_csv_file}', index=False)

            # plt.plot(compute_velocity(patient_df['wrist_X'], patient_df['wrist_Y'], patient_df['wrist_Z']))
        # plt.show()
# in_patient_path = '../compensation_detection/Data/Soroka/raw_data/Patients'
# in_control_path = '../compensation_detection/Data/Soroka/raw_data/Controls'
# out_path = '../compensation_detection/Data/Soroka/organized_train_data'

# read_data(in_patient_path, out_path)
# read_data(in_control_path, out_path)


# To run on a test set:
# in_test_path = '../compensation_detection/Data/Soroka/raw_data/Test_Patients'
# out_test_path = '../compensation_detection/Data/Soroka/organized_test_data'
# read_data(in_test_path, out_test_path)


# TODO: add force code if necessary

