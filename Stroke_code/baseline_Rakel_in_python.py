import pandas as pd
# import tensorflow as tf
import os
import random
import numpy as np
from skmultilearn.ensemble import RakelD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_score, recall_score
from Stroke_code.utils import precision_recall_graph


LABELS = ['trunk-flex', 'scapular-e', 'scapular-r', 'shoulder-flex', 'elbow-flex', 'distal-dys-syn']

# Seed value
seed_value = 42

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)


def reconstructed_experiment():
    """
    This code is based on the R code of the algorithm (from the paper).
    It runs RAKEL with the features after the feature selection.
    * the experiment results is not same as the paper results, but also the indices list is not full so it probably
     influenced the results.
     in the paper there are 156 features.
     here we have 139 features.
    """
    data_path = '../compensation_detection/Data/Soroka/raw_data'
    data = pd.read_csv(f'{data_path}/data.csv')

    len_ts = 3138
    num_of_labels = 6

    # read shir list of indices
    ts_fresh_selected_features = pd.read_csv(f'{data_path}/tsfeature_indexs_432.csv').T.values - 1
    matlab_selected_features = pd.read_csv(f'{data_path}/matlab_features_5432.csv').T.values - 1
    ts_fresh_selected_features = np.sort(ts_fresh_selected_features)
    # matlab features indices are starting from len_ts in the data set
    matlab_selected_features = np.sort(matlab_selected_features) + len_ts

    all_features = np.concatenate([ts_fresh_selected_features, matlab_selected_features], axis=1)[0]

    labels_indexs = list(range(data.shape[1] - 7, data.shape[1] - 1))
    labels_indexs = labels_indexs[:num_of_labels]
    total_sum_movements = data.shape[0]
    n_subjects = int(total_sum_movements / 18)

    # indexs_per_each_subject = [[] for sub in n_subjects]
    data = data.fillna(0)  # dealing with Missing Data!

    labels = data.iloc[:, labels_indexs].values
    data1_x = data.iloc[:, all_features]  # cannot handle catagorial predictors
    data1_y = data.iloc[:, labels_indexs]
    predictions = np.zeros((total_sum_movements, num_of_labels))

    # print("feature number:")
    # print(q)

    for i in range(n_subjects):
        print("subject number: ")
        print(i)
        x_train_agg = data1_x.drop(list(range(i*18, i*18+18)), axis=0)
        x_test_agg = data1_x.iloc[list(range(i*18, i*18+18))]

        y_train = data1_y.drop(list(range(i * 18, i * 18 + 18)), axis=0)
        y_test = data1_y.iloc[list(range(i * 18, i * 18 + 18))]

        clf = RakelD(
            base_classifier=RandomForestClassifier(n_estimators=1000),
            labelset_size=2
        )

        clf.fit(x_train_agg, y_train)
        pred = clf.predict_proba(x_test_agg).toarray()

        predictions[i*18:i*18+18, :num_of_labels] = pred

    # ###################### Evaluation Metrics #################################
    #
    Hamming_loss = np.zeros(num_of_labels)

    # Hamming loss
    for i in range(num_of_labels):
        y_pred = predictions[:, i]
        y_true = labels[:, i]

        Hamming_loss[i] = hamming_loss(y_true, np.round(y_pred))
    print(f'Hamming_loss {np.mean(Hamming_loss)}')

    ### recall precision
    print(f'mean_precision_score micro {precision_score(labels, np.round(predictions), average="micro")}')
    print(f'mean_precision_score macro {precision_score(labels, np.round(predictions), average="macro")}')
    print(f'mean_recall_score micro {recall_score(labels, np.round(predictions), average="micro")}')
    print(f'mean_recall_score macro {recall_score(labels, np.round(predictions), average="macro")}')

    precision_recall_graph(labels, predictions, LABELS[:num_of_labels], method='baseline_original_features')


def read_matlab_data():
    """
    read matlab data set (table is created by running matlab code and choosing all sensors)
    :return: datax, datay - features and labels
    """
    data_path = '../compensation_detection/Data/Soroka'
    # aggregated features from matlab
    data = pd.read_excel(f'{data_path}/raw_data/matlab_features_full.xlsx')
    data = data[data.Subjects.str.startswith('P')].reset_index(drop=True)
    data['height'] = data['height'].replace({1: 'H', 2: 'M', 3: 'L'})
    data['empty'] = data['empty'].replace({1: 'E', 2: 'F'})
    data['key'] = data['Subjects'].str[1:] + '_' + data['height'] + data["moveNum"].astype(str) + data['empty']
    data_labels = pd.read_csv(f'{data_path}/code_Products/patients_target.csv', index_col=0).iloc[:, -6:]
    data = data.merge(data_labels, how='left', left_on='key', right_index=True)
    data.drop(['classification_thesis', 'Subjects', 'id', 'height', 'empty', 'moveNum', 'key'], axis=1, inplace=True)
    data = data.dropna(subset=LABELS, axis=0)
    datax = data.drop(LABELS,  axis=1)
    datax = datax.fillna(0)
    datay = data.loc[:, LABELS]
    return datax, datay


# X, Y = read_matlab_data()
# feature_selection_multi_label(X, Y)
reconstructed_experiment()